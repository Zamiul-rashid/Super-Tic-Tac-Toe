#include "sttt_core.hpp"
#include "sttt_search.hpp"
#include "sttt_mcts.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <string>

using namespace sttt;
using namespace std::chrono;

// M8: every primitive loop must consume its output into `checksum`, which is
// printed. Previously bench_play_move built a state and never read it, so at
// -O3 the compiler was free to delete the loop body entirely and the reported
// ">400M moves/sec" could have been the cost of an empty loop.
//
// Positions are pre-generated and varied: timing one fixed state measures a
// perfectly predicted branch and a hot cache line, and generating positions
// inside the timed region measures the generator instead of the operation.
static std::vector<BoardState> make_positions(int count, uint64_t seed = 20260914ULL) {
    std::vector<BoardState> positions;
    positions.reserve(count);
    FastRng rng(seed);
    uint8_t legal[81];
    BoardState s = BoardState::initial();
    while ((int)positions.size() < count) {
        if (s.result != RESULT_ONGOING) { s = BoardState::initial(); continue; }
        positions.push_back(s);
        int n = s.get_legal_actions(legal);
        if (n == 0) { s = BoardState::initial(); continue; }
        s.play_inplace(legal[rng.next_bounded((uint32_t)n)]);
    }
    return positions;
}

// Repeat a timed block and report the median, so one noisy sample is not the
// result. Returns median seconds; `spread` receives min and max.
template <typename Fn>
static double timed_median(int repeats, Fn&& block, double* out_min, double* out_max) {
    std::vector<double> samples;
    samples.reserve(repeats);
    block();   // warm-up: pay page faults and cache fill outside the samples
    for (int r = 0; r < repeats; ++r) {
        auto t0 = high_resolution_clock::now();
        block();
        auto t1 = high_resolution_clock::now();
        samples.push_back(duration<double>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    *out_min = samples.front();
    *out_max = samples.back();
    size_t mid = samples.size() / 2;
    return samples.size() % 2 ? samples[mid] : (samples[mid - 1] + samples[mid]) / 2.0;
}

// Perft: recursive move path counting for validation & benchmarking
uint64_t perft(const BoardState& s, int depth) {
    if (depth == 0 || s.result != RESULT_ONGOING) {
        return 1;
    }
    uint8_t actions[81];
    int n = s.get_legal_actions(actions);
    if (n == 0) return 1;
    if (depth == 1) return n;

    uint64_t nodes = 0;
    for (int i = 0; i < n; ++i) {
        BoardState child = s;
        child.play_inplace(actions[i]);
        nodes += perft(child, depth - 1);
    }
    return nodes;
}

static const int POSITION_COUNT = 4096;
static int REPEATS = 5;

void bench_legal_actions(int iterations = 5000000) {
    std::vector<BoardState> positions = make_positions(POSITION_COUNT);
    uint8_t actions[81];
    uint64_t checksum = 0;
    double lo = 0, hi = 0;
    double sec = timed_median(REPEATS, [&]() {
        for (int i = 0; i < iterations; ++i) {
            checksum += positions[i & (POSITION_COUNT - 1)].get_legal_actions(actions) + actions[0];
        }
    }, &lo, &hi);
    std::cout << "[Benchmark][native-primitive] Legal Move Generation: "
              << std::fixed << std::setprecision(1) << (iterations / sec / 1e6) << " million calls/sec ("
              << std::setprecision(2) << (sec * 1e9 / iterations) << " ns/call, median of "
              << REPEATS << ", min " << std::setprecision(2) << (lo * 1e9 / iterations)
              << " max " << (hi * 1e9 / iterations) << " ns) checksum=" << checksum << "\n";
}

void bench_play_move(int iterations = 10000000) {
    std::vector<BoardState> positions = make_positions(POSITION_COUNT);
    // Pre-resolve one legal action per position, OUTSIDE the timed region, so
    // this measures play_inplace alone and not move generation as well.
    std::vector<uint8_t> moves(positions.size());
    {
        uint8_t legal[81];
        for (size_t i = 0; i < positions.size(); ++i) {
            int n = positions[i].get_legal_actions(legal);
            moves[i] = n ? legal[i % n] : 0;
        }
    }
    uint64_t checksum = 0;
    double lo = 0, hi = 0;
    double sec = timed_median(REPEATS, [&]() {
        for (int i = 0; i < iterations; ++i) {
            int slot = i & (POSITION_COUNT - 1);
            BoardState next = positions[slot];
            next.play_inplace(moves[slot]);
            // Consume the result: without this the whole body is dead code.
            checksum += next.macro_x + next.macro_o + (uint8_t)next.result;
        }
    }, &lo, &hi);
    std::cout << "[Benchmark][native-primitive] Move Execution (play): "
              << std::fixed << std::setprecision(1) << (iterations / sec / 1e6) << " million moves/sec ("
              << std::setprecision(2) << (sec * 1e9 / iterations) << " ns/move, median of "
              << REPEATS << ", min " << std::setprecision(2) << (lo * 1e9 / iterations)
              << " max " << (hi * 1e9 / iterations) << " ns) checksum=" << checksum << "\n";
}

void bench_encoding(int iterations = 1000000) {
    std::vector<BoardState> positions = make_positions(POSITION_COUNT);
    alignas(32) float features[289];
    uint64_t checksum = 0;
    double lo = 0, hi = 0;
    double sec = timed_median(REPEATS, [&]() {
        for (int i = 0; i < iterations; ++i) {
            positions[i & (POSITION_COUNT - 1)].encode(features);
            checksum += (uint64_t)(features[0] + features[144] + features[288]);
        }
    }, &lo, &hi);
    std::cout << "[Benchmark][native-primitive] Neural Encoding (289 floats): "
              << std::fixed << std::setprecision(1) << (iterations / sec / 1e6) << " million states/sec ("
              << std::setprecision(2) << (sec * 1e9 / iterations) << " ns/encode, median of "
              << REPEATS << ", min " << std::setprecision(2) << (lo * 1e9 / iterations)
              << " max " << (hi * 1e9 / iterations) << " ns) checksum=" << checksum << "\n";
}

void bench_random_rollouts(int num_games = 500000, int num_threads = 1) {
    std::atomic<uint64_t> total_moves(0);
    std::atomic<uint64_t> x_wins(0), o_wins(0), draws(0);

    auto t0 = high_resolution_clock::now();

    auto worker = [&](int games, uint64_t seed) {
        FastRng rng(seed);
        uint8_t legal[81];
        uint64_t local_moves = 0;
        uint64_t local_x = 0, local_o = 0, local_d = 0;

        for (int i = 0; i < games; ++i) {
            BoardState s = BoardState::initial();
            while (s.result == RESULT_ONGOING) {
                int n = s.get_legal_actions(legal);
                if (n == 0) break;
                uint32_t a_idx = rng.next_bounded(static_cast<uint32_t>(n));
                s.play_inplace(legal[a_idx]);
                local_moves++;
            }
            if (s.result == RESULT_X) local_x++;
            else if (s.result == RESULT_O) local_o++;
            else if (s.result == RESULT_DRAW) local_d++;
        }
        total_moves += local_moves;
        x_wins += local_x;
        o_wins += local_o;
        draws += local_d;
    };

    if (num_threads <= 1) {
        worker(num_games, 123456789ULL);
    } else {
        std::vector<std::thread> threads;
        int games_per_th = num_games / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(worker, games_per_th, 123456789ULL + t * 99991ULL);
        }
        for (auto& th : threads) th.join();
    }

    auto t1 = high_resolution_clock::now();
    double sec = duration<double>(t1 - t0).count();

    // Independent games across threads: aggregate throughput, NOT the latency
    // of any single move or game.
    const char* category = num_threads > 1 ? "parallel-throughput" : "selfplay-game";
    std::cout << "[Benchmark][" << category << "] Random Rollouts (" << num_threads << " thread" << (num_threads > 1 ? "s" : "") << "):\n"
              << "  Games: " << num_games << " in " << sec << "s\n"
              << "  Games/sec: " << std::fixed << std::setprecision(1) << (num_games / sec) << "\n"
              << "  Total moves: " << total_moves.load() << "\n"
              << "  Moves/sec: " << std::fixed << std::setprecision(1) << (total_moves.load() / sec)
              << " (" << (total_moves.load() / sec / 1e6) << " M moves/s)\n"
              << "  Avg moves/game: " << (double)total_moves.load() / num_games << "\n"
              << "  Outcomes: X=" << x_wins.load() << ", O=" << o_wins.load() << ", Draw=" << draws.load() << "\n";
}

void bench_perft() {
    BoardState root = BoardState::initial();
    std::cout << "[Benchmark][native-primitive] Perft Validation & Move Tree Enumeration:\n";
    for (int d = 1; d <= 5; ++d) {
        auto t0 = high_resolution_clock::now();
        uint64_t count = perft(root, d);
        auto t1 = high_resolution_clock::now();
        double sec = duration<double>(t1 - t0).count();
        std::cout << "  Depth " << d << ": " << count << " nodes in "
                  << std::setprecision(4) << sec << "s ("
                  << std::setprecision(1) << (count / sec / 1e6) << " M nodes/s)\n";
    }
}

void bench_alphabeta() {
    BoardState root = BoardState::initial();
    AlphaBetaSearcher searcher;
    std::cout << "[Benchmark][native-heuristic-search] Alpha-Beta Search:\n";
    for (int depth = 1; depth <= 6; ++depth) {
        auto t0 = high_resolution_clock::now();
        int action = searcher.choose_move(root, depth, 10000000);
        auto t1 = high_resolution_clock::now();
        double sec = duration<double>(t1 - t0).count();
        std::cout << "  Depth " << depth << ": chosen action=" << action
                  << ", nodes=" << searcher.node_count
                  << " in " << std::setprecision(4) << sec << "s ("
                  << std::setprecision(1) << (searcher.node_count / sec / 1e6) << " M nodes/s)\n";
    }
}

void bench_mcts() {
    BoardState root = BoardState::initial();
    MCTSConfig cfg;
    cfg.proofs = true;
    cfg.reuse = true;
    MCTSEngine engine(cfg);
    // No neural network is involved here: this is the built-in heuristic, and
    // it must not be read as neural self-play search throughput.
    std::cout << "[Benchmark][native-heuristic-search] C++ MCTS Engine (no neural network):\n";
    for (int sims : {1000, 5000, 20000, 50000}) {
        double lo = 0, hi = 0;
        double sec = timed_median(3, [&]() { engine.run_heuristic(root, sims, 8); }, &lo, &hi);
        // Exact proofs can end a search early, so the rate denominator is the
        // simulations actually completed, not the budget requested.
        uint64_t completed = engine.stats.completed_simulations;
        std::cout << "  " << sims << " simulations requested (batch=8): "
                  << std::setprecision(4) << sec << "s median, completed=" << completed << " ("
                  << std::fixed << std::setprecision(1) << (completed / sec) << " completed sims/s, max_depth="
                  << engine.stats.max_depth << ", min " << std::setprecision(4) << lo
                  << " max " << hi << "s)\n";
    }
}

int main(int argc, char* argv[]) {
    // --quick keeps the regression test cheap; it changes iteration counts only,
    // never what is measured or how.
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--quick") quick = true;
    }
    if (quick) REPEATS = 3;

    std::cout << "=========================================================\n";
    std::cout << "      Super Tic-Tac-Toe C++ Bitboard Engine Benchmark     \n";
    std::cout << "=========================================================\n";
    std::cout << "Categories are declared per measurement in [brackets]; a\n"
              << "heuristic search rate is not a neural self-play rate, and\n"
              << "parallel throughput is not single-move latency.\n";

    bench_legal_actions(quick ? 200000 : 5000000);
    bench_play_move(quick ? 200000 : 10000000);
    bench_encoding(quick ? 100000 : 1000000);
    std::cout << "---------------------------------------------------------\n";
    if (!quick) {
        bench_perft();
        std::cout << "---------------------------------------------------------\n";
    }
    bench_random_rollouts(quick ? 5000 : 200000, 1);
    std::cout << "---------------------------------------------------------\n";
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw > 1 && !quick) {
        bench_random_rollouts(1000000, hw > 10 ? 10 : hw);
        std::cout << "---------------------------------------------------------\n";
    }
    if (!quick) {
        bench_alphabeta();
        std::cout << "---------------------------------------------------------\n";
    }
    bench_mcts();
    std::cout << "=========================================================\n";
    return 0;
}
