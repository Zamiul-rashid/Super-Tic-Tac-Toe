#include "sttt_core.hpp"
#include "sttt_search.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>

using namespace sttt;
using namespace std::chrono;

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

void bench_legal_actions(int iterations = 5000000) {
    BoardState s = BoardState::initial();
    uint8_t actions[81];
    auto t0 = high_resolution_clock::now();
    uint64_t total_actions = 0;
    for (int i = 0; i < iterations; ++i) {
        total_actions += s.get_legal_actions(actions);
    }
    auto t1 = high_resolution_clock::now();
    double sec = duration<double>(t1 - t0).count();
    std::cout << "[Benchmark] Legal Move Generation: "
              << std::fixed << std::setprecision(1) << (iterations / sec / 1e6) << " million calls/sec ("
              << std::setprecision(2) << (sec * 1e9 / iterations) << " ns/call)\n";
}

void bench_play_move(int iterations = 10000000) {
    BoardState s = BoardState::initial();
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        BoardState next = s;
        next.play_inplace(40); // center move
    }
    auto t1 = high_resolution_clock::now();
    double sec = duration<double>(t1 - t0).count();
    std::cout << "[Benchmark] Move Execution (play): "
              << std::fixed << std::setprecision(1) << (iterations / sec / 1e6) << " million moves/sec ("
              << std::setprecision(2) << (sec * 1e9 / iterations) << " ns/move)\n";
}

void bench_encoding(int iterations = 1000000) {
    BoardState s = BoardState::initial();
    s.play_inplace(40);
    s.play_inplace(38);
    alignas(32) float features[289];

    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        s.encode(features);
    }
    auto t1 = high_resolution_clock::now();
    double sec = duration<double>(t1 - t0).count();
    std::cout << "[Benchmark] Neural Encoding (289 floats): "
              << std::fixed << std::setprecision(1) << (iterations / sec / 1e6) << " million states/sec ("
              << std::setprecision(2) << (sec * 1e9 / iterations) << " ns/encode)\n";
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

    std::cout << "[Benchmark] Random Rollouts (" << num_threads << " thread" << (num_threads > 1 ? "s" : "") << "):\n"
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
    std::cout << "[Benchmark] Perft Validation & Move Tree Enumeration:\n";
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
    std::cout << "[Benchmark] Alpha-Beta Search:\n";
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

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    std::cout << "=========================================================\n";
    std::cout << "      Super Tic-Tac-Toe C++ Bitboard Engine Benchmark     \n";
    std::cout << "=========================================================\n";

    bench_legal_actions();
    bench_play_move();
    bench_encoding();
    std::cout << "---------------------------------------------------------\n";
    bench_perft();
    std::cout << "---------------------------------------------------------\n";
    bench_random_rollouts(200000, 1);
    std::cout << "---------------------------------------------------------\n";
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw > 1) {
        bench_random_rollouts(1000000, hw > 10 ? 10 : hw);
        std::cout << "---------------------------------------------------------\n";
    }
    bench_alphabeta();
    std::cout << "=========================================================\n";
    return 0;
}
