#include "sttt_c_api.h"
#include "sttt_core.hpp"
#include "sttt_search.hpp"
#include <thread>
#include <mutex>
#include <exception>
#include <vector>
#include <atomic>

using namespace sttt;

static_assert(sizeof(CBoardState) == sizeof(BoardState), "Size mismatch between CBoardState and BoardState");

static inline const BoardState* to_cpp(const CBoardState* s) {
    return reinterpret_cast<const BoardState*>(s);
}

static inline BoardState* to_cpp(CBoardState* s) {
    return reinterpret_cast<BoardState*>(s);
}

extern "C" {

void sttt_init_state(CBoardState* state) {
    *to_cpp(state) = BoardState::initial();
}

void sttt_from_raw(CBoardState* state, const int8_t* cells, const int8_t* boards, int8_t turn, int8_t forced, int8_t result) {
    BoardState* s = to_cpp(state);
    std::memset(s, 0, sizeof(BoardState));

    for (int b = 0; b < 9; ++b) {
        for (int c = 0; c < 9; ++c) {
            int8_t v = cells[b * 9 + c];
            if (v == 1) {
                s->x_cells[b] |= (1 << c);
            } else if (v == -1) {
                s->o_cells[b] |= (1 << c);
            }
        }
    }

    for (int b = 0; b < 9; ++b) {
        int8_t bv = boards[b];
        if (bv == 1) {
            s->macro_x |= (1 << b);
        } else if (bv == -1) {
            s->macro_o |= (1 << b);
        } else if (bv == 2) {
            s->macro_draw |= (1 << b);
        }
    }

    s->turn = turn;
    s->forced = forced;
    s->result = result;
}

int sttt_legal_actions(const CBoardState* state, uint8_t* out_actions) {
    return to_cpp(state)->get_legal_actions(out_actions);
}

int sttt_is_legal(const CBoardState* state, uint8_t action) {
    return to_cpp(state)->is_legal(action) ? 1 : 0;
}

int sttt_play(const CBoardState* state, uint8_t action, CBoardState* next_state) {
    const BoardState* s = to_cpp(state);
    if (!s->is_legal(action)) {
        return 0; // Failure: illegal action
    }
    BoardState* next = to_cpp(next_state);
    *next = *s;
    next->play_inplace(action);
    return 1; // Success
}

void sttt_play_inplace(CBoardState* state, uint8_t action) {
    to_cpp(state)->play_inplace(action);
}

void sttt_get_cells(const CBoardState* state, int8_t* out_cells) {
    to_cpp(state)->to_cells(out_cells);
}

void sttt_get_boards(const CBoardState* state, int8_t* out_boards) {
    to_cpp(state)->to_boards(out_boards);
}

int8_t sttt_get_turn(const CBoardState* state) {
    return to_cpp(state)->turn;
}

int8_t sttt_get_forced(const CBoardState* state) {
    return to_cpp(state)->forced;
}

int8_t sttt_get_result(const CBoardState* state) {
    return to_cpp(state)->result;
}

int sttt_is_terminal(const CBoardState* state) {
    return to_cpp(state)->is_terminal() ? 1 : 0;
}

void sttt_encode(const CBoardState* state, float* out_features) {
    to_cpp(state)->encode(out_features);
}

void sttt_encode_batch(const CBoardState* states, int count, float* out_features) {
    for (int i = 0; i < count; ++i) {
        to_cpp(&states[i])->encode(out_features + i * 289);
    }
}

int8_t sttt_rollout(CBoardState state, uint64_t seed) {
    FastRng rng(seed);
    return rollout_game(*to_cpp(&state), rng);
}

void sttt_benchmark_rollouts(int num_games, uint64_t seed, int num_threads, double* out_elapsed, uint64_t* out_total_moves) {
    if (num_threads < 1) num_threads = 1;
    if (num_games < 0) num_games = 0;
    // More threads than games would spawn workers with zero work; cap instead so
    // benchmark_rollouts(1, 2, seed) is a valid one-thread run, not an abort.
    if (num_games > 0 && num_threads > num_games) num_threads = num_games;

    std::atomic<uint64_t> total_moves(0);
    auto t_start = std::chrono::high_resolution_clock::now();

    // An exception escaping a std::thread calls std::terminate and aborts the
    // whole interpreter. Capture instead, and rethrow on the joining thread.
    std::mutex err_mutex;
    std::exception_ptr first_error;
    auto worker = [&](int games_for_thread, uint64_t thread_seed) {
      try {
        FastRng rng(thread_seed);
        uint64_t moves = 0;
        uint8_t legal[81];
        for (int i = 0; i < games_for_thread; ++i) {
            BoardState s = BoardState::initial();
            while (s.result == RESULT_ONGOING) {
                int n = s.get_legal_actions(legal);
                if (n == 0) break;
                uint32_t a_idx = rng.next_bounded(static_cast<uint32_t>(n));
                s.play_inplace(legal[a_idx]);
                moves++;
            }
        }
        total_moves.fetch_add(moves, std::memory_order_relaxed);
      } catch (...) {
        std::lock_guard<std::mutex> lock(err_mutex);
        if (!first_error) first_error = std::current_exception();
      }
    };

    if (num_threads == 1) {
        worker(num_games, seed);
    } else {
        std::vector<std::thread> threads;
        int games_per_thread = num_games / num_threads;
        int rem = num_games % num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int g = games_per_thread + (t < rem ? 1 : 0);
            threads.emplace_back(worker, g, seed + t * 10007ULL);
        }
        for (auto& th : threads) {
            th.join();
        }
    }
    // Joined: it is now safe to surface a worker failure to the caller.
    if (first_error) {
        std::rethrow_exception(first_error);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t_end - t_start;
    *out_elapsed = diff.count();
    *out_total_moves = total_moves.load();
}

float sttt_evaluate(const CBoardState* state) {
    return evaluate_state(*to_cpp(state));
}

int sttt_search_alphabeta(const CBoardState* state, int depth, uint64_t budget) {
    AlphaBetaSearcher searcher;
    return searcher.choose_move(*to_cpp(state), depth, budget);
}

} // extern "C"
