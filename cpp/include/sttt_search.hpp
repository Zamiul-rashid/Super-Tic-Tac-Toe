#pragma once

#include "sttt_core.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>

namespace sttt {

// Evaluate lines value for 9 cells
inline float eval_local_board(uint16_t own, uint16_t opp) {
    constexpr float weights[4] = {0.0f, 0.15f, 1.0f, 4.0f};
    float val = 0.0f;
    for (uint16_t mask : WIN_MASKS) {
        int own_cnt = __builtin_popcount(own & mask);
        int opp_cnt = __builtin_popcount(opp & mask);
        if (opp_cnt == 0) {
            val += weights[own_cnt];
        }
        if (own_cnt == 0) {
            val -= weights[opp_cnt];
        }
    }
    return val;
}

// Evaluate macro board
inline float eval_macro_board(uint16_t own, uint16_t opp, uint16_t draw) {
    constexpr float weights[4] = {0.0f, 3.0f, 16.0f, 100.0f};
    float val = 0.0f;
    for (uint16_t mask : WIN_MASKS) {
        if (draw & mask) {
            continue; // Drawn board blocks winning line
        }
        int own_cnt = __builtin_popcount(own & mask);
        int opp_cnt = __builtin_popcount(opp & mask);
        if (opp_cnt == 0) {
            val += weights[own_cnt];
        }
        if (own_cnt == 0) {
            val -= weights[opp_cnt];
        }
    }
    return val;
}

// Full heuristic evaluation matching sttt.bots.value()
inline float evaluate_state(const BoardState& s) {
    if (s.result != RESULT_ONGOING) {
        return static_cast<float>(s.result * s.turn);
    }

    uint16_t my_macro = (s.turn == 1) ? s.macro_x : s.macro_o;
    uint16_t opp_macro = (s.turn == 1) ? s.macro_o : s.macro_x;
    uint16_t draw_macro = s.macro_draw;

    float score = eval_macro_board(my_macro, opp_macro, draw_macro);
    int my_boards = __builtin_popcount(my_macro);
    int opp_boards = __builtin_popcount(opp_macro);
    score += 5.0f * static_cast<float>(my_boards - opp_boards);

    const uint16_t* my_cells = (s.turn == 1) ? s.x_cells : s.o_cells;
    const uint16_t* opp_cells = (s.turn == 1) ? s.o_cells : s.x_cells;

    uint16_t closed = s.macro_closed();
    for (int b = 0; b < 9; ++b) {
        if (!(closed & (1 << b))) {
            score += eval_local_board(my_cells[b], opp_cells[b]);
        }
    }

    return 0.95f * std::tanh(score / 30.0f);
}

// Single fast random rollout to terminal state
inline int8_t rollout_game(BoardState state, FastRng& rng, int max_plies = 81) {
    uint8_t legal[81];
    int plies = 0;
    while (state.result == RESULT_ONGOING && plies < max_plies) {
        int n = state.get_legal_actions(legal);
        if (n == 0) break;
        uint32_t idx = rng.next_bounded(static_cast<uint32_t>(n));
        state.play_inplace(legal[idx]);
        plies++;
    }
    return state.result;
}

// Fast Alpha-Beta Search
class AlphaBetaSearcher {
public:
    uint64_t node_count = 0;

    int choose_move(const BoardState& s, int max_depth, uint64_t node_budget = 100000) {
        node_count = 0;
        uint8_t actions[81];
        int n = s.get_legal_actions(actions);
        if (n == 0) return -1;
        if (n == 1) return actions[0];

        int best_action = actions[0];
        float best_val = -std::numeric_limits<float>::infinity();
        float alpha = -10.0f;
        float beta = 10.0f;

        for (int i = 0; i < n; ++i) {
            BoardState child = s.play(actions[i]);
            float val = -negamax(child, max_depth - 1, -beta, -alpha, node_budget);
            if (val > best_val) {
                best_val = val;
                best_action = actions[i];
            }
            alpha = std::max(alpha, best_val);
            if (node_count >= node_budget) break;
        }
        return best_action;
    }

private:
    float negamax(const BoardState& s, int depth, float alpha, float beta, uint64_t budget) {
        node_count++;
        if (s.result != RESULT_ONGOING) {
            return static_cast<float>(s.result * s.turn);
        }
        if (depth <= 0 || node_count >= budget) {
            return evaluate_state(s);
        }

        uint8_t actions[81];
        int n = s.get_legal_actions(actions);
        if (n == 0) {
            return evaluate_state(s);
        }

        float best = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; ++i) {
            BoardState child = s;
            child.play_inplace(actions[i]);
            float val = -negamax(child, depth - 1, -beta, -alpha, budget);
            best = std::max(best, val);
            alpha = std::max(alpha, val);
            if (alpha >= beta || node_count >= budget) {
                break;
            }
        }
        return best;
    }
};

} // namespace sttt
