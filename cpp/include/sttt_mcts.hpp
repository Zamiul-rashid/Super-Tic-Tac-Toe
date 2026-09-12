#pragma once

#include "sttt_core.hpp"
#include "sttt_search.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <random>

namespace sttt {

struct MCTSConfig {
    bool soft_pruning = true;
    bool proofs = true;
    bool reuse = true;
    float c_puct = 1.5f;
    float soft_margin = 0.2f;
    float soft_strength = 3.0f;
    int32_t min_visits = 8;
    int32_t revisit_interval = 64;
};

struct MCTSStats {
    int32_t completed_simulations = 0;
    int32_t neural_positions = 0;
    int32_t inference_batches = 0;
    int32_t max_depth = 0;
    int32_t hard_pruned_choices = 0;
    int32_t soft_rechecks = 0;
    int32_t retained_visits = 0;
    int8_t root_solved = RESULT_ONGOING;
};

#pragma pack(push, 1)
struct MCTSNode {
    BoardState state;            // 46 bytes
    double prior = 1.0;          // 8 bytes: prior probability P(s, a)
    double total = 0.0;          // 8 bytes: cumulative value W(s)
    int32_t n = 0;               // 4 bytes: visit count N(s)
    int32_t in_flight = 0;       // 4 bytes: virtual visits
    int32_t selections = 0;      // 4 bytes: total selection count
    int32_t last_selected = 0;   // 4 bytes: selection counter when last picked
    int32_t parent = -1;         // 4 bytes: parent node index in arena
    int32_t first_child = -1;    // 4 bytes: first child index in arena
    uint8_t num_children = 0;    // 1 byte: number of children
    uint8_t action = 255;        // 1 byte: action from parent that reached this node
    int8_t solved = RESULT_ONGOING; // 1 byte: RESULT_ONGOING (2), 1, -1, or 0
    bool pending = false;        // 1 byte: currently selected awaiting evaluation
    uint8_t padding[6] = {0, 0, 0, 0, 0, 0}; // 6 bytes: pad to 96 bytes
};
#pragma pack(pop)

static_assert(sizeof(MCTSNode) == 96, "MCTSNode size must be 96 bytes for cache line packing");

class MCTSEngine {
public:
    std::vector<MCTSNode> arena;
    int32_t root_idx = -1;
    MCTSConfig config;
    MCTSStats stats;
    int8_t agent_side = 0;
    FastRng rng;
    std::vector<float> root_priors_override;

    explicit MCTSEngine(const MCTSConfig& cfg = MCTSConfig(), uint64_t seed = 42)
        : config(cfg), rng(seed) {
        arena.reserve(65536);
    }

    inline bool use_proofs() const {
        return config.proofs;
    }

    void reset() {
        arena.clear();
        root_idx = -1;
        stats = MCTSStats();
        root_priors_override.clear();
        agent_side = 0;
    }

    void advance(uint8_t action) {
        if (root_idx >= 0 && root_idx < static_cast<int32_t>(arena.size())) {
            const MCTSNode& root = arena[root_idx];
            int32_t target_child = -1;
            if (root.first_child >= 0) {
                for (uint8_t i = 0; i < root.num_children; ++i) {
                    if (arena[root.first_child + i].action == action) {
                        target_child = root.first_child + i;
                        break;
                    }
                }
            }
            if (target_child >= 0) {
                root_idx = target_child;
                arena[root_idx].parent = -1;
                root_priors_override.clear();
                return;
            }
            // Subtree child not found: create state from played action
            BoardState next_s = root.state.play(action);
            init_root(next_s);
        }
    }

    void init_root(const BoardState& state) {
        arena.clear();
        root_idx = 0;
        stats = MCTSStats();
        root_priors_override.clear();
        if (agent_side == 0) {
            agent_side = state.turn;
        }

        MCTSNode root{};
        root.state = state;
        root.prior = 1.0f;
        root.parent = -1;
        root.first_child = -1;
        root.num_children = 0;
        root.action = 255;
        root.solved = state.is_terminal() ? static_cast<int8_t>(state.result * state.turn) : RESULT_ONGOING;
        arena.push_back(root);
    }

    void prove(int32_t node_idx) {
        if (!use_proofs()) return;
        MCTSNode& node = arena[node_idx];
        if (node.first_child < 0 || node.num_children == 0 || node.solved != RESULT_ONGOING) {
            return;
        }

        bool all_solved = true;
        int8_t max_neg = -128;
        for (uint8_t i = 0; i < node.num_children; ++i) {
            int8_t s = arena[node.first_child + i].solved;
            if (s == -1) {
                // If any child is proven -1 (opponent loss), this node is proven +1 (win)!
                arena[node_idx].solved = 1;
                return;
            }
            if (s == RESULT_ONGOING) {
                all_solved = false;
            } else {
                int8_t neg = static_cast<int8_t>(-s);
                if (neg > max_neg) {
                    max_neg = neg;
                }
            }
        }
        if (all_solved && max_neg > -128) {
            arena[node_idx].solved = max_neg;
        }
    }

    void expand(int32_t node_idx, const float* policy, int policy_len = 81) {
        uint8_t legal[81];
        int num_legal = arena[node_idx].state.get_legal_actions(legal);
        if (num_legal == 0) return;

        double probs[81];
        double sum = 0.0;
        for (int i = 0; i < num_legal; ++i) {
            uint8_t a = legal[i];
            double p = (a < policy_len) ? static_cast<double>(policy[a]) : 0.0;
            p = std::max(p, 1e-5);
            probs[i] = p;
            sum += p;
        }
        if (sum > 0.0) {
            for (int i = 0; i < num_legal; ++i) {
                probs[i] /= sum;
            }
        }

        int32_t first_child = static_cast<int32_t>(arena.size());
        arena[node_idx].first_child = first_child;
        arena[node_idx].num_children = static_cast<uint8_t>(num_legal);

        // Preallocate space in arena
        arena.resize(first_child + num_legal);

        for (int i = 0; i < num_legal; ++i) {
            uint8_t a = legal[i];
            MCTSNode child{};
            child.state = arena[node_idx].state.play(a);
            child.prior = probs[i];
            child.total = 0.0;
            child.n = 0;
            child.in_flight = 0;
            child.selections = 0;
            child.last_selected = 0;
            child.parent = node_idx;
            child.first_child = -1;
            child.num_children = 0;
            child.action = a;
            child.solved = child.state.is_terminal() ? static_cast<int8_t>(child.state.result * child.state.turn) : RESULT_ONGOING;
            child.pending = false;

            arena[first_child + i] = child;
        }

        prove(node_idx);
    }

    int get_eligible(int32_t node_idx, int32_t* out_children) {
        const MCTSNode& node = arena[node_idx];
        if (node.first_child < 0 || node.num_children == 0) return 0;
        int first = node.first_child;
        int total_children = node.num_children;

        if (use_proofs()) {
            if (node.solved != RESULT_ONGOING) {
                int8_t target = static_cast<int8_t>(-node.solved);
                int count = 0;
                for (int i = 0; i < total_children; ++i) {
                    if (arena[first + i].solved == target) {
                        out_children[count++] = first + i;
                    }
                }
                return count;
            }

            int32_t alts[81];
            int alt_count = 0;
            for (int i = 0; i < total_children; ++i) {
                if (arena[first + i].solved != 1) {
                    alts[alt_count++] = first + i;
                }
            }
            if (alt_count > 0 && alt_count < total_children) {
                stats.hard_pruned_choices += (total_children - alt_count);
                for (int i = 0; i < alt_count; ++i) {
                    out_children[i] = alts[i];
                }
                return alt_count;
            }
        }

        for (int i = 0; i < total_children; ++i) {
            out_children[i] = first + i;
        }
        return total_children;
    }

    inline double q_value(int32_t child_idx) const {
        const MCTSNode& c = arena[child_idx];
        if (use_proofs() && c.solved != RESULT_ONGOING) {
            return -static_cast<double>(c.solved);
        }
        return -(c.total + static_cast<double>(c.in_flight)) /
               static_cast<double>(std::max(1, c.n + c.in_flight));
    }

    int32_t select_leaf(int32_t node_idx, std::vector<int32_t>& path) {
        if (arena[node_idx].pending) {
            return -1;
        }
        if (arena[node_idx].state.is_terminal() ||
            (use_proofs() && arena[node_idx].solved != RESULT_ONGOING) ||
            arena[node_idx].first_child < 0 ||
            arena[node_idx].num_children == 0) {
            return node_idx;
        }

        int32_t eligible[81];
        int count = get_eligible(node_idx, eligible);

        while (count > 0) {
            int32_t chosen_idx = -1;
            int chosen_slot = -1;

            if (config.soft_pruning && arena[node_idx].selections > 0 &&
                (arena[node_idx].selections % config.revisit_interval == 0)) {
                int min_last = std::numeric_limits<int>::max();
                int min_vis = std::numeric_limits<int>::max();
                for (int i = 0; i < count; ++i) {
                    const MCTSNode& c = arena[eligible[i]];
                    int vis = c.n + c.in_flight;
                    if (c.last_selected < min_last || (c.last_selected == min_last && vis < min_vis)) {
                        min_last = c.last_selected;
                        min_vis = vis;
                        chosen_idx = eligible[i];
                        chosen_slot = i;
                    }
                }
                stats.soft_rechecks++;
            } else {
                bool has_reliable = false;
                double best_reliable_q = -1e18;
                for (int i = 0; i < count; ++i) {
                    const MCTSNode& c = arena[eligible[i]];
                    if (c.n >= config.min_visits) {
                        double q = q_value(eligible[i]);
                        if (!has_reliable || q > best_reliable_q) {
                            best_reliable_q = q;
                            has_reliable = true;
                        }
                    }
                }

                double parent_term = static_cast<double>(config.c_puct) * std::sqrt(static_cast<double>(arena[node_idx].n + arena[node_idx].in_flight + 1));
                double best_score = -1e18;

                for (int i = 0; i < count; ++i) {
                    int32_t c_idx = eligible[i];
                    const MCTSNode& c = arena[c_idx];
                    double q = q_value(c_idx);
                    double weight = 1.0;
                    if (config.soft_pruning && has_reliable && c.n >= config.min_visits) {
                        double gap = std::max(0.0, best_reliable_q - q - static_cast<double>(config.soft_margin));
                        weight = std::max(0.2, 1.0 / (1.0 + static_cast<double>(config.soft_strength) * gap));
                    }
                    double prior = c.prior;
                    if (node_idx == root_idx && !root_priors_override.empty()) {
                        int local_idx = c_idx - arena[node_idx].first_child;
                        if (local_idx >= 0 && local_idx < static_cast<int>(root_priors_override.size())) {
                            prior = static_cast<double>(root_priors_override[local_idx]);
                        }
                    }
                    double u = weight * prior * parent_term / static_cast<double>(1 + c.n + c.in_flight);
                    double score = q + u;
                    if (chosen_slot < 0 || score > best_score) {
                        best_score = score;
                        chosen_idx = c_idx;
                        chosen_slot = i;
                    }
                }
            }

            arena[node_idx].selections++;
            arena[chosen_idx].last_selected = arena[node_idx].selections;

            path.push_back(chosen_idx);
            int32_t leaf = select_leaf(chosen_idx, path);
            if (leaf >= 0) {
                return leaf;
            }
            path.pop_back();

            // Remove chosen_slot from eligible
            eligible[chosen_slot] = eligible[count - 1];
            count--;
        }

        return -1;
    }

    void backup(const std::vector<int32_t>& path, double value) {
        for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
            int32_t idx = path[i];
            prove(idx);
            if (use_proofs() && arena[idx].solved != RESULT_ONGOING) {
                value = static_cast<double>(arena[idx].solved);
            }
            arena[idx].n += 1;
            arena[idx].total += value;
            value = -value;
        }
        stats.completed_simulations++;
        stats.max_depth = std::max(stats.max_depth, static_cast<int32_t>(path.size()) - 1);
    }

    void release(const std::vector<int32_t>& path) {
        arena[path.back()].pending = false;
        for (int32_t idx : path) {
            arena[idx].in_flight -= 1;
        }
    }

    void get_policy(float* out_policy) {
        std::memset(out_policy, 0, 81 * sizeof(float));
        if (root_idx < 0 || arena[root_idx].first_child < 0) return;

        stats.root_solved = use_proofs() ? arena[root_idx].solved : RESULT_ONGOING;

        int32_t eligible[81];
        int count = get_eligible(root_idx, eligible);
        float sum = 0.0f;

        for (int i = 0; i < count; ++i) {
            const MCTSNode& c = arena[eligible[i]];
            out_policy[c.action] = static_cast<float>(c.n);
            sum += out_policy[c.action];
        }

        if (sum == 0.0f) {
            for (int i = 0; i < count; ++i) {
                const MCTSNode& c = arena[eligible[i]];
                out_policy[c.action] = c.prior;
                sum += c.prior;
            }
        }

        if (sum > 0.0f) {
            for (int a = 0; a < 81; ++a) {
                out_policy[a] /= sum;
            }
        }
    }

    // Pure C++ heuristic search: 100k+ sims/s without Python interaction
    void run_heuristic(const BoardState& state, int simulations, int batch_size = 1) {
        if (simulations < 1 || batch_size < 1 || state.is_terminal()) {
            throw std::invalid_argument("Search requires nonterminal state and positive budgets");
        }

        if (root_idx < 0 || !config.reuse || arena.empty() || arena[root_idx].state != state) {
            init_root(state);
        }

        stats = MCTSStats();
        stats.retained_visits = arena[root_idx].n;

        if (arena[root_idx].first_child < 0) {
            float uniform_policy[81];
            for (int i = 0; i < 81; ++i) uniform_policy[i] = 1.0f / 81.0f;
            expand(root_idx, uniform_policy, 81);
            stats.neural_positions++;
            stats.inference_batches++;
        }

        std::vector<std::vector<int32_t>> pending_paths;
        pending_paths.reserve(batch_size);

        while (stats.completed_simulations < simulations) {
            if (use_proofs() && arena[root_idx].solved != RESULT_ONGOING) {
                break;
            }

            pending_paths.clear();
            while (static_cast<int>(pending_paths.size()) < batch_size &&
                   stats.completed_simulations + static_cast<int>(pending_paths.size()) < simulations) {
                std::vector<int32_t> path = {root_idx};
                int32_t leaf = select_leaf(root_idx, path);
                if (leaf < 0) break;

                MCTSNode& leaf_node = arena[leaf];
                if (leaf_node.state.is_terminal() || (use_proofs() && leaf_node.solved != RESULT_ONGOING)) {
                    backup(path, static_cast<float>(leaf_node.solved));
                    if (use_proofs() && arena[root_idx].solved != RESULT_ONGOING) {
                        break;
                    }
                } else {
                    leaf_node.pending = true;
                    for (int32_t idx : path) {
                        arena[idx].in_flight += 1;
                    }
                    pending_paths.push_back(std::move(path));
                }
            }

            if (!pending_paths.empty()) {
                stats.inference_batches++;
                stats.neural_positions += static_cast<int32_t>(pending_paths.size());

                for (auto& path : pending_paths) {
                    int32_t leaf = path.back();
                    float uniform_policy[81];
                    for (int i = 0; i < 81; ++i) uniform_policy[i] = 1.0f / 81.0f;
                    expand(leaf, uniform_policy, 81);

                    float val = evaluate_state(arena[leaf].state);
                    backup(path, val);
                    release(path);
                }
            }
        }

        stats.root_solved = use_proofs() ? arena[root_idx].solved : RESULT_ONGOING;
    }
};

} // namespace sttt
