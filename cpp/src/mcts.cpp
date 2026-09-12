#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "sttt_mcts.hpp"
#include "sttt_core.hpp"
#include "sttt_search.hpp"
#include <chrono>
#include <thread>
#include <vector>
#include <string>

using namespace sttt;

// Forward declaration from python_module.cpp
extern PyTypeObject PyFastStateType;
extern const BoardState* extract_board_state(PyObject* obj, PyObject** cleanup);

typedef struct {
    PyObject_HEAD
    MCTSEngine* engine;
    PyObject* model;
    PyObject* opponent;
    PyObject* rng;
    PyObject* config_obj;
    int agent_side;
    bool has_agent_side;
} PyFastTreeSearch;

typedef struct {
    PyObject_HEAD
    PyFastTreeSearch* tree;
    int32_t node_idx;
} PyFastNode;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
static PyTypeObject PyFastNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyTypeObject PyFastTreeSearchType = {
    PyVarObject_HEAD_INIT(NULL, 0)
};
#pragma GCC diagnostic pop

// --- PyFastNode implementation ---

static PyObject* PyFastNode_new(PyFastTreeSearch* tree, int32_t node_idx) {
    PyFastNode* self = (PyFastNode*)PyFastNodeType.tp_alloc(&PyFastNodeType, 0);
    if (self != NULL) {
        Py_INCREF(tree);
        self->tree = tree;
        self->node_idx = node_idx;
    }
    return (PyObject*)self;
}

static void PyFastNode_dealloc(PyFastNode* self) {
    Py_XDECREF(self->tree);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyFastNode_get_solved(PyFastNode* self, void* /*closure*/) {
    if (!self->tree || !self->tree->engine || self->node_idx < 0 ||
        self->node_idx >= static_cast<int32_t>(self->tree->engine->arena.size())) {
        Py_RETURN_NONE;
    }
    int8_t s = self->tree->engine->arena[self->node_idx].solved;
    if (s == RESULT_ONGOING) {
        Py_RETURN_NONE;
    }
    return PyLong_FromLong(s);
}

static PyObject* PyFastNode_get_n(PyFastNode* self, void* /*closure*/) {
    if (!self->tree || !self->tree->engine || self->node_idx < 0 ||
        self->node_idx >= static_cast<int32_t>(self->tree->engine->arena.size())) {
        return PyLong_FromLong(0);
    }
    return PyLong_FromLong(self->tree->engine->arena[self->node_idx].n);
}

static PyObject* PyFastNode_get_total(PyFastNode* self, void* /*closure*/) {
    if (!self->tree || !self->tree->engine || self->node_idx < 0 ||
        self->node_idx >= static_cast<int32_t>(self->tree->engine->arena.size())) {
        return PyFloat_FromDouble(0.0);
    }
    return PyFloat_FromDouble(self->tree->engine->arena[self->node_idx].total);
}

static PyObject* PyFastNode_get_in_flight(PyFastNode* self, void* /*closure*/) {
    if (!self->tree || !self->tree->engine || self->node_idx < 0 ||
        self->node_idx >= static_cast<int32_t>(self->tree->engine->arena.size())) {
        return PyLong_FromLong(0);
    }
    return PyLong_FromLong(self->tree->engine->arena[self->node_idx].in_flight);
}

static PyObject* PyFastNode_get_pending(PyFastNode* self, void* /*closure*/) {
    if (!self->tree || !self->tree->engine || self->node_idx < 0 ||
        self->node_idx >= static_cast<int32_t>(self->tree->engine->arena.size())) {
        Py_RETURN_FALSE;
    }
    if (self->tree->engine->arena[self->node_idx].pending) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject* PyFastNode_get_prior(PyFastNode* self, void* /*closure*/) {
    if (!self->tree || !self->tree->engine || self->node_idx < 0 ||
        self->node_idx >= static_cast<int32_t>(self->tree->engine->arena.size())) {
        return PyFloat_FromDouble(1.0);
    }
    return PyFloat_FromDouble(self->tree->engine->arena[self->node_idx].prior);
}

static PyObject* PyFastNode_get_state(PyFastNode* self, void* /*closure*/) {
    if (!self->tree || !self->tree->engine || self->node_idx < 0 ||
        self->node_idx >= static_cast<int32_t>(self->tree->engine->arena.size())) {
        Py_RETURN_NONE;
    }
    PyObject* fs = PyObject_CallObject((PyObject*)&PyFastStateType, NULL);
    if (!fs) return NULL;
    // Set state
    const BoardState& bs = self->tree->engine->arena[self->node_idx].state;
    // FastState layout: head followed by BoardState state
    // Let's copy state directly
    struct RawFastState {
        PyObject_HEAD
        BoardState state;
    };
    ((RawFastState*)fs)->state = bs;
    return fs;
}

static PyObject* PyFastNode_get_children(PyFastNode* self, void* /*closure*/) {
    PyObject* dict = PyDict_New();
    if (!dict) return NULL;
    if (!self->tree || !self->tree->engine || self->node_idx < 0 ||
        self->node_idx >= static_cast<int32_t>(self->tree->engine->arena.size())) {
        return dict;
    }

    const MCTSNode& node = self->tree->engine->arena[self->node_idx];
    if (node.first_child >= 0) {
        for (uint8_t i = 0; i < node.num_children; ++i) {
            int32_t child_idx = node.first_child + i;
            const MCTSNode& child = self->tree->engine->arena[child_idx];
            PyObject* key = PyLong_FromLong(child.action);
            PyObject* child_obj = PyFastNode_new(self->tree, child_idx);
            PyDict_SetItem(dict, key, child_obj);
            Py_DECREF(key);
            Py_DECREF(child_obj);
        }
    }
    return dict;
}

static PyGetSetDef PyFastNode_getseters[] = {
    {(char*)"solved", (getter)PyFastNode_get_solved, NULL, (char*)"Node proven solved value", NULL},
    {(char*)"n", (getter)PyFastNode_get_n, NULL, (char*)"Visit count", NULL},
    {(char*)"total", (getter)PyFastNode_get_total, NULL, (char*)"Cumulative value", NULL},
    {(char*)"in_flight", (getter)PyFastNode_get_in_flight, NULL, (char*)"Virtual loss visits", NULL},
    {(char*)"pending", (getter)PyFastNode_get_pending, NULL, (char*)"Pending status", NULL},
    {(char*)"prior", (getter)PyFastNode_get_prior, NULL, (char*)"Prior probability", NULL},
    {(char*)"state", (getter)PyFastNode_get_state, NULL, (char*)"Board state", NULL},
    {(char*)"children", (getter)PyFastNode_get_children, NULL, (char*)"Children dict {action: Node}", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

// --- PyFastTreeSearch implementation ---

static PyObject* PyFastTreeSearch_new(PyTypeObject* type, PyObject* /*args*/, PyObject* /*kwds*/) {
    PyFastTreeSearch* self = (PyFastTreeSearch*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->engine = new MCTSEngine();
        self->model = NULL;
        self->opponent = NULL;
        self->rng = NULL;
        self->config_obj = NULL;
        self->agent_side = 0;
        self->has_agent_side = false;
    }
    return (PyObject*)self;
}

static void PyFastTreeSearch_dealloc(PyFastTreeSearch* self) {
    if (self->engine) {
        delete self->engine;
        self->engine = NULL;
    }
    Py_XDECREF(self->model);
    Py_XDECREF(self->opponent);
    Py_XDECREF(self->rng);
    Py_XDECREF(self->config_obj);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int PyFastTreeSearch_init(PyFastTreeSearch* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"model", "rng", "config", "opponent", "agent_side", NULL};
    PyObject* model = NULL;
    PyObject* rng = NULL;
    PyObject* config = NULL;
    PyObject* opponent = NULL;
    PyObject* agent_side = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOO", (char**)kwlist,
                                     &model, &rng, &config, &opponent, &agent_side)) {
        return -1;
    }

    Py_XDECREF(self->model);
    Py_XINCREF(model);
    self->model = model;

    Py_XDECREF(self->rng);
    Py_XINCREF(rng);
    self->rng = rng;

    Py_XDECREF(self->config_obj);
    Py_XINCREF(config);
    self->config_obj = config;

    Py_XDECREF(self->opponent);
    Py_XINCREF(opponent);
    self->opponent = opponent;

    if (agent_side != NULL && agent_side != Py_None) {
        self->agent_side = (int)PyLong_AsLong(agent_side);
        self->has_agent_side = true;
    } else {
        self->agent_side = 0;
        self->has_agent_side = false;
    }

    // Configure MCTS
    MCTSConfig cfg;
    if (config != NULL && config != Py_None) {
        PyObject* val;
        val = PyObject_GetAttrString(config, "soft_pruning");
        if (val) { cfg.soft_pruning = PyObject_IsTrue(val); Py_DECREF(val); } else PyErr_Clear();
        val = PyObject_GetAttrString(config, "proofs");
        if (val) { cfg.proofs = PyObject_IsTrue(val); Py_DECREF(val); } else PyErr_Clear();
        val = PyObject_GetAttrString(config, "reuse");
        if (val) { cfg.reuse = PyObject_IsTrue(val); Py_DECREF(val); } else PyErr_Clear();
        val = PyObject_GetAttrString(config, "c_puct");
        if (val) { cfg.c_puct = (float)PyFloat_AsDouble(val); Py_DECREF(val); } else PyErr_Clear();
        val = PyObject_GetAttrString(config, "soft_margin");
        if (val) { cfg.soft_margin = (float)PyFloat_AsDouble(val); Py_DECREF(val); } else PyErr_Clear();
        val = PyObject_GetAttrString(config, "soft_strength");
        if (val) { cfg.soft_strength = (float)PyFloat_AsDouble(val); Py_DECREF(val); } else PyErr_Clear();
        val = PyObject_GetAttrString(config, "min_visits");
        if (val) { cfg.min_visits = (int32_t)PyLong_AsLong(val); Py_DECREF(val); } else PyErr_Clear();
        val = PyObject_GetAttrString(config, "revisit_interval");
        if (val) { cfg.revisit_interval = (int32_t)PyLong_AsLong(val); Py_DECREF(val); } else PyErr_Clear();
    }

    // Minimax losses need not be losses against a fallible opponent model
    if (opponent != NULL && opponent != Py_None) {
        cfg.proofs = false;
    }

    if (self->engine) {
        delete self->engine;
    }
    self->engine = new MCTSEngine(cfg);
    if (self->has_agent_side) {
        self->engine->agent_side = static_cast<int8_t>(self->agent_side);
    }
    return 0;
}

static PyObject* PyFastTreeSearch_reset(PyFastTreeSearch* self, PyObject* /*args*/) {
    if (self->engine) {
        self->engine->reset();
    }
    Py_RETURN_NONE;
}

static PyObject* PyFastTreeSearch_advance(PyFastTreeSearch* self, PyObject* args) {
    int action;
    if (!PyArg_ParseTuple(args, "i", &action)) {
        return NULL;
    }
    if (self->engine) {
        self->engine->advance(static_cast<uint8_t>(action));
    }
    Py_RETURN_NONE;
}

static PyObject* PyFastTreeSearch_get_stats(PyFastTreeSearch* self, void* /*closure*/) {
    if (!self->engine) {
        Py_RETURN_NONE;
    }
    const MCTSStats& st = self->engine->stats;
    PyObject* d = PyDict_New();
    if (!d) return NULL;

    PyDict_SetItemString(d, "completed_simulations", PyLong_FromLong(st.completed_simulations));
    PyDict_SetItemString(d, "neural_positions", PyLong_FromLong(st.neural_positions));
    PyDict_SetItemString(d, "inference_batches", PyLong_FromLong(st.inference_batches));
    PyDict_SetItemString(d, "max_depth", PyLong_FromLong(st.max_depth));
    PyDict_SetItemString(d, "hard_pruned_choices", PyLong_FromLong(st.hard_pruned_choices));
    PyDict_SetItemString(d, "soft_rechecks", PyLong_FromLong(st.soft_rechecks));
    PyDict_SetItemString(d, "retained_visits", PyLong_FromLong(st.retained_visits));

    if (st.root_solved == RESULT_ONGOING) {
        PyDict_SetItemString(d, "root_solved", Py_None);
    } else {
        PyDict_SetItemString(d, "root_solved", PyLong_FromLong(st.root_solved));
    }
    return d;
}

static PyObject* PyFastTreeSearch_get_root(PyFastTreeSearch* self, void* /*closure*/) {
    if (!self->engine || self->engine->root_idx < 0) {
        Py_RETURN_NONE;
    }
    return PyFastNode_new(self, self->engine->root_idx);
}

static PyObject* PyFastTreeSearch_get_use_proofs(PyFastTreeSearch* self, void* /*closure*/) {
    if (!self->engine) {
        Py_RETURN_FALSE;
    }
    if (self->engine->use_proofs()) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject* PyFastTreeSearch_get_model(PyFastTreeSearch* self, void* /*closure*/) {
    if (self->model) {
        Py_INCREF(self->model);
        return self->model;
    }
    Py_RETURN_NONE;
}

static PyObject* PyFastTreeSearch_get_config(PyFastTreeSearch* self, void* /*closure*/) {
    if (self->config_obj) {
        Py_INCREF(self->config_obj);
        return self->config_obj;
    }
    Py_RETURN_NONE;
}

static PyGetSetDef PyFastTreeSearch_getseters[] = {
    {(char*)"stats", (getter)PyFastTreeSearch_get_stats, NULL, (char*)"Search statistics dictionary", NULL},
    {(char*)"root", (getter)PyFastTreeSearch_get_root, NULL, (char*)"Root node of search tree", NULL},
    {(char*)"use_proofs", (getter)PyFastTreeSearch_get_use_proofs, NULL, (char*)"Proof pruning enabled", NULL},
    {(char*)"model", (getter)PyFastTreeSearch_get_model, NULL, (char*)"Evaluator model", NULL},
    {(char*)"config", (getter)PyFastTreeSearch_get_config, NULL, (char*)"Search configuration", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyObject* PyFastTreeSearch_run(PyFastTreeSearch* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"state", "simulations", "batch_size", "noise", NULL};
    PyObject* state_obj;
    int simulations;
    int batch_size = 1;
    PyObject* noise_obj = Py_False;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oi|iO", (char**)kwlist,
                                     &state_obj, &simulations, &batch_size, &noise_obj)) {
        return NULL;
    }

    if (simulations < 1 || batch_size < 1) {
        PyErr_SetString(PyExc_ValueError, "Search requires a nonterminal state and positive budgets");
        return NULL;
    }

    PyObject* cleanup = NULL;
    const BoardState* bs = extract_board_state(state_obj, &cleanup);
    if (!bs) {
        PyErr_SetString(PyExc_TypeError, "state must be a State, FastState, or CppState");
        return NULL;
    }

    if (bs->is_terminal()) {
        Py_XDECREF(cleanup);
        PyErr_SetString(PyExc_ValueError, "Search requires a nonterminal state and positive budgets");
        return NULL;
    }

    MCTSEngine* engine = self->engine;
    if (!engine) {
        Py_XDECREF(cleanup);
        PyErr_SetString(PyExc_RuntimeError, "MCTS engine not initialized");
        return NULL;
    }

    // Setup root if needed
    if (engine->root_idx < 0 || !engine->config.reuse || engine->arena.empty() || engine->arena[engine->root_idx].state != *bs) {
        engine->init_root(*bs);
    }
    if (!self->has_agent_side) {
        engine->agent_side = bs->turn;
    }

    int retained_visits = engine->arena[engine->root_idx].n;
    engine->stats = MCTSStats();
    engine->stats.retained_visits = retained_visits;
    engine->root_priors_override.clear();

    // Expand root if needed
    if (engine->arena[engine->root_idx].first_child < 0) {
        if (self->model != NULL && self->model != Py_None) {
            PyObject* eval_res = PyObject_CallMethod(self->model, "evaluate", "O", state_obj);
            if (!eval_res) {
                Py_XDECREF(cleanup);
                return NULL;
            }
            PyObject* py_policy = PyTuple_GetItem(eval_res, 0);
            if (!py_policy) {
                Py_DECREF(eval_res);
                Py_XDECREF(cleanup);
                return NULL;
            }
            float policy_buf[81];
            PyObject* seq = PySequence_Fast(py_policy, "policy must be a sequence of 81 numbers");
            if (!seq || PySequence_Fast_GET_SIZE(seq) != 81) {
                Py_XDECREF(seq);
                Py_DECREF(eval_res);
                Py_XDECREF(cleanup);
                PyErr_SetString(PyExc_ValueError, "Evaluator returned an invalid policy");
                return NULL;
            }
            for (int i = 0; i < 81; ++i) {
                PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
                policy_buf[i] = static_cast<float>(PyFloat_AsDouble(item));
            }
            Py_DECREF(seq);
            Py_DECREF(eval_res);

            engine->expand(engine->root_idx, policy_buf, 81);
            engine->stats.neural_positions++;
            engine->stats.inference_batches++;
        } else {
            // Heuristic uniform expansion
            float uniform[81];
            for (int i = 0; i < 81; ++i) uniform[i] = 1.0f / 81.0f;
            engine->expand(engine->root_idx, uniform, 81);
            engine->stats.neural_positions++;
            engine->stats.inference_batches++;
        }
    }

    // Root Dirichlet noise
    bool add_noise = PyObject_IsTrue(noise_obj);
    if (add_noise && engine->arena[engine->root_idx].first_child >= 0) {
        int num_children = engine->arena[engine->root_idx].num_children;
        int first_child = engine->arena[engine->root_idx].first_child;
        engine->root_priors_override.resize(num_children);

        // Sample Dirichlet(0.3) in C++ using Gamma distribution
        std::gamma_distribution<float> gamma_dist(0.3f, 1.0f);
        std::mt19937 mt(static_cast<uint32_t>(engine->rng.next()));
        float gamma_sum = 0.0f;
        std::vector<float> etas(num_children);
        for (int i = 0; i < num_children; ++i) {
            etas[i] = std::max(1e-6f, gamma_dist(mt));
            gamma_sum += etas[i];
        }
        for (int i = 0; i < num_children; ++i) {
            float eta = etas[i] / gamma_sum;
            float p = engine->arena[first_child + i].prior;
            engine->root_priors_override[i] = 0.75f * p + 0.25f * eta;
        }
    }

    // Pure C++ fast path when model is None
    if (self->model == NULL || self->model == Py_None) {
        Py_BEGIN_ALLOW_THREADS
        while (engine->stats.completed_simulations < simulations) {
            if (engine->use_proofs() && engine->arena[engine->root_idx].solved != RESULT_ONGOING) {
                break;
            }
            std::vector<std::vector<int32_t>> pending;
            while (static_cast<int>(pending.size()) < batch_size &&
                   engine->stats.completed_simulations + static_cast<int>(pending.size()) < simulations) {
                std::vector<int32_t> path = {engine->root_idx};
                int32_t leaf = engine->select_leaf(engine->root_idx, path);
                if (leaf < 0) break;

                MCTSNode& leaf_node = engine->arena[leaf];
                if (leaf_node.state.is_terminal() || (engine->use_proofs() && leaf_node.solved != RESULT_ONGOING)) {
                    engine->backup(path, static_cast<float>(leaf_node.solved));
                    if (engine->use_proofs() && engine->arena[engine->root_idx].solved != RESULT_ONGOING) break;
                } else {
                    leaf_node.pending = true;
                    for (int32_t idx : path) engine->arena[idx].in_flight += 1;
                    pending.push_back(std::move(path));
                }
            }

            if (!pending.empty()) {
                engine->stats.inference_batches++;
                engine->stats.neural_positions += static_cast<int32_t>(pending.size());

                for (auto& path : pending) {
                    int32_t leaf = path.back();
                    float uniform[81];
                    for (int i = 0; i < 81; ++i) uniform[i] = 1.0f / 81.0f;
                    engine->expand(leaf, uniform, 81);
                    float val = evaluate_state(engine->arena[leaf].state);
                    engine->backup(path, val);
                    engine->release(path);
                }
            }
        }
        Py_END_ALLOW_THREADS
    } else {
        // Python Model Evaluator Loop
        bool has_evaluate_many = PyObject_HasAttrString(self->model, "evaluate_many");

        while (engine->stats.completed_simulations < simulations) {
            if (engine->use_proofs() && engine->arena[engine->root_idx].solved != RESULT_ONGOING) {
                break;
            }

            std::vector<std::vector<int32_t>> pending;
            while (static_cast<int>(pending.size()) < batch_size &&
                   engine->stats.completed_simulations + static_cast<int>(pending.size()) < simulations) {
                std::vector<int32_t> path = {engine->root_idx};
                int32_t leaf = engine->select_leaf(engine->root_idx, path);
                if (leaf < 0) break;

                MCTSNode& leaf_node = engine->arena[leaf];
                if (leaf_node.state.is_terminal() || (engine->use_proofs() && leaf_node.solved != RESULT_ONGOING)) {
                    engine->backup(path, static_cast<float>(leaf_node.solved));
                    if (engine->use_proofs() && engine->arena[engine->root_idx].solved != RESULT_ONGOING) break;
                } else {
                    leaf_node.pending = true;
                    for (int32_t idx : path) engine->arena[idx].in_flight += 1;
                    pending.push_back(std::move(path));
                }
            }

            if (!pending.empty()) {
                // Prepare states for neural model
                PyObject* py_states_list = PyList_New(pending.size());
                struct RawFastState {
                    PyObject_HEAD
                    BoardState state;
                };

                for (size_t i = 0; i < pending.size(); ++i) {
                    int32_t leaf = pending[i].back();
                    PyObject* fs = PyObject_CallObject((PyObject*)&PyFastStateType, NULL);
                    ((RawFastState*)fs)->state = engine->arena[leaf].state;
                    PyList_SET_ITEM(py_states_list, i, fs);
                }

                PyObject* outputs = NULL;
                if (has_evaluate_many) {
                    outputs = PyObject_CallMethod(self->model, "evaluate_many", "O", py_states_list);
                } else {
                    outputs = PyList_New(pending.size());
                    for (size_t i = 0; i < pending.size(); ++i) {
                        PyObject* s = PyList_GET_ITEM(py_states_list, i);
                        PyObject* res = PyObject_CallMethod(self->model, "evaluate", "O", s);
                        if (!res) {
                            Py_DECREF(outputs);
                            outputs = NULL;
                            break;
                        }
                        PyList_SET_ITEM(outputs, i, res);
                    }
                }
                Py_DECREF(py_states_list);

                if (!outputs) {
                    // Exception occurred: release all pending paths before returning
                    for (auto& path : pending) {
                        engine->release(path);
                    }
                    Py_XDECREF(cleanup);
                    return NULL;
                }

                PyObject* seq = PySequence_Fast(outputs, "Outputs must be sequence");
                if (!seq || PySequence_Fast_GET_SIZE(seq) != static_cast<Py_ssize_t>(pending.size())) {
                    Py_XDECREF(seq);
                    Py_DECREF(outputs);
                    for (auto& path : pending) engine->release(path);
                    Py_XDECREF(cleanup);
                    PyErr_SetString(PyExc_ValueError, "Evaluator batch length mismatch");
                    return NULL;
                }

                engine->stats.inference_batches++;
                engine->stats.neural_positions += static_cast<int32_t>(pending.size());

                for (size_t i = 0; i < pending.size(); ++i) {
                    PyObject* pair = PySequence_Fast_GET_ITEM(seq, i);
                    PyObject* py_pol = PyTuple_GetItem(pair, 0);
                    PyObject* py_val = PyTuple_GetItem(pair, 1);

                    float val = static_cast<float>(PyFloat_AsDouble(py_val));
                    if (PyErr_Occurred() || !std::isfinite(val) || val < -1.00001f || val > 1.00001f) {
                        Py_DECREF(seq);
                        Py_DECREF(outputs);
                        for (auto& path : pending) engine->release(path);
                        Py_XDECREF(cleanup);
                        PyErr_SetString(PyExc_ValueError, "Evaluator returned an invalid value");
                        return NULL;
                    }

                    PyObject* pol_seq = PySequence_Fast(py_pol, "Policy must be sequence");
                    if (!pol_seq || PySequence_Fast_GET_SIZE(pol_seq) != 81) {
                        Py_XDECREF(pol_seq);
                        Py_DECREF(seq);
                        Py_DECREF(outputs);
                        for (auto& path : pending) engine->release(path);
                        Py_XDECREF(cleanup);
                        PyErr_SetString(PyExc_ValueError, "Evaluator returned an invalid policy");
                        return NULL;
                    }

                    float policy_buf[81];
                    for (int k = 0; k < 81; ++k) {
                        policy_buf[k] = static_cast<float>(PyFloat_AsDouble(PySequence_Fast_GET_ITEM(pol_seq, k)));
                    }
                    Py_DECREF(pol_seq);

                    int32_t leaf = pending[i].back();
                    engine->expand(leaf, policy_buf, 81);
                    engine->backup(pending[i], val);
                    engine->release(pending[i]);
                }

                Py_DECREF(seq);
                Py_DECREF(outputs);
            }
        }
    }

    Py_XDECREF(cleanup);

    float final_policy[81];
    engine->get_policy(final_policy);

    // Build return numpy array or list
    // Import numpy to create numpy.ndarray
    PyObject* np_mod = PyImport_ImportModule("numpy");
    if (np_mod) {
        PyObject* fromiter = PyObject_GetAttrString(np_mod, "fromiter");
        PyObject* float32_type = PyObject_GetAttrString(np_mod, "float32");
        if (fromiter && float32_type) {
            PyObject* py_list = PyList_New(81);
            for (int a = 0; a < 81; ++a) {
                PyList_SET_ITEM(py_list, a, PyFloat_FromDouble(final_policy[a]));
            }
            PyObject* arr = PyObject_CallFunction(fromiter, "OOi", py_list, float32_type, 81);
            Py_DECREF(py_list);
            Py_DECREF(float32_type);
            Py_DECREF(fromiter);
            Py_DECREF(np_mod);
            if (arr) return arr;
        }
        Py_XDECREF(float32_type);
        Py_XDECREF(fromiter);
        Py_DECREF(np_mod);
        PyErr_Clear();
    }

    PyObject* ret_list = PyList_New(81);
    for (int a = 0; a < 81; ++a) {
        PyList_SET_ITEM(ret_list, a, PyFloat_FromDouble(final_policy[a]));
    }
    return ret_list;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
static PyMethodDef PyFastTreeSearch_methods[] = {
    {"reset", (PyCFunction)PyFastTreeSearch_reset, METH_NOARGS, "Reset the search tree arena"},
    {"advance", (PyCFunction)PyFastTreeSearch_advance, METH_VARARGS, "Retain the played subtree and advance root"},
    {"run", (PyCFunction)PyFastTreeSearch_run, METH_VARARGS | METH_KEYWORDS, "Run MCTS simulations on the given state"},
    {NULL, NULL, 0, NULL}
};
#pragma GCC diagnostic pop

// Benchmark function exposed directly to Python
static PyObject* py_benchmark_mcts(PyObject* /*self*/, PyObject* args) {
    int total_sims = 10000;
    int batch_size = 8;
    int num_threads = 1;

    if (!PyArg_ParseTuple(args, "|iii", &total_sims, &batch_size, &num_threads)) {
        return NULL;
    }

    if (total_sims < 1 || batch_size < 1 || num_threads < 1) {
        PyErr_SetString(PyExc_ValueError, "Parameters must be >= 1");
        return NULL;
    }

    int sims_per_thread = total_sims / num_threads;
    std::vector<std::thread> workers;
    std::vector<double> thread_durations(num_threads, 0.0);
    std::vector<int> thread_sims(num_threads, 0);

    auto start_time = std::chrono::high_resolution_clock::now();

    Py_BEGIN_ALLOW_THREADS
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([t, sims_per_thread, batch_size, &thread_durations, &thread_sims]() {
            MCTSConfig cfg;
            cfg.proofs = true;
            cfg.reuse = true;
            MCTSEngine engine(cfg, 1000 + t * 47);
            BoardState initial = BoardState::initial();

            auto t_start = std::chrono::high_resolution_clock::now();
            engine.run_heuristic(initial, sims_per_thread, batch_size);
            auto t_end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> diff = t_end - t_start;
            thread_durations[t] = diff.count();
            thread_sims[t] = engine.stats.completed_simulations;
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) {
            w.join();
        }
    }
    Py_END_ALLOW_THREADS

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_time - start_time;

    int total_completed = 0;
    for (int s : thread_sims) total_completed += s;

    double sims_per_sec = (total_duration.count() > 0) ? (total_completed / total_duration.count()) : 0.0;

    PyObject* res = PyDict_New();
    PyDict_SetItemString(res, "total_simulations", PyLong_FromLong(total_completed));
    PyDict_SetItemString(res, "batch_size", PyLong_FromLong(batch_size));
    PyDict_SetItemString(res, "num_threads", PyLong_FromLong(num_threads));
    PyDict_SetItemString(res, "elapsed_seconds", PyFloat_FromDouble(total_duration.count()));
    PyDict_SetItemString(res, "simulations_per_sec", PyFloat_FromDouble(sims_per_sec));

    return res;
}

static PyMethodDef py_benchmark_mcts_def = {
    "benchmark_mcts", (PyCFunction)py_benchmark_mcts, METH_VARARGS, "Benchmark high-performance C++ MCTS engine"
};

// Module initialization hook
extern "C" void init_mcts_module(PyObject* m) {
    PyFastNodeType.tp_name = "sttt_cpp.FastNode";
    PyFastNodeType.tp_basicsize = sizeof(PyFastNode);
    PyFastNodeType.tp_itemsize = 0;
    PyFastNodeType.tp_flags = Py_TPFLAGS_DEFAULT;
    PyFastNodeType.tp_doc = "Fast C++ MCTS Node";
    PyFastNodeType.tp_dealloc = (destructor)PyFastNode_dealloc;
    PyFastNodeType.tp_getset = PyFastNode_getseters;

    if (PyType_Ready(&PyFastNodeType) < 0) {
        return;
    }

    PyFastTreeSearchType.tp_name = "sttt_cpp.FastTreeSearch";
    PyFastTreeSearchType.tp_basicsize = sizeof(PyFastTreeSearch);
    PyFastTreeSearchType.tp_itemsize = 0;
    PyFastTreeSearchType.tp_flags = Py_TPFLAGS_DEFAULT;
    PyFastTreeSearchType.tp_doc = "Fast C++ Monte Carlo Tree Search Engine";
    PyFastTreeSearchType.tp_new = PyFastTreeSearch_new;
    PyFastTreeSearchType.tp_init = (initproc)PyFastTreeSearch_init;
    PyFastTreeSearchType.tp_dealloc = (destructor)PyFastTreeSearch_dealloc;
    PyFastTreeSearchType.tp_methods = PyFastTreeSearch_methods;
    PyFastTreeSearchType.tp_getset = PyFastTreeSearch_getseters;

    if (PyType_Ready(&PyFastTreeSearchType) < 0) {
        return;
    }

    Py_INCREF(&PyFastNodeType);
    PyModule_AddObject(m, "FastNode", (PyObject*)&PyFastNodeType);

    Py_INCREF(&PyFastTreeSearchType);
    PyModule_AddObject(m, "FastTreeSearch", (PyObject*)&PyFastTreeSearchType);
    Py_INCREF(&PyFastTreeSearchType);
    PyModule_AddObject(m, "TreeSearch", (PyObject*)&PyFastTreeSearchType);

    PyObject* py_bench_fn = PyCFunction_NewEx(&py_benchmark_mcts_def, NULL, NULL);
    if (py_bench_fn) {
        PyModule_AddObject(m, "benchmark_mcts", py_bench_fn);
    }
}
