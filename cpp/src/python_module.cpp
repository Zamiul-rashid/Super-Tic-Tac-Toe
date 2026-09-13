#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "sttt_core.hpp"
#include "sttt_search.hpp"
#include "sttt_c_api.h"
#include <vector>
#include <exception>

#ifndef STTT_CPP_VERSION
#define STTT_CPP_VERSION "0.9.0"
#endif

#ifndef STTT_BUILD_ID
#define STTT_BUILD_ID "unknown"
#endif

#ifndef STTT_SOURCE_REVISION
#define STTT_SOURCE_REVISION "unknown"
#endif

#ifndef STTT_COMPILER_FLAGS
#define STTT_COMPILER_FLAGS ""
#endif

using namespace sttt;

typedef struct {
    PyObject_HEAD
    BoardState state;
} PyFastState;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
PyTypeObject PyFastStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
};
#pragma GCC diagnostic pop

static PyObject* PyFastState_new(PyTypeObject* type, PyObject* /*args*/, PyObject* /*kwds*/) {
    PyFastState* self = (PyFastState*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->state = BoardState::initial();
    }
    return (PyObject*)self;
}

static int PyFastState_init(PyFastState* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"cells", "boards", "turn", "forced", "result", NULL};
    PyObject* py_cells = NULL;
    PyObject* py_boards = NULL;
    int turn = 1;
    int forced = -1;
    PyObject* py_result = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOiiO", (char**)kwlist,
                                     &py_cells, &py_boards, &turn, &forced, &py_result)) {
        return -1;
    }

    bool cells_given = (py_cells != NULL && py_cells != Py_None);
    bool boards_given = (py_boards != NULL && py_boards != Py_None);

    if (!cells_given && !boards_given) {
        self->state = BoardState::initial();
        self->state.turn = static_cast<int8_t>(turn);
        self->state.forced = static_cast<int8_t>(forced);
        if (py_result != Py_None && py_result != NULL) {
            self->state.result = static_cast<int8_t>(PyLong_AsLong(py_result));
        }
        return 0;
    }

    std::memset(&self->state, 0, sizeof(BoardState));

    if (cells_given) {
        PyObject* seq = PySequence_Fast(py_cells, "cells must be a sequence of 81 ints");
        if (!seq) return -1;
        if (PySequence_Fast_GET_SIZE(seq) != 81) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "cells sequence must have length 81");
            return -1;
        }
        for (int b = 0; b < 9; ++b) {
            for (int c = 0; c < 9; ++c) {
                PyObject* item = PySequence_Fast_GET_ITEM(seq, b * 9 + c);
                long val = PyLong_AsLong(item);
                if (val == -1 && PyErr_Occurred()) {
                    Py_DECREF(seq);
                    return -1;
                }
                if (val == 1) {
                    self->state.x_cells[b] |= (1 << c);
                } else if (val == -1) {
                    self->state.o_cells[b] |= (1 << c);
                } else if (val != 0) {
                    Py_DECREF(seq);
                    PyErr_Format(PyExc_ValueError, "cells elements must be -1, 0, or 1; got %ld", val);
                    return -1;
                }
            }
        }
        Py_DECREF(seq);
    }

    if (boards_given) {
        PyObject* seq = PySequence_Fast(py_boards, "boards must be a sequence of 9 ints");
        if (!seq) return -1;
        if (PySequence_Fast_GET_SIZE(seq) != 9) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "boards sequence must have length 9");
            return -1;
        }
        for (int b = 0; b < 9; ++b) {
            PyObject* item = PySequence_Fast_GET_ITEM(seq, b);
            long val = PyLong_AsLong(item);
            if (val == -1 && PyErr_Occurred()) {
                Py_DECREF(seq);
                return -1;
            }
            if (val == 1) {
                self->state.macro_x |= (1 << b);
            } else if (val == -1) {
                self->state.macro_o |= (1 << b);
            } else if (val == 2) {
                self->state.macro_draw |= (1 << b);
            } else if (val != 0) {
                Py_DECREF(seq);
                PyErr_Format(PyExc_ValueError, "boards elements must be -1, 0, 1, or 2; got %ld", val);
                return -1;
            }
        }
        Py_DECREF(seq);
    }

    self->state.turn = static_cast<int8_t>(turn);
    self->state.forced = static_cast<int8_t>(forced);
    if (py_result == Py_None || py_result == NULL) {
        self->state.result = RESULT_ONGOING;
    } else {
        self->state.result = static_cast<int8_t>(PyLong_AsLong(py_result));
    }

    return 0;
}

static PyObject* PyFastState_get_cells(PyFastState* self, void* /*closure*/) {
    int8_t cells[81];
    self->state.to_cells(cells);
    PyObject* tuple = PyTuple_New(81);
    for (int i = 0; i < 81; ++i) {
        PyTuple_SET_ITEM(tuple, i, PyLong_FromLong(cells[i]));
    }
    return tuple;
}

static PyObject* PyFastState_get_boards(PyFastState* self, void* /*closure*/) {
    int8_t boards[9];
    self->state.to_boards(boards);
    PyObject* tuple = PyTuple_New(9);
    for (int i = 0; i < 9; ++i) {
        PyTuple_SET_ITEM(tuple, i, PyLong_FromLong(boards[i]));
    }
    return tuple;
}

static PyObject* PyFastState_get_turn(PyFastState* self, void* /*closure*/) {
    return PyLong_FromLong(self->state.turn);
}

static PyObject* PyFastState_get_forced(PyFastState* self, void* /*closure*/) {
    return PyLong_FromLong(self->state.forced);
}

static PyObject* PyFastState_get_result(PyFastState* self, void* /*closure*/) {
    if (self->state.result == RESULT_ONGOING) {
        Py_RETURN_NONE;
    }
    return PyLong_FromLong(self->state.result);
}

static PyGetSetDef PyFastState_getseters[] = {
    {(char*)"cells", (getter)PyFastState_get_cells, NULL, (char*)"81-tuple of cells (-1, 0, 1)", NULL},
    {(char*)"boards", (getter)PyFastState_get_boards, NULL, (char*)"9-tuple of local boards status", NULL},
    {(char*)"turn", (getter)PyFastState_get_turn, NULL, (char*)"1 for X, -1 for O", NULL},
    {(char*)"forced", (getter)PyFastState_get_forced, NULL, (char*)"Forced local board or -1", NULL},
    {(char*)"result", (getter)PyFastState_get_result, NULL, (char*)"Game result: None, 1, -1, 0", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyObject* PyFastState_legal_actions(PyFastState* self, PyObject* Py_UNUSED(ignored)) {
    uint8_t actions[81];
    int n = self->state.get_legal_actions(actions);
    PyObject* list = PyList_New(n);
    for (int i = 0; i < n; ++i) {
        PyList_SET_ITEM(list, i, PyLong_FromLong(actions[i]));
    }
    return list;
}

static PyObject* PyFastState_play(PyFastState* self, PyObject* args) {
    int action;
    if (!PyArg_ParseTuple(args, "i", &action)) {
        return NULL;
    }
    if (action < 0 || action >= 81 || !self->state.is_legal(static_cast<uint8_t>(action))) {
        PyErr_Format(PyExc_ValueError, "Illegal action: %d", action);
        return NULL;
    }

    PyFastState* next = (PyFastState*)PyFastStateType.tp_alloc(&PyFastStateType, 0);
    if (!next) return NULL;
    next->state = self->state;
    next->state.play_inplace(static_cast<uint8_t>(action));
    return (PyObject*)next;
}

static PyObject* PyFastState_play_inplace(PyFastState* self, PyObject* args) {
    int action;
    if (!PyArg_ParseTuple(args, "i", &action)) {
        return NULL;
    }
    if (action < 0 || action >= 81 || !self->state.is_legal(static_cast<uint8_t>(action))) {
        PyErr_Format(PyExc_ValueError, "Illegal action: %d", action);
        return NULL;
    }
    self->state.play_inplace(static_cast<uint8_t>(action));
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject* PyFastState_encode(PyFastState* self, PyObject* Py_UNUSED(ignored)) {
    float buf[289];
    self->state.encode(buf);
    PyObject* list = PyList_New(289);
    for (int i = 0; i < 289; ++i) {
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(buf[i]));
    }
    return list;
}

static PyObject* PyFastState_encode_bytes(PyFastState* self, PyObject* Py_UNUSED(ignored)) {
    float buf[289];
    self->state.encode(buf);
    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(buf), sizeof(buf));
}

static PyObject* PyFastState_evaluate(PyFastState* self, PyObject* Py_UNUSED(ignored)) {
    float val = evaluate_state(self->state);
    return PyFloat_FromDouble(val);
}

static PyObject* PyFastState_render(PyFastState* self, PyObject* Py_UNUSED(ignored)) {
    std::string s = self->state.render();
    return PyUnicode_FromStringAndSize(s.c_str(), s.size());
}

static PyObject* PyFastState_reduce(PyFastState* self, PyObject* Py_UNUSED(ignored)) {
    PyObject* cells = PyFastState_get_cells(self, NULL);
    PyObject* boards = PyFastState_get_boards(self, NULL);
    PyObject* turn = PyFastState_get_turn(self, NULL);
    PyObject* forced = PyFastState_get_forced(self, NULL);
    PyObject* result = PyFastState_get_result(self, NULL);

    PyObject* args = PyTuple_Pack(5, cells, boards, turn, forced, result);
    Py_DECREF(cells);
    Py_DECREF(boards);
    Py_DECREF(turn);
    Py_DECREF(forced);
    Py_DECREF(result);

    PyObject* res = PyTuple_Pack(2, (PyObject*)&PyFastStateType, args);
    Py_DECREF(args);
    return res;
}

// Defined below, referenced by the method table above them.
static PyObject* PyFastState_state_key(PyFastState* s, PyObject* ignored);
static PyObject* PyFastState_clone(PyFastState* self, PyObject* ignored);

static PyMethodDef PyFastState_methods[] = {
    {"legal_actions", (PyCFunction)PyFastState_legal_actions, METH_NOARGS, "Return list of legal actions (0..80)"},
    {"play", (PyCFunction)PyFastState_play, METH_VARARGS, "Play an action, returning a new FastState"},
    {"play_inplace", (PyCFunction)PyFastState_play_inplace, METH_VARARGS, "Play an action in-place on this FastState"},
    {"encode", (PyCFunction)PyFastState_encode, METH_NOARGS, "Encode canonical 289 float features as list"},
    {"encode_bytes", (PyCFunction)PyFastState_encode_bytes, METH_NOARGS, "Encode canonical 289 float features as raw bytes"},
    {"evaluate", (PyCFunction)PyFastState_evaluate, METH_NOARGS, "Heuristic evaluation score in [-1, 1] matching sttt.bots.value()"},
    {"value", (PyCFunction)PyFastState_evaluate, METH_NOARGS, "Alias for evaluate() matching sttt.bots.value()"},
    {"render", (PyCFunction)PyFastState_render, METH_NOARGS, "Render board as ASCII string"},
    {"state_key", (PyCFunction)PyFastState_state_key, METH_NOARGS,
     "Immutable canonical (cells, boards, turn, forced, result) tuple for dict/set keys"},
    {"clone", (PyCFunction)PyFastState_clone, METH_NOARGS,
     "Return an independent copy that shares no storage with this state"},
    {"__reduce__", (PyCFunction)PyFastState_reduce, METH_NOARGS, "Pickle serialization support"},
    {NULL, NULL, 0, NULL}
};

static PyObject* PyFastState_richcompare(PyObject* v, PyObject* w, int op) {
    if (op != Py_EQ && op != Py_NE) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (PyObject_TypeCheck(v, &PyFastStateType) && PyObject_TypeCheck(w, &PyFastStateType)) {
        PyFastState* a = (PyFastState*)v;
        PyFastState* b = (PyFastState*)w;
        bool eq = (a->state == b->state);
        if (op == Py_EQ) {
            if (eq) Py_RETURN_TRUE; else Py_RETURN_FALSE;
        } else {
            if (!eq) Py_RETURN_TRUE; else Py_RETURN_FALSE;
        }
    }

    // Cross-comparison with Python State, CppState, or any state-like object
    if (PyObject_TypeCheck(v, &PyFastStateType)) {
        PyFastState* a = (PyFastState*)v;
        if (PyObject_HasAttrString(w, "cells") && PyObject_HasAttrString(w, "boards") &&
            PyObject_HasAttrString(w, "turn") && PyObject_HasAttrString(w, "forced") &&
            PyObject_HasAttrString(w, "result")) {

            PyObject* cells = PyFastState_get_cells(a, NULL);
            PyObject* w_cells = PyObject_GetAttrString(w, "cells");
            int eq_cells = PyObject_RichCompareBool(cells, w_cells, Py_EQ);
            Py_DECREF(cells); Py_DECREF(w_cells);
            if (eq_cells <= 0) {
                if (op == Py_EQ) Py_RETURN_FALSE; else Py_RETURN_TRUE;
            }

            PyObject* boards = PyFastState_get_boards(a, NULL);
            PyObject* w_boards = PyObject_GetAttrString(w, "boards");
            int eq_boards = PyObject_RichCompareBool(boards, w_boards, Py_EQ);
            Py_DECREF(boards); Py_DECREF(w_boards);
            if (eq_boards <= 0) {
                if (op == Py_EQ) Py_RETURN_FALSE; else Py_RETURN_TRUE;
            }

            PyObject* turn = PyFastState_get_turn(a, NULL);
            PyObject* w_turn = PyObject_GetAttrString(w, "turn");
            int eq_turn = PyObject_RichCompareBool(turn, w_turn, Py_EQ);
            Py_DECREF(turn); Py_DECREF(w_turn);
            if (eq_turn <= 0) {
                if (op == Py_EQ) Py_RETURN_FALSE; else Py_RETURN_TRUE;
            }

            PyObject* forced = PyFastState_get_forced(a, NULL);
            PyObject* w_forced = PyObject_GetAttrString(w, "forced");
            int eq_forced = PyObject_RichCompareBool(forced, w_forced, Py_EQ);
            Py_DECREF(forced); Py_DECREF(w_forced);
            if (eq_forced <= 0) {
                if (op == Py_EQ) Py_RETURN_FALSE; else Py_RETURN_TRUE;
            }

            PyObject* result = PyFastState_get_result(a, NULL);
            PyObject* w_result = PyObject_GetAttrString(w, "result");
            int eq_res = PyObject_RichCompareBool(result, w_result, Py_EQ);
            Py_DECREF(result); Py_DECREF(w_result);
            if (eq_res <= 0) {
                if (op == Py_EQ) Py_RETURN_FALSE; else Py_RETURN_TRUE;
            }

            if (op == Py_EQ) Py_RETURN_TRUE; else Py_RETURN_FALSE;
        }
    }

    Py_RETURN_NOTIMPLEMENTED;
}

// M1: FastState is MUTABLE (play_inplace rewrites it), so it must not be
// hashable -- a state used as a dict key would silently move to a different
// bucket the moment it was advanced, and lookups would miss. tp_hash is set to
// PyObject_HashNotImplemented below. Callers that need a dictionary/set key ask
// for this immutable canonical tuple instead, which is identical across the
// Python, FastState and CppState backends so keys stay interchangeable.
static PyObject* PyFastState_state_key(PyFastState* s, PyObject* /*ignored*/) {
    PyObject* cells = PyFastState_get_cells(s, NULL);
    PyObject* boards = PyFastState_get_boards(s, NULL);
    PyObject* result_obj;
    if (s->state.result == RESULT_ONGOING) {
        Py_INCREF(Py_None);
        result_obj = Py_None;
    } else {
        result_obj = PyLong_FromLong(s->state.result);
    }
    PyObject* key = NULL;
    if (cells && boards && result_obj) {
        key = Py_BuildValue("(OOiiO)", cells, boards, (int)s->state.turn,
                            (int)s->state.forced, result_obj);
    }
    Py_XDECREF(cells);
    Py_XDECREF(boards);
    Py_XDECREF(result_obj);
    return key;
}

// An independent copy: the returned state shares no storage with the original,
// so play_inplace on either one cannot be observed by the other.
static PyObject* PyFastState_clone(PyFastState* self, PyObject* /*ignored*/) {
    PyFastState* out = (PyFastState*)PyFastStateType.tp_alloc(&PyFastStateType, 0);
    if (out == NULL) {
        return NULL;
    }
    out->state = self->state;
    return (PyObject*)out;
}

static PyObject* PyFastState_repr(PyFastState* self) {
    int8_t res = self->state.result;
    const char* res_str = (res == RESULT_ONGOING) ? "None" : (res == 1 ? "1" : (res == -1 ? "-1" : "0"));
    return PyUnicode_FromFormat("FastState(turn=%d, forced=%d, result=%s)",
                                self->state.turn, self->state.forced, res_str);
}

static PyObject* py_benchmark_rollouts(PyObject* /*self*/, PyObject* args) {
    int num_games = 10000;
    int num_threads = 1;
    unsigned long long seed = 12345ULL;

    if (!PyArg_ParseTuple(args, "|iiK", &num_games, &num_threads, &seed)) {
        return NULL;
    }

    double elapsed = 0.0;
    uint64_t total_moves = 0;
    Py_BEGIN_ALLOW_THREADS
    sttt_benchmark_rollouts(num_games, seed, num_threads, &elapsed, &total_moves);
    Py_END_ALLOW_THREADS

    double games_per_sec = (elapsed > 0) ? (num_games / elapsed) : 0;
    double moves_per_sec = (elapsed > 0) ? (total_moves / elapsed) : 0;

    return Py_BuildValue("dKd d", elapsed, total_moves, games_per_sec, moves_per_sec);
}

const BoardState* extract_board_state(PyObject* state_obj, PyObject** cleanup) {
    *cleanup = NULL;
    if (PyObject_TypeCheck(state_obj, &PyFastStateType)) {
        return &((PyFastState*)state_obj)->state;
    }
    if (PyObject_HasAttrString(state_obj, "_fast")) {
        PyObject* fast_obj = PyObject_GetAttrString(state_obj, "_fast");
        if (fast_obj && PyObject_TypeCheck(fast_obj, &PyFastStateType)) {
            *cleanup = fast_obj;
            return &((PyFastState*)fast_obj)->state;
        }
        Py_XDECREF(fast_obj);
    }
    if (PyObject_HasAttrString(state_obj, "cells") && PyObject_HasAttrString(state_obj, "boards")) {
        PyObject* cells = PyObject_GetAttrString(state_obj, "cells");
        PyObject* boards = PyObject_GetAttrString(state_obj, "boards");
        PyObject* turn = PyObject_GetAttrString(state_obj, "turn");
        PyObject* forced = PyObject_GetAttrString(state_obj, "forced");
        PyObject* result = PyObject_GetAttrString(state_obj, "result");
        PyObject* args = PyTuple_Pack(5, cells, boards, turn, forced, result);
        Py_XDECREF(cells); Py_XDECREF(boards); Py_XDECREF(turn); Py_XDECREF(forced); Py_XDECREF(result);
        if (args) {
            PyObject* fast_obj = PyObject_CallObject((PyObject*)&PyFastStateType, args);
            Py_DECREF(args);
            if (fast_obj) {
                *cleanup = fast_obj;
                return &((PyFastState*)fast_obj)->state;
            }
        }
    }
    return NULL;
}

static PyObject* py_alphabeta(PyObject* /*self*/, PyObject* args) {
    PyObject* state_obj;
    int depth = 3;
    unsigned long long budget = 100000;

    if (!PyArg_ParseTuple(args, "O|iK", &state_obj, &depth, &budget)) {
        return NULL;
    }

    PyObject* cleanup = NULL;
    const BoardState* bs = extract_board_state(state_obj, &cleanup);
    if (!bs) {
        PyErr_SetString(PyExc_TypeError, "Expected a FastState or CppState instance");
        return NULL;
    }

    if (bs->is_terminal()) {
        Py_XDECREF(cleanup);
        PyErr_SetString(PyExc_ValueError, "Cannot choose a move in a terminal state");
        return NULL;
    }

    AlphaBetaSearcher searcher;
    int action = -1;

    try {
        Py_BEGIN_ALLOW_THREADS
        action = searcher.choose_move(*bs, depth, budget);
        Py_END_ALLOW_THREADS
    } catch (const std::exception& e) {
        Py_XDECREF(cleanup);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    Py_XDECREF(cleanup);
    return Py_BuildValue("iK", action, searcher.node_count);
}

static PyObject* py_evaluate(PyObject* /*self*/, PyObject* args) {
    PyObject* state_obj;
    if (!PyArg_ParseTuple(args, "O", &state_obj)) {
        return NULL;
    }

    PyObject* cleanup = NULL;
    const BoardState* bs = extract_board_state(state_obj, &cleanup);
    if (!bs) {
        PyErr_SetString(PyExc_TypeError, "Expected a FastState or CppState instance");
        return NULL;
    }

    float val = evaluate_state(*bs);
    Py_XDECREF(cleanup);
    return PyFloat_FromDouble(val);
}

static PyObject* py_encode_batch(PyObject* /*self*/, PyObject* args) {
    PyObject* seq_obj;
    if (!PyArg_ParseTuple(args, "O", &seq_obj)) {
        return NULL;
    }

    PyObject* seq = PySequence_Fast(seq_obj, "Expected a sequence of FastState or CppState objects");
    if (!seq) return NULL;

    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n == 0) {
        Py_DECREF(seq);
        return PyBytes_FromStringAndSize("", 0);
    }

    std::vector<float> buffer(n * 289);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        PyObject* cleanup = NULL;
        const BoardState* bs = extract_board_state(item, &cleanup);
        if (!bs) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_TypeError, "Item %zd is not a FastState or CppState", i);
            return NULL;
        }
        bs->encode(buffer.data() + i * 289);
        Py_XDECREF(cleanup);
    }
    Py_DECREF(seq);

    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(buffer.data()),
                                     buffer.size() * sizeof(float));
}

static PyMethodDef ModuleMethods[] = {
    {"benchmark_rollouts", (PyCFunction)py_benchmark_rollouts, METH_VARARGS, "Run C++ bitboard rollouts benchmark (num_games, num_threads, seed)"},
    {"alphabeta", (PyCFunction)py_alphabeta, METH_VARARGS, "Run C++ Alpha-Beta search (state, depth, budget)"},
    {"evaluate", (PyCFunction)py_evaluate, METH_VARARGS, "Heuristic evaluation score in [-1, 1] matching sttt.bots.value()"},
    {"encode_batch", (PyCFunction)py_encode_batch, METH_VARARGS, "Vectorized batch neural encoding for states sequence"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef sttt_cpp_module = {
    PyModuleDef_HEAD_INIT,
    "sttt_cpp",
    "High-Performance Bitboard Engine for Ultimate Tic-Tac-Toe",
    -1,
    ModuleMethods,
    NULL, NULL, NULL, NULL
};

extern "C" void init_mcts_module(PyObject* m);

PyMODINIT_FUNC PyInit_sttt_cpp(void) {
    PyFastStateType.tp_name = "sttt_cpp.FastState";
    PyFastStateType.tp_basicsize = sizeof(PyFastState);
    PyFastStateType.tp_itemsize = 0;
    PyFastStateType.tp_flags = Py_TPFLAGS_DEFAULT;
    PyFastStateType.tp_doc = "Fast C++ Bitboard State for Ultimate Tic-Tac-Toe";
    PyFastStateType.tp_new = PyFastState_new;
    PyFastStateType.tp_init = (initproc)PyFastState_init;
    PyFastStateType.tp_repr = (reprfunc)PyFastState_repr;
    // Mutable -> deliberately unhashable; use state_key() for dict/set keys.
    PyFastStateType.tp_hash = PyObject_HashNotImplemented;
    PyFastStateType.tp_richcompare = PyFastState_richcompare;
    PyFastStateType.tp_methods = PyFastState_methods;
    PyFastStateType.tp_getset = PyFastState_getseters;

    if (PyType_Ready(&PyFastStateType) < 0) {
        return NULL;
    }
    PyObject* m = PyModule_Create(&sttt_cpp_module);
    if (m == NULL) {
        return NULL;
    }

    PyModule_AddStringConstant(m, "__version__", STTT_CPP_VERSION);
    PyModule_AddStringConstant(m, "VERSION", STTT_CPP_VERSION);
    PyModule_AddStringConstant(m, "BUILD_ID", STTT_BUILD_ID);
    PyModule_AddStringConstant(m, "SOURCE_REVISION", STTT_SOURCE_REVISION);
    PyModule_AddStringConstant(m, "COMPILER_FLAGS", STTT_COMPILER_FLAGS);

    Py_INCREF(&PyFastStateType);
    if (PyModule_AddObject(m, "FastState", (PyObject*)&PyFastStateType) < 0) {
        Py_DECREF(&PyFastStateType);
        Py_DECREF(m);
        return NULL;
    }

    init_mcts_module(m);

    return m;
}
