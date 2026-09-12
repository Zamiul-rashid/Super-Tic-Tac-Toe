#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "sttt_core.hpp"
#include "sttt_search.hpp"
#include "sttt_c_api.h"

using namespace sttt;

typedef struct {
    PyObject_HEAD
    BoardState state;
} PyFastState;

static PyTypeObject PyFastStateType = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

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

    if (py_cells == NULL && py_boards == NULL) {
        self->state = BoardState::initial();
        self->state.turn = static_cast<int8_t>(turn);
        self->state.forced = static_cast<int8_t>(forced);
        if (py_result != Py_None) {
            self->state.result = static_cast<int8_t>(PyLong_AsLong(py_result));
        }
        return 0;
    }

    std::memset(&self->state, 0, sizeof(BoardState));

    if (py_cells != NULL) {
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
                if (val == 1) {
                    self->state.x_cells[b] |= (1 << c);
                } else if (val == -1) {
                    self->state.o_cells[b] |= (1 << c);
                }
            }
        }
        Py_DECREF(seq);
    }

    if (py_boards != NULL) {
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
            if (val == 1) {
                self->state.macro_x |= (1 << b);
            } else if (val == -1) {
                self->state.macro_o |= (1 << b);
            } else if (val == 2) {
                self->state.macro_draw |= (1 << b);
            }
        }
        Py_DECREF(seq);
    }

    self->state.turn = static_cast<int8_t>(turn);
    self->state.forced = static_cast<int8_t>(forced);
    if (py_result == Py_None) {
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

static PyObject* PyFastState_render(PyFastState* self, PyObject* Py_UNUSED(ignored)) {
    std::string s = self->state.render();
    return PyUnicode_FromStringAndSize(s.c_str(), s.size());
}

static PyMethodDef PyFastState_methods[] = {
    {"legal_actions", (PyCFunction)PyFastState_legal_actions, METH_NOARGS, "Return list of legal actions (0..80)"},
    {"play", (PyCFunction)PyFastState_play, METH_VARARGS, "Play an action, returning a new FastState"},
    {"play_inplace", (PyCFunction)PyFastState_play_inplace, METH_VARARGS, "Play an action in-place on this FastState"},
    {"encode", (PyCFunction)PyFastState_encode, METH_NOARGS, "Encode canonical 289 float features"},
    {"render", (PyCFunction)PyFastState_render, METH_NOARGS, "Render board as ASCII string"},
    {NULL, NULL, 0, NULL}
};

static PyObject* PyFastState_richcompare(PyObject* v, PyObject* w, int op) {
    if (!PyObject_TypeCheck(v, &PyFastStateType) || !PyObject_TypeCheck(w, &PyFastStateType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    PyFastState* a = (PyFastState*)v;
    PyFastState* b = (PyFastState*)w;
    bool eq = (a->state == b->state);
    if (op == Py_EQ) {
        if (eq) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    } else if (op == Py_NE) {
        if (!eq) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    }
    Py_RETURN_NOTIMPLEMENTED;
}

static Py_hash_t PyFastState_hash(PyObject* self) {
    PyFastState* s = (PyFastState*)self;
    uint64_t h = 14695981039346656037ULL;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&s->state);
    for (size_t i = 0; i < sizeof(BoardState); ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return static_cast<Py_hash_t>(h);
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

static PyObject* py_alphabeta(PyObject* /*self*/, PyObject* args) {
    PyObject* state_obj;
    int depth = 3;
    unsigned long long budget = 100000;

    if (!PyArg_ParseTuple(args, "O|iK", &state_obj, &depth, &budget)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(state_obj, &PyFastStateType)) {
        PyErr_SetString(PyExc_TypeError, "Expected a FastState instance");
        return NULL;
    }

    PyFastState* s = (PyFastState*)state_obj;
    AlphaBetaSearcher searcher;
    int action;

    Py_BEGIN_ALLOW_THREADS
    action = searcher.choose_move(s->state, depth, budget);
    Py_END_ALLOW_THREADS

    return Py_BuildValue("iK", action, searcher.node_count);
}

static PyMethodDef ModuleMethods[] = {
    {"benchmark_rollouts", (PyCFunction)py_benchmark_rollouts, METH_VARARGS, "Run C++ bitboard rollouts benchmark (num_games, num_threads, seed)"},
    {"alphabeta", (PyCFunction)py_alphabeta, METH_VARARGS, "Run C++ Alpha-Beta search (state, depth, budget)"},
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

PyMODINIT_FUNC PyInit_sttt_cpp(void) {
    PyFastStateType.tp_name = "sttt_cpp.FastState";
    PyFastStateType.tp_basicsize = sizeof(PyFastState);
    PyFastStateType.tp_itemsize = 0;
    PyFastStateType.tp_flags = Py_TPFLAGS_DEFAULT;
    PyFastStateType.tp_doc = "Fast C++ Bitboard State for Ultimate Tic-Tac-Toe";
    PyFastStateType.tp_new = PyFastState_new;
    PyFastStateType.tp_init = (initproc)PyFastState_init;
    PyFastStateType.tp_repr = (reprfunc)PyFastState_repr;
    PyFastStateType.tp_hash = (hashfunc)PyFastState_hash;
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
    Py_INCREF(&PyFastStateType);
    if (PyModule_AddObject(m, "FastState", (PyObject*)&PyFastStateType) < 0) {
        Py_DECREF(&PyFastStateType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
