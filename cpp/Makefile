CXX ?= g++
CXXFLAGS ?= -O3 -march=native -Wall -Wextra -std=c++17 -fPIC -pthread -I./include
PYTHON ?= /home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python
PY_INCLUDE ?= /usr/include/python3.14

all: libsttt_core.so bench_main sttt_cpp.so

libsttt_core.so: src/sttt_c_api.cpp include/sttt_core.hpp include/sttt_search.hpp include/sttt_c_api.h
	$(CXX) $(CXXFLAGS) -shared -o libsttt_core.so src/sttt_c_api.cpp

bench_main: benchmarks/bench_main.cpp include/sttt_core.hpp include/sttt_search.hpp
	$(CXX) $(CXXFLAGS) -o bench_main benchmarks/bench_main.cpp

sttt_cpp.so: src/python_module.cpp src/sttt_c_api.cpp include/sttt_core.hpp include/sttt_search.hpp
	$(CXX) $(CXXFLAGS) -I$(PY_INCLUDE) -shared -o sttt_cpp.so src/python_module.cpp src/sttt_c_api.cpp

clean:
	rm -f libsttt_core.so bench_main sttt_cpp.so

.PHONY: all clean
