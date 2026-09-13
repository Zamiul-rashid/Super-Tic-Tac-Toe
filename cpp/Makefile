CXX ?= g++
PYTHON ?= $(shell if [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; elif which python3 >/dev/null 2>&1; then which python3; else echo python; fi)
CXXFLAGS ?= -O3 -march=native -Wall -Wextra -Werror -std=c++17 -fPIC -pthread -I./include

# Dynamically resolve include directory and extension suffix from $(PYTHON)
PY_INCLUDE ?= $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || echo "/usr/include/python3")
EXT_SUFFIX ?= $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')" 2>/dev/null || echo ".so")

GIT_REV ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_ID ?= $(shell date -u +'%Y%m%d%H%M%S')_$(GIT_REV)
VERSION ?= 0.9.0

CPP_DEFINES = -DSTTT_CPP_VERSION=\"$(VERSION)\" \
              -DSTTT_BUILD_ID=\"$(BUILD_ID)\" \
              -DSTTT_SOURCE_REVISION=\"$(GIT_REV)\" \
              -DSTTT_COMPILER_FLAGS='"$(CXXFLAGS)"'

all: libsttt_core.so bench_main sttt_cpp.so

libsttt_core.so: src/sttt_c_api.cpp include/sttt_core.hpp include/sttt_search.hpp include/sttt_c_api.h
	$(CXX) $(CXXFLAGS) -shared -o libsttt_core.so src/sttt_c_api.cpp

bench_main: benchmarks/bench_main.cpp include/sttt_core.hpp include/sttt_search.hpp include/sttt_mcts.hpp
	$(CXX) $(CXXFLAGS) -o bench_main benchmarks/bench_main.cpp

sttt_cpp.so: src/python_module.cpp src/sttt_c_api.cpp src/mcts.cpp include/sttt_core.hpp include/sttt_search.hpp include/sttt_mcts.hpp
	$(CXX) $(CXXFLAGS) $(CPP_DEFINES) -I$(PY_INCLUDE) -shared -o sttt_cpp.so src/python_module.cpp src/sttt_c_api.cpp src/mcts.cpp
	@if [ "$(EXT_SUFFIX)" != ".so" ] && [ -n "$(EXT_SUFFIX)" ]; then cp -f sttt_cpp.so sttt_cpp$(EXT_SUFFIX) 2>/dev/null || true; fi

clean:
	rm -f libsttt_core.so bench_main sttt_cpp.so sttt_cpp*.so

.PHONY: all clean

