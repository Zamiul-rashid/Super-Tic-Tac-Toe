cmake_minimum_required(VERSION 3.14)
project(sttt_cpp VERSION 0.9 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-O3 -march=native -Wall -Wextra -Werror -pthread)

include_directories(cpp/include)

# C++ Core Shared Library
add_library(sttt_core SHARED
    cpp/src/sttt_c_api.cpp
)
set_target_properties(sttt_core PROPERTIES
    OUTPUT_NAME "sttt_core"
    PREFIX "lib"
)

# Benchmark executable
add_executable(bench_main
    cpp/benchmarks/bench_main.cpp
)
target_link_libraries(bench_main PRIVATE pthread)

# Python Extension module
find_package(Python3 COMPONENTS Development.Module QUIET)
if (Python3_FOUND)
    Python3_add_library(sttt_cpp MODULE
        cpp/src/python_module.cpp
        cpp/src/sttt_c_api.cpp
        cpp/src/mcts.cpp
    )
    target_include_directories(sttt_cpp PRIVATE ${Python3_INCLUDE_DIRS} cpp/include)
    target_link_libraries(sttt_cpp PRIVATE ${Python3_LIBRARIES} pthread)
    set_target_properties(sttt_cpp PROPERTIES PREFIX "")
endif()
