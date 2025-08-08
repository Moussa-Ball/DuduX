cmake_minimum_required(VERSION 3.12)
project(dudux_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DUDUX_ENABLE_NATIVE "Enable -march=native" ON)
option(DUDUX_ENABLE_METRICS "Enable runtime metrics (popcount counters)" ON)
option(DUDUX_ENABLE_CUDA "Build optional CUDA backend (GPU kernels)" OFF)
if(DUDUX_ENABLE_NATIVE)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-O3 -DNDEBUG -march=native -mtune=native -Wall -Wextra -Wpedantic)
  endif()
endif()

if (DUDUX_ENABLE_METRICS)
  add_compile_definitions(DUDUX_ENABLE_METRICS)
endif()

add_library(dudux_core INTERFACE)
 target_include_directories(dudux_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Optional CUDA backend
if (DUDUX_ENABLE_CUDA)
  enable_language(CUDA)
  add_library(cudux_gpu STATIC src/gpu/attention1b.cu)
  target_include_directories(cudux_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(cudux_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # Allow setting arch externally: -DCMAKE_CUDA_ARCHITECTURES=75
  if (NOT CMAKE_CUDA_ARCHITECTURES)
  # Default to a broad baseline if not specified; override with -DCMAKE_CUDA_ARCHITECTURES=XX
  # Examples: 61 (Pascal), 70 (Volta), 75 (Turing), 86 (Ampere)
  set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
endif()

add_executable(dudux src/main.cpp)
 target_include_directories(dudux PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
 target_link_libraries(dudux PRIVATE dudux_core)
 if (DUDUX_ENABLE_CUDA)
   target_link_libraries(dudux PRIVATE cudux_gpu)
 endif()

# Benchmark executable
add_executable(dudux_perf src/benchmark.cpp)
 target_include_directories(dudux_perf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
 target_link_libraries(dudux_perf PRIVATE dudux_core)
 if (DUDUX_ENABLE_CUDA)
   target_link_libraries(dudux_perf PRIVATE cudux_gpu)
 endif()
 # Benchmark defaults configurable via CMake cache variables (numeric/string)
 # (Optional) You can set runtime args; compile-time defaults are defined in source.

# Tests
include(CTest)
if (BUILD_TESTING)
  # Core/bitops tests
  add_executable(dudux_unit_tests tests/test_bitops.cpp)
  target_include_directories(dudux_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(dudux_unit_tests PRIVATE dudux_core)
  add_test(NAME dudux_unit_tests COMMAND dudux_unit_tests)

  # Router tests (separate binary to avoid multiple mains)
  add_executable(dudux_router_tests tests/test_router.cpp)
  target_include_directories(dudux_router_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(dudux_router_tests PRIVATE dudux_core)
  add_test(NAME dudux_router_tests COMMAND dudux_router_tests)

  # NN tests
  add_executable(dudux_nn_tests tests/test_nn.cpp)
  target_include_directories(dudux_nn_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(dudux_nn_tests PRIVATE dudux_core)
  add_test(NAME dudux_nn_tests COMMAND dudux_nn_tests)

  add_executable(dudux_mlp_tests tests/test_mlp.cpp)
  target_include_directories(dudux_mlp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(dudux_mlp_tests PRIVATE dudux_core)
  add_test(NAME dudux_mlp_tests COMMAND dudux_mlp_tests)

  # Attention tests
  add_executable(dudux_attention_tests tests/test_attention.cpp)
  target_include_directories(dudux_attention_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(dudux_attention_tests PRIVATE dudux_core)
  add_test(NAME dudux_attention_tests COMMAND dudux_attention_tests)

  add_executable(dudux_mha_tests tests/test_mha.cpp)
  target_include_directories(dudux_mha_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(dudux_mha_tests PRIVATE dudux_core)
  add_test(NAME dudux_mha_tests COMMAND dudux_mha_tests)

  add_executable(dudux_mha_multistream_tests tests/test_mha_multistream.cpp)
  target_include_directories(dudux_mha_multistream_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(dudux_mha_multistream_tests PRIVATE dudux_core)
  if (DUDUX_ENABLE_CUDA)
    target_link_libraries(dudux_mha_multistream_tests PRIVATE cudux_gpu)
  endif()
  add_test(NAME dudux_mha_multistream_tests COMMAND dudux_mha_multistream_tests)

  # Training sample (not a test by default)
  add_executable(dudux_train_ste src/train_ste.cpp)
  target_include_directories(dudux_train_ste PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(dudux_train_ste PRIVATE dudux_core)

  add_executable(dudux_train_mlp_ste src/train_mlp_ste.cpp)
  target_include_directories(dudux_train_mlp_ste PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(dudux_train_mlp_ste PRIVATE dudux_core)

  # Smoke test for MLP STE trainer (small run to verify it executes)
  add_test(NAME dudux_train_mlp_ste_smoke
           COMMAND dudux_train_mlp_ste 64 16 64 32 2 0.1 1)

  option(DUDUX_ENABLE_PERF_TEST "Enable perf smoke test" OFF)
  if (DUDUX_ENABLE_PERF_TEST)
    add_test(NAME dudux_perf_smoke COMMAND dudux_perf)
    set_tests_properties(dudux_perf_smoke PROPERTIES LABELS "perf")
  endif()

  # Small tool to check packed MLP files
  add_executable(dudux_check_packed_mlp tools/check_packed_mlp.cpp)
  # Optional: we could add an end-to-end export+check test later.
endif()
