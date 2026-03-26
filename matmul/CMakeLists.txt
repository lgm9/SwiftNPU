# tt_simple/multi_core_reuse/CMakeLists.txt
cmake_minimum_required(VERSION 3.22...3.30)
project(matmul)

add_subdirectory(matmul_common)
add_subdirectory(matmul_multi_core)
add_subdirectory(matmul_multicore_reuse)
add_subdirectory(matmul_multicore_multitenant)
add_subdirectory(matmul_multicore_anyshape)