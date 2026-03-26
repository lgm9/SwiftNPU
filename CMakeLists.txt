cmake_minimum_required(VERSION 3.16)
project(SwiftNPU)

set(TT_METAL_ARCH "blackhole" CACHE STRING "TT target arch")
add_subdirectory(tt-metal)

add_subdirectory(matmul)
add_subdirectory(source)