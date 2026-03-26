add_executable(tt_matmul_multicore_anyshape)
target_sources(tt_matmul_multicore_anyshape PRIVATE matmul_multicore_anyshape.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    tt_matmul_multicore_anyshape
    PRIVATE
        TT::Metalium
        Matmul::Common
)

target_include_directories(tt_matmul_multicore_anyshape PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tt_matmul_multicore_anyshape PRIVATE TT::Metalium TT::STL)