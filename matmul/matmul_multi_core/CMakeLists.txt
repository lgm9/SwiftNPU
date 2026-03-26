add_executable(tt_matmul_multi_core)
target_sources(tt_matmul_multi_core PRIVATE matmul_multi_core.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    tt_matmul_multi_core
    PRIVATE
        TT::Metalium
        Matmul::Common
)

target_include_directories(tt_matmul_multi_core PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tt_matmul_multi_core PRIVATE TT::Metalium TT::STL)