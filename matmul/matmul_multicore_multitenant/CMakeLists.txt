add_executable(tt_matmul_multicore_multitenant)
target_sources(tt_matmul_multicore_multitenant PRIVATE matmul_multicore_multitenant.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    tt_matmul_multicore_multitenant
    PRIVATE
        TT::Metalium
        Matmul::Common
)

target_include_directories(tt_matmul_multicore_multitenant PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tt_matmul_multicore_multitenant PRIVATE TT::Metalium TT::STL)