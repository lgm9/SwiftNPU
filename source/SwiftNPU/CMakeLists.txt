add_executable(SwiftNPU)
target_sources(SwiftNPU PRIVATE main.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    SwiftNPU
    PRIVATE
        TT::Metalium
        Matmul::Common
)

target_include_directories(SwiftNPU PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(SwiftNPU PRIVATE TT::Metalium TT::STL)