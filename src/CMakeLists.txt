cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(CFR)



find_package(CUDAToolkit)
find_package(Torch REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost COMPONENTS serialization filesystem property_tree)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(CFR CFR.cpp CFR.hpp)
target_link_libraries(CFR "${TORCH_LIBRARIES}")
target_link_libraries(CFR OpenSSL::SSL)
target_link_libraries(CFR OpenSSL::Crypto)
set_property(TARGET CFR PROPERTY CXX_STANDARD 14)
