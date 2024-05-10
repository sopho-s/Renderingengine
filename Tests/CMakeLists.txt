cmake_minimum_required (VERSION 3.8)

project (Testing LANGUAGES CUDA CXX)

include_directories(Tester ../Renderingengine/Maths)

add_subdirectory(Tester)

add_executable(Tests Tests.cu)

set_target_properties(Tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Tests PRIVATE Tester AMaths GMaths)
