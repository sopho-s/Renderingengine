add_library(AMaths AMaths.cuh AMaths.cu)
add_library(GMaths GMaths.cuh GMaths.cu)
include_directories(../Includes)
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH PARENT_DIR)
target_compile_features(AMaths PUBLIC cxx_std_11)
target_compile_features(GMaths PUBLIC cxx_std_11)
#set_target_properties(AMaths PROPERTIES
#                            CUDA_SEPARABLE_COMPILATION ON)