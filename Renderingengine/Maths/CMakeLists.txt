add_library(AMaths AMaths.cuh AMaths.cu)
add_library(GMaths GMaths.cuh GMaths.cu)
target_compile_features(AMaths PUBLIC cuda_std_11 )
target_compile_features(GMaths PUBLIC cuda_std_11 )
set_target_properties(AMaths PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 
set_target_properties(GMaths PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 
#set_target_properties(AMaths PROPERTIES
#                            CUDA_SEPARABLE_COMPILATION ON)