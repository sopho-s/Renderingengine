add_library(Calls Calls.cu Calls.cuh)
add_library(IdleWork IdleWork.cu IdleWork.cuh)
target_compile_features(IdleWork PUBLIC cuda_std_11 )
set_target_properties(IdleWork PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 