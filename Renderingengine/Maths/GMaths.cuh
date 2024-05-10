#ifndef GMATHS_CUH
#define GMATHS_CUH
namespace GMaths {
    __device__ void Add(float val1, float val2, float &out);
    __global__ void VecAdd(float* vec1, float* vec2, float* out);
	__global__ void MatMulVec4(float* matrix[16], float* vector[4], float* out[4]);
}
#endif