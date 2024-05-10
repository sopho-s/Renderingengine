#include "GMaths.cuh"
namespace GMaths {
    __global__ void VecAdd(float* vec1, float* vec2, float* out) {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        out[i] = vec1[i] + vec2[i];
    }
	__global__ void MatMulVec4(float* matrix[16], float* vector[4], float* out[4]) {
        int temp = threadIdx.x + blockDim.x * blockIdx.x;
        int i = temp % 4;
        int s = (int)(temp / (4));
        out[s][i] = matrix[s][i] * vector[s][0] + matrix[s][i+4] * vector[s][1] + matrix[s][i+8] * vector[s][2] + matrix[s][i+12] * vector[s][3];
    }
}