#include "InputAssembler.cuh"
namespace Pipeline {
    namespace InputAssembler {
        __global__ void AssemblePoints(float* vertices, float4* out)  {
            int i = threadIdx.x + blockDim.x * blockIdx.x;
            out[i].x = vertices[i*3];
            out[i].y = vertices[i*3 + 1];
            out[i].z = vertices[i*3 + 2];
            out[i].w = 0;
        }
    }
}