#include "Calls.cuh"
namespace GPUInterface {
    __global__ void GlobalIdle() {
        for (int i = 0; i < 1; i++) {
            ;
        }
    }
}