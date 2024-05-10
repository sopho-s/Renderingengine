
#include <stdlib.h>
#include <stdio.h>
namespace Pipeline {
    namespace InputAssembler {
        __global__ void AssemblePoints(float* vertices, float4* out);
    }
}