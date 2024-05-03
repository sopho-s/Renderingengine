namespace GMaths {
    __global__ void VecAdd(float* vec1, float* vec2, float* out) {
        int i = threadIdx.x;
        out[i] = vec1[i] + vec2[i];
    }
}