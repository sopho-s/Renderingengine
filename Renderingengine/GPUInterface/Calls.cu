#include "Calls.cuh"
namespace GPUInterface {
    Caller::Caller() {
        startpointer = 0;
        endpointer = 0;
    }
    Caller::Caller operator<<(int* data) {
        int count = 0;
        while (data[count] != -1) {
            if (endpointer > 500) {
                endpointer = -1;
            }
            endpointer++;
            stack[endpointer] = data[count];
            count++;
        }
    }
}