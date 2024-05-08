#include "Renderingengine.cuh"

using namespace std;

__global__ void DrawCallCheck(int* drawcall, bool* out) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i == 0) {
		if ((drawcall[0] == GPUInterface::DrawType::Triangle) &&
			(drawcall[1] == 3) &&
			(drawcall[2] == -1) &&
			(drawcall[3] == 1) &&
			(drawcall[4] == 2) &&
			(drawcall[5] == 3)) {
				out[0] = true;
		} else {
			out[0] = false;
		}
	}
}

int main()
{
	std::cout << "PERFORMING DRAW CALL CHECK" << std::endl;
	GPUInterface::DrawCall DrawCalltest(GPUInterface::DrawType::Triangle, 3, -1, new int[3] {1, 2, 3});
	bool* gpuout;
	cudaMalloc(&gpuout, sizeof(bool));
	int* datacpu;
	DrawCalltest.Dump(datacpu);
	int* datagpu;
	cudaMalloc(&datagpu, sizeof(int) * (3+datacpu[1]));
	cudaMemcpy(datagpu, datacpu, sizeof(int) * (3+datacpu[1]), cudaMemcpyHostToDevice);
	DrawCallCheck <<< 1, 1 >>> (datagpu, gpuout);
	cudaDeviceSynchronize();
	bool* out = new bool[1];
	cudaMemcpy(out, gpuout, sizeof(bool), cudaMemcpyDeviceToHost);
	if (out[0]) {
		std::cout << "DRAW CALL OK!" << std::endl;
	} else {
		std::cout << "DRAW CALL DID NOT ARRIVE AT GPU IN PROPER STATE" << std::endl;
		throw Errors::Exception("DRAW CALL DID NOT ARRIVE AT GPU IN PROPER STATE");
	}
	cudaFree(gpuout);
	cudaFree(datagpu);
	std::cout << "FINISHED" << std::endl;
	return 0;
}
