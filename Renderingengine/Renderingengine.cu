#include "Renderingengine.cuh"

using namespace std;

__global__ void DrawCallCheck(int* drawcall, bool* out) {
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

__global__ void InputAssemblerCheck(int* drawcall, bool* out) {
	float4 *float4out;
	cudaMalloc(&float4out, sizeof(float4));
	Pipeline::InputAssembler::AssemblePoints <<< 1, 1 >>> (new float[3] {1, 2, 3}, float4out);
	if ((float4out[0].x == 1) &&
		(float4out[0].y == 2) &&
		(float4out[0].z == 3) &&
		(float4out[0].w == 0)) {
		out[0] = true;
	} else {
		out[0] = false;
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
	std::cout << "PERFORMING INPUT ASSEMBLER CHECK" << std::endl;
	InputAssemblerCheck <<< 1, 1 >>> (datagpu, gpuout);
	cudaDeviceSynchronize();
	out = new bool[1];
	cudaMemcpy(out, gpuout, sizeof(bool), cudaMemcpyDeviceToHost);
	if (out[0]) {
		std::cout << "DATA SENT TO INPUT ASSEMBLER WAS RECIEVED AND FORMATTED PROPERLY!" << std::endl;
	} else {
		std::cout << "DATA SENT TO THE INPUT ASSEMBLER WAS EITHER NOT RECEIVED IN THE RIGHT FORMAT OR THERE WAS AN ERROR WHILE FORMATTING" << std::endl;
		throw Errors::Exception("DATA SENT TO THE INPUT ASSEMBLER WAS EITHER NOT RECEIVED IN THE RIGHT FORMAT OR THERE WAS AN ERROR WHILE FORMATTING");
	}
	cudaDeviceSynchronize();
	cudaFree(gpuout);
	cudaFree(datagpu);
	std::cout << "FINISHED" << std::endl;
	return 0;
}
