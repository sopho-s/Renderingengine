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

__device__ float vertices[9];
__global__ void InputAssemblerCheck(int* drawcall, float* vertexholder, float4* float4out, bool* out) {
	Pipeline::InputAssembler::AssemblePoints <<< 1, 1 >>> (vertexholder, float4out);
	GPUInterface::GlobalIdle <<< 1, 1 >>> ();
	if ((float4out[0].x == (float)1) &&
		(float4out[0].y == (float)2) &&
		(float4out[0].z == (float)3) &&
		(float4out[0].w == (float)0)) {
		out[0] = true;
	} else {
		out[0] = false;
	}
}

int main()
{
	std::cout << "PERFORMING DRAW CALL CHECK" << std::endl;
	// creates draw call object
	GPUInterface::DrawCall DrawCalltest(GPUInterface::DrawType::Triangle, 3, -1, new int[3] {1, 2, 3});
	bool* gpuout;
	cudaMalloc(&gpuout, sizeof(bool));
	int* datacpu;
	// dumps drawcall into a int array
	DrawCalltest.Dump(datacpu);
	int* datagpu;
	cudaMalloc(&datagpu, sizeof(int) * (3+datacpu[1]));
	// copies the drawcall to the gpu
	cudaMemcpy(datagpu, datacpu, sizeof(int) * (3+datacpu[1]), cudaMemcpyHostToDevice);
	// performs check
	DrawCallCheck <<< 1, 1 >>> (datagpu, gpuout);
	// waits for the kernel to finish
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
		throw Errors::Exception("CUDA ERROR");
    }
	cudaFree(datagpu);
	bool* out = new bool[1];
	// copies the result back
	cudaMemcpy(out, gpuout, sizeof(bool), cudaMemcpyDeviceToHost);
	// checks if the draw call transfers ok
	if (out[0]) {
		std::cout << "DRAW CALL OK!" << std::endl;
	} else {
		std::cout << "DRAW CALL DID NOT ARRIVE AT GPU IN PROPER STATE" << std::endl;
		throw Errors::Exception("DRAW CALL DID NOT ARRIVE AT GPU IN PROPER STATE");
	}
	std::cout << "PERFORMING INPUT ASSEMBLER CHECK" << std::endl;
	float4* float4out;
	float* temp;
	cudaMalloc(&float4out, sizeof(float4));
    cudaMalloc(&temp, 3 * sizeof(float));
	// fills up fake verticies
	float myvertices[9];
	for (int i=0; i< 9; i++) {
		myvertices[i] = (float)(i+1);
	}
	std::cout << myvertices[6] << std::endl;
	// copies data across
	cudaMemcpyToSymbol(vertices, myvertices, 9*sizeof(float));
	cudaMalloc(&gpuout, sizeof(bool));
	datacpu[0] = 0;
	datacpu[1] = 1;
	datacpu[2] = 2;
	// copies the indices across
	cudaMemcpy(datagpu, datacpu, sizeof(int) * 3, cudaMemcpyHostToDevice);
	// performs input assembler check
	InputAssemblerCheck <<< 1, 1 >>> (datagpu, temp, float4out, gpuout);
	// waits for kernel 
	cudaDeviceSynchronize();
	// checks for error
	err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
		throw Errors::Exception("CUDA ERROR");
    }
	out = new bool[1];
	// gets the result
	cudaMemcpy(out, gpuout, sizeof(bool), cudaMemcpyDeviceToHost);
	// checks the result
	if (out[0]) {
		std::cout << "DATA SENT TO INPUT ASSEMBLER WAS RECIEVED AND FORMATTED PROPERLY!" << std::endl;
	} else {
		std::cout << "DATA SENT TO THE INPUT ASSEMBLER WAS EITHER NOT RECEIVED IN THE RIGHT FORMAT OR THERE WAS AN ERROR WHILE FORMATTING" << std::endl;
		throw Errors::Exception("DATA SENT TO THE INPUT ASSEMBLER WAS EITHER NOT RECEIVED IN THE RIGHT FORMAT OR THERE WAS AN ERROR WHILE FORMATTING");
	}
	cudaDeviceSynchronize();
	cudaFree(gpuout);
	std::cout << "FINISHED" << std::endl;
	return 0;
}
