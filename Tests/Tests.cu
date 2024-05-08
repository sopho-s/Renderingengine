#include "AMaths.cuh"
#include "GMaths.cuh"
#include "Tester.h"
#include <iostream>
#include <stdlib.h>

void EulerToQuaternion() {
	AMaths::Euler testval;
	testval.yaw = 1;
	testval.pitch = 2;
	testval.roll = 1;
	AMaths::Quaternion quat = AMaths::EulerToQuaternion(testval);
	Tester::ASSERT_NEAR_EQUAL<float>(quat.w, 0.610, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(quat.i, -0.127, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(quat.j, 0.772, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(quat.k, -0.127, 0.001);
}
void QuaternionToEuler() {
	AMaths::Quaternion testval;
	testval.w = 0.610;
	testval.i = -0.127;
	testval.j = 0.772;
	testval.k = -0.127;
	AMaths::Euler eul = AMaths::QuaternionToEuler(testval);
	Tester::ASSERT_NEAR_EQUAL<float>(eul.yaw, -2.1392, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(eul.pitch, 1.1423, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(eul.roll, -2.1392, 0.001);
}
void EQ() {
	AMaths::Euler eul;
	eul.yaw = 9;
	eul.pitch = 22;
	eul.roll = 1;
	AMaths::Quaternion quat = AMaths::EulerToQuaternion(eul);
}
void QE() {
	AMaths::Quaternion quat;
	quat.w = 0.949;
	quat.i = 0.133;
	quat.j = 0.169;
	quat.k = 0.231;
	AMaths::Euler eul = AMaths::QuaternionToEuler(quat);
}

void QEQ() {
	AMaths::Quaternion testval;
	testval.w = 0.610;
	testval.i = -0.127;
	testval.j = 0.772;
	testval.k = -0.127;
	AMaths::Quaternion truth;
	truth.w = 0.610;
	truth.i = -0.127;
	truth.j = 0.772;
	truth.k = -0.127;
	AMaths::Euler eul = AMaths::QuaternionToEuler(testval);
	AMaths::Quaternion quat = AMaths::EulerToQuaternion(eul);
	Tester::ASSERT_NEAR_EQUAL<float>(quat.w, truth.w, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(quat.i, truth.i, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(quat.j, truth.j, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(quat.k, truth.k, 0.001);
}

void QuaternionMultiplication() {
	AMaths::Quaternion inp1;
	inp1.w = 2;
	inp1.i = 1;
	inp1.j = 3;
	inp1.k = -1;
	AMaths::Quaternion inp2;
	inp2.w = 2;
	inp2.i = -0.1;
	inp2.j = -2;
	inp2.k = 9;
	AMaths::Quaternion out;
	AMaths::QuaternionMultiplication(inp1, inp2, out);
	Tester::ASSERT_NEAR_EQUAL<float>(out.w, 19.1, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(out.i, 26.8, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(out.j, -6.9, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(out.k, 14.3, 0.001);
}

void Rotation() {
	AMaths::Quaternion testval;
	testval.w = 0.707;
	testval.i = 0;
	testval.j = 0.707;
	testval.k = 0;
	AMaths::Vector3 out;
	out.x = 1;
	out.y = 0;
	out.z = 0;
	AMaths::Quaternion vecquat;
	AMaths::Vector3ToQuaternion(out, vecquat);
	AMaths::Quaternion inverseout;
	AMaths::InverseQuaternion(testval, inverseout);
	AMaths::Quaternion result1;
	AMaths::QuaternionMultiplication(testval, vecquat, result1);
	AMaths::Quaternion result2;
	AMaths::QuaternionMultiplication(result1, inverseout, result2);
	AMaths::QuaternionToVector3(result2, out);
	Tester::ASSERT_NEAR_EQUAL<float>(out.x, 0, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(out.y, 0, 0.001);
	Tester::ASSERT_NEAR_EQUAL<float>(out.z, -1, 0.001);
}

int main() {
	std::function<void()> test1 = [] { 
		EulerToQuaternion();
	};
	std::function<void()> test2 = [] { 
		QuaternionToEuler();
	};
	std::function<void()> test3 = [] { 
		QEQ();
	};
	std::function<void()> test4 = [] { 
		EQ();
	};
	std::function<void()> test5 = [] { 
		QE();
	};
	std::function<void()> test6 = [] { 
		QuaternionMultiplication();
	};
	std::function<void()> test7 = [] { 
		Rotation();
	};
	std::function<void()> test9 = [] { 
		srand (time(NULL));
		// initalises arrays
		float* vector1 = new float[262144];
		float* vector2 = new float[262144];
		float* out = new float[262144];
		// gives random numbers to arrays
		for (int i = 0; i < 262144; i++) {
			vector1[i] = rand() / RAND_MAX;
			vector2[i] = rand() / RAND_MAX;
		}
		// calls function
		for (int i = 0; i < 100000; i++) {
			AMaths::VecAdd(vector1, vector2, 262144, out);
		}
		delete[] vector1;
		delete[] vector2;
		delete[] out;
	};
	std::function<void()> test8 = [] {
		int N = 1<<18;
		srand (time(NULL));
		// creates uninitialised pointers
		float *gpuvector1, *gpuvector2, *gpuout;
		// gives memory to the pointers
		cudaMalloc(&gpuvector1, sizeof(float) * 262144);
		cudaMalloc(&gpuvector2, sizeof(float) * 262144);
		cudaMalloc(&gpuout, sizeof(float) * 262144);
		float *cpuvector1 = new float[262144];
		float *cpuvector2 = new float[262144];
		// assigns random numbers to both
		for (int i = 0; i < N; i++) {
			cpuvector1[i] = rand() / RAND_MAX;
			cpuvector2[i] = rand() / RAND_MAX;
		}
		cudaMemcpy(gpuvector1, cpuvector1, sizeof(float) * 262144, cudaMemcpyHostToDevice);
		cudaMemcpy(gpuvector2, cpuvector2, sizeof(float) * 262144, cudaMemcpyHostToDevice);
		// calculates blocks and block sizes
		int blockSize = 256;
		int numBlocks = (N + blockSize - 1) / blockSize;
		// calls function
		for (int i = 0; i < 100000; i++) {
			GMaths::VecAdd <<< numBlocks, blockSize >>> (gpuvector1, gpuvector2, gpuout);
		}
		// frees memory
		cudaFree(gpuvector1);
		cudaFree(gpuvector2);
		cudaFree(gpuout);
		delete[] cpuvector1;
		delete[] cpuvector2;
	};
	Tester::Tester tester = Tester::Tester();
	tester.AddGroup("Angles");
	tester.AddAverageTimeTest(test1, 1000000, "Euler To Quaternion");
	tester.AddAverageTimeTest(test2, 1000000, "Quaternion To Euler");
	tester.AddAverageTimeTest(test3, 1000000, "Quaternion To Euler To Quaternion");
	tester.AddAverageTimeTest(test4, 1000000, "Euler To Quaternion");
	tester.AddAverageTimeTest(test5, 1000000, "Quaternion To Euler");
	tester.AddAverageTimeTest(test6, 1000000, "Quaternion Multiplication");
	tester.AddAverageTimeTest(test7, 1000000, "Vector3 Rotation");
	tester.AddAverageTimeTest(test7, 1000000, "Vector3 Rotation");
	tester.AddGroup("GPU Functionality");
	//tester.AddAverageTimeTest(test8, 1, "Vector Add");
	//tester.AddTest(test9, "Vector Add CPU");
	tester.RunTests();
	/*Tester::PerformanceTester testerper = Tester::PerformanceTester();
	testerper.AddGroup("GPU");
	testerper.AddAverageTest(test9, 1, "Vector Add CPU");
	testerper.AddAverageTest(test8, 1, "Vector Add GPU");
	testerper.AddAverageTest(test8, 1, "Vector Add GPU");
	testerper.RunTests();*/
}