#ifndef AMATHS_H
#define AMATHS_H
#include <cmath>
#include <iostream>

namespace AMaths {
	#define PI 3.14159265359
}

namespace Literals {
	constexpr long double operator"" _mm(long double x) { return x / 1000; };
	constexpr long double  operator"" _cm(long double x) { return x / 100; };
	constexpr long double operator"" _m(long double x) { return x; };
	constexpr long double operator"" _deg(long double x) { return x * (PI / 180); };
	constexpr long double operator"" _rad(long double x) { return x; };
}

namespace AMaths {
	#define HALFPI 1.57079632679
	#define QUATERPI 0.785398163397
	#define ONEOVERPI 0.318309886184
	#define A 0.0776509570923569
	#define B -0.287434475393028
	#define C (QUATERPI - A - B)
	#define S1 0.166666666667
	#define S2 0.00833333333333
	#define S3 0.000198412698413
	struct Vector2 {
		float x, y;
	};
	struct Vector3 {
		float x, y, z;
		Vector3() {
			this->x = 0;
			this->y = 0;
			this->z = 0;
		}
		Vector3(float x, float y, float z) {
			this->x = x;
			this->y = y;
			this->z = z;
		}
	};
	struct Euler {
		float yaw, pitch, roll;
		Euler() {
			this->yaw = 0;
			this->pitch = 0;
			this->roll = 0;
		}
		Euler(float yaw, float pitch, float roll) {
			this->yaw = yaw;
			this->pitch = pitch;
			this->roll = roll;
		}
	};
	struct Quaternion {
		float w, i, j, k;
		Quaternion() {
			this->w = 0;
			this->i = 0;
			this->j = 0;
			this->k = 0;
		}
		Quaternion(float w, float i, float j, float k) {
			this->w = w;
			this->i = i;
			this->j = j;
			this->k = k;
		}
	};
	struct RotationMatrix {
		float x1, y1, z1;
		float x2, y2, z2;
		float x3, y3, z3;
	};
	void VecAdd(float* vec1, float* vec2, int vecsize, float* out);
	float V2Dot(Vector2 v1, Vector2 v2);
	float V3Dot(Vector3 v1, Vector3 v2);
	Quaternion EulerToQuaternion(Euler);
	Euler QuaternionToEuler(Quaternion);
	void Vector3ToQuaternion(Vector3 vector, Quaternion &out);
	void QuaternionToVector3(Quaternion quat, Vector3 &out);
	void QuaternionMultiplication(Quaternion Q1, Quaternion Q2, Quaternion &out);
	void InverseQuaternion(Quaternion Q, Quaternion &out);
	void QuaternionToRotationMatrix(Quaternion Q, RotationMatrix &out);
}
#endif