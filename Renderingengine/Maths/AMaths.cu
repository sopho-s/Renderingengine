#include "AMaths.cuh"

namespace AMaths {
	void VecAdd(float* vec1, float* vec2, int vecsize, float* out) {
		for (int i = 0; i < vecsize; i++) {
			out[i] = vec1[i] + vec2[i];
		}
	}
	float V2Dot(Vector2 v1, Vector2 v2) {
		return v1.x * v2.x + v1.y * v2.y;
	}
	float V3Dot(Vector3 v1, Vector3 v2) {
		return v1.x * v2.x + v1.y * v2.y + v1.z *v2.z;
	}
	Quaternion EulerToQuaternion(Euler angles) {
		Quaternion result(std::cos(angles.roll / 2) * std::cos(angles.pitch / 2) * std::cos(angles.yaw / 2) + std::sin(angles.roll / 2) * std::sin(angles.pitch / 2) * std::sin(angles.yaw / 2),
		std::sin(angles.roll / 2) * std::cos(angles.pitch / 2) * std::cos(angles.yaw / 2) - std::cos(angles.roll / 2) * std::sin(angles.pitch / 2) * std::sin(angles.yaw / 2),
		std::cos(angles.roll / 2) * std::sin(angles.pitch / 2) * std::cos(angles.yaw / 2) + std::sin(angles.roll / 2) * std::cos(angles.pitch / 2) * std::sin(angles.yaw / 2),
		std::cos(angles.roll / 2) * std::cos(angles.pitch / 2) * std::sin(angles.yaw / 2) - std::sin(angles.roll / 2) * std::sin(angles.pitch / 2) * std::cos(angles.yaw / 2));
		return result;
	}
	Euler QuaternionToEuler(Quaternion angles) {
		float sinr_cosangles = 2 * (angles.w * angles.i + angles.j * angles.k);
		float cosr_cosangles = 1 - 2 * (angles.i * angles.i + angles.j * angles.j);

		float sinangles = std::sqrt(1 + 2 * (angles.w * angles.j - angles.i * angles.k));
		float cosangles = std::sqrt(1 - 2 * (angles.w * angles.j - angles.i * angles.k));

		float siny_cosangles = 2 * (angles.w * angles.k + angles.i * angles.j);
		float cosy_cosangles = 1 - 2 * (angles.j * angles.j + angles.k * angles.k);
		
		Euler result(std::atan2(siny_cosangles, cosy_cosangles),
			2 * std::atan2(sinangles, cosangles) - PI / 2,
			std::atan2(sinr_cosangles, cosr_cosangles));
		return result;
	}
	void Vector3ToQuaternion(Vector3 vector, Quaternion &out) {
		out.w = 0;
		out.i = vector.x;
		out.j = vector.y;
		out.k = vector.z;
	}
	void QuaternionToVector3(Quaternion quat, Vector3 &out) {
		out.x = quat.i;
		out.y = quat.j;
		out.z = quat.k;
	}
	void QuaternionMultiplication(Quaternion Q1, Quaternion Q2, Quaternion &out) {
		out.w = Q1.w * Q2.w - Q1.i * Q2.i - Q1.j * Q2.j - Q1.k  * Q2.k;
		out.i = Q1.w * Q2.i + Q1.i * Q2.w + Q1.j * Q2.k - Q1.k  * Q2.j;
		out.j = Q1.w * Q2.j - Q1.i * Q2.k + Q1.j * Q2.w + Q1.k  * Q2.i;
		out.k = Q1.w * Q2.k + Q1.i * Q2.j - Q1.j * Q2.i + Q1.k  * Q2.w;
	}
	void InverseQuaternion(Quaternion Q, Quaternion &out) {
		out.w = Q.w;
		out.i = -Q.i;
		out.j = -Q.w;
		out.k = -Q.k;
	}
	void QuaternionToRotationMatrix(Quaternion Q, RotationMatrix &out) {
		float v01 = Q.w * Q.i;
		float v02 = Q.w * Q.j;
		float v03 = Q.w * Q.k;
		float v12 = Q.i * Q.j;
		float v13 = Q.i * Q.k;
		float v23 = Q.j * Q.k;
		float q02 = Q.w * Q.w;
		out.x1 = 2 * (q02 + Q.i * Q.i) - 1;
		out.y2 = 2 * (q02 + Q.j * Q.j) - 1;
		out.z3 = 2 * (q02 + Q.k * Q.k) - 1;
		out.y1 = 2 * (v12 - v03);
		out.z1 = 2 * (v13 + v02);
		out.x2 = 2 * (v12 + v03);
		out.z2 = 2 * (v23 - v01);
		out.x3 = 2 * (v13 - v02);
		out.y3 = 2 * (v23 + v01);
	}
}