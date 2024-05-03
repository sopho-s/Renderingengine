#ifndef CAMERA_H
#define CAMERA_H
#include "AMaths.cuh"
namespace Camera {
	class Camera {
	private:
		float focallength;
		float width;
		float height;
		AMaths::Vector3 forwards;
		AMaths::Vector3 right;
	public:
		Camera(long double focallength, long double width, long double height);
	};
}
#endif