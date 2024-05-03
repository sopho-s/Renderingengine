#include "Camera.h"

using namespace Literals;

namespace Camera {
	Camera::Camera(long double focallength = 18.0_mm, long double width = 20.0_cm, long double height = 15.0_cm) {
		this->focallength = (float)focallength;
		this->height = (float)height;
		this->width = (float)width;
		this->forwards = AMaths::Vector3(1, 0, 0);
		this->right = AMaths::Vector3(0, 1, 0);
	}
}