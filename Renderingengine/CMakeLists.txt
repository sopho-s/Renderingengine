

cmake_minimum_required (VERSION 3.8)

project (Engine LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(Maths Camera)

add_executable (Renderingengine Renderingengine.cpp Renderingengine.h)

add_subdirectory(Includes)

add_subdirectory(Maths)

target_link_libraries(Renderingengine
	PRIVATE
	AMaths
	GMaths)
