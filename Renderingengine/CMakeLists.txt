

cmake_minimum_required (VERSION 3.8)

project (Engine LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -lcudadevrt -lcublas_device -dc")

include_directories(Maths Camera GPUInterface Errors Pipeline/InputAssembler)

add_library(Renderingengine Renderingengine.cu Renderingengine.cuh)
set_property(TARGET Renderingengine PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(Maths)

add_subdirectory(Pipeline)

add_subdirectory(GPUInterface)

add_subdirectory(Errors)

target_link_libraries(Renderingengine
	PRIVATE
	AMaths
	GMaths
	InputAssembler
	VertexShader
	Calls
	Errors)

add_executable(Main MAIN.cu)
set_property(TARGET Main PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Main
	PRIVATE
	Renderingengine)
