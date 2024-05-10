

cmake_minimum_required (VERSION 3.8)

project (Engine LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(Maths Camera GPUInterface Errors Pipeline/InputAssembler)

add_executable(Renderingengine Renderingengine.cu Renderingengine.cuh)

set_target_properties(Renderingengine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


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
	Errors
    IdleWork)