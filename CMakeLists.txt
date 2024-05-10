# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("Renderingengine")
#add_definitions(-D_ITERATOR_DEBUG_LEVEL=0) 
add_definitions(-w)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
# Include sub-projects.
add_subdirectory ("Renderingengine")

add_subdirectory(Tests)
