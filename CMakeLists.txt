CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

#
# 
# Name the project you can choose any name you want here
PROJECT(MyProject LANGUAGES C CXX CUDA) 


# Check requirements
#FIND_PACKAGE(CUDA REQUIRED)
SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_C_COMPILER gcc)

# Release Mode
SET(CUDA_PROPAGATE_HOST_FLAGS "OFF")
SET(CMAKE_CXX_FLAGS "-Wall -O3 -Wfatal-errors -fstrict-aliasing -m64 -std=c++11 -fopenmp")
SET(CUDA_NVCC_FLAGS "-arch=sm_75 --ptxas-options=-v -Xcompiler -Wall -O3 -std=c++11 -Xcompiler -fopenmp --expt-extended-lambda --default-stream per-thread")

# Add needed packages (e.g. libraries)
#FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
find_package(OpenMP)

# Define where your executables should be put
set(EXECUTABLE_OUTPUT_PATH ${MyProject_BINARY_DIR})


# Define where your build libraries should be put
set(LIBRARY_OUTPUT_PATH ${MyProject_BINARY_DIR}/lib)

# Include directories and Link directories
# you should add all your folders here, as this makes copying and reusing your code easier
# pay attention to QT folders, as you might need to add the binary folder as well
INCLUDE_DIRECTORIES(
			${CUDA_INCLUDE_DIRS}
		    ${MyProject_SOURCE_DIR}/libjpeg
			${MyProject_SOURCE_DIR}/libpng
		    ${MyProject_SOURCE_DIR}/zlib
                        ${MyProject_SOURCE_DIR}/image
)

# Tell CMake to process the sub-directories
ADD_SUBDIRECTORY(zlib)
ADD_SUBDIRECTORY(libjpeg)
ADD_SUBDIRECTORY(libpng)
ADD_SUBDIRECTORY(kernel)
ADD_SUBDIRECTORY(image)
ADD_SUBDIRECTORY(filter)


