# Put all your source files here 
SET(image_SRCS
	cuda_image.cpp
	float_image.cpp
	image.cpp
	jpeg_image.cpp
	lsmrBase.cpp
	lsqrBase.cpp
	png_image.cpp
	#random_functions.cpp
)

CUDA_ADD_LIBRARY(image STATIC ${image_SRCS})

#target_compile_features(image PUBLIC CUDA)
target_compile_features(image PUBLIC cxx_std_14)
#target_include_directories(image PUBLIC ${CUDA_INCLUDE_DIRS})
#add_definitions(-D_OPENMP)

TARGET_LINK_LIBRARIES(image z jpeg png kernel)

