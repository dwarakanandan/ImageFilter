# Put all your source files here 
SET(filter_SRCS
	main.cpp
)


# this command creates your executable from the files specified in MyProject_SRCS
CUDA_ADD_EXECUTABLE(filter.exe ${filter_SRCS})

target_compile_features(filter.exe PUBLIC cxx_std_14)

# link all necessary libraries here (only the names are needed, the rest is done by CMake)
TARGET_LINK_LIBRARIES(filter.exe image z jpeg png kernel)

