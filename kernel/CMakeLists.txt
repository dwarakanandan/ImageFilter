# Put all your source files here 
SET(kernel_SRCS
  cuda_memory.cu
  filter.cu
  lab2rgb.cu
  #lap_solver.cu
  math.cu
  rgb2lab.cu
  sample.cu
)

# this command creates your executable from the files specified in MyProject_SRCS
CUDA_ADD_LIBRARY(kernel ${kernel_SRCS})

target_compile_features(kernel PUBLIC cxx_std_14)
target_compile_options(kernel PRIVATE --expt-extended-lambda)
add_definitions(-D_OPENMP)

# link all necessary libraries here (only the names are needed, the rest is done by CMake)
#TARGET_LINK_LIBRARIES(kernel)

