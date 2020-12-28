#include "filter.h"

template <typename T>
__global__ void GaussianFilterSTY_GPU_kernel(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x == 0 && y == 0) printf("GPU kernel %f\n", source[0]);
}

template <typename T>
void GaussianFilterSTY_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    
	GaussianFilterSTY_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add);
	CHECK_LAUNCH_ERROR();
}

template <typename T>
void GaussianFilterSTX_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {

}

template void GaussianFilterSTY_GPU<float>(float* target, float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add);
template void GaussianFilterSTY_GPU<double>(double* target, double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add);

template void GaussianFilterSTX_GPU<float>(float* target, float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add);
template void GaussianFilterSTX_GPU<double>(double* target, double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add);
