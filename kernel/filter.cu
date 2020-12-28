#include "filter.h"

#define GAUSSIAN_RANGE 3
__device__ double gaussian(double x0, double size)
{
	double d = x0 / size;
	return exp(-0.5 * d * d);
}

__device__ int _min(int a, int b)
{
	return a>b? b: a;
}

__device__ int _max(int a, int b)
{
	return a>b? a: b;
}

template <typename T>
__global__ void GaussianFilterSTY_GPU_kernel(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    int bound = (int)floor(GAUSSIAN_RANGE * scale);
    T weight = T(0);
	T r_weight = T(1);
	if (boundary != BoundaryCondition::Renormalize)
	{
		weight = gaussian(d, scale);

		for (int y0 = 1; y0 <= bound; y0++)
		{
			T ga = gaussian(-y0 + d, scale);
			T gb = gaussian(y0 + d, scale);
			weight += ga + gb;
		}
		r_weight = 1.0f / weight;
    }

    T g = gaussian(d, scale);
    T t = source[x + y * width] * g;

    for (int y0 = 1; y0 <= bound; y0++)
    {
        T ga = gaussian(-y0 + d, scale);
        T gb = gaussian(y0 + d, scale);
        int ya = _max(0, y - y0);
        int yb = _min(y + y0, height - 1);
        t += source[x + ya * width] * ga + source[x + yb * width] * gb;
    }

    if (add) target[x + y * width] += t * r_weight;
	else target[x + y * width] = t * r_weight;
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
__global__ void GaussianFilterSTX_GPU_kernel(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    int bound = (int)floor(GAUSSIAN_RANGE * scale);

    T weight = gaussian(d, scale);
    for (int x0 = 1; x0 <= bound; x0++)
    {
        T ga = gaussian(-x0 + d, scale);
        T gb = gaussian(x0 + d, scale);
        weight += ga + gb;
    }
    T r_weight = 1.0f / weight;

    T g = gaussian(d, scale);
    T t = source[x + y * width] * g;

    for (int x0 = 1; x0 <= bound; x0++)
    {
        T ga = gaussian(-x0 + d, scale);
        T gb = gaussian(x0 + d, scale);
        int xa = _max(0, x - x0);
        int xb = _min(x + x0, width - 1);
        t += source[xa + y * width] * ga + source[xb + y * width] * gb;
    }

    if (add) target[x + y * width] += t * r_weight;
	else target[x + y * width] = t * r_weight;
}

template <typename T>
void GaussianFilterSTX_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    
	GaussianFilterSTX_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add);
	CHECK_LAUNCH_ERROR();
}

template void GaussianFilterSTY_GPU<float>(float* target, float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add);
template void GaussianFilterSTY_GPU<double>(double* target, double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add);

template void GaussianFilterSTX_GPU<float>(float* target, float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add);
template void GaussianFilterSTX_GPU<double>(double* target, double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add);
