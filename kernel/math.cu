#include "math_kernel.h"

template <typename T> __global__ void add_kernel(T *p_target, T f, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= width) || (y >= height)) return;

	int idx = x + y * pitch;
	T a;
	a = p_target[idx];
	T c = a + f;
	p_target[idx] = c;
}

template <typename T>
void add(void * d_target, T f, unsigned int width, unsigned int height, size_t pitch)
{
	if (f == 0.0f) return;
	dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;

	add_kernel << <grid_size, block_size >> >((T *)d_target, f, width, height, (int)(pitch / sizeof(T)));
	CHECK_LAUNCH_ERROR();
}

template <typename T> __global__ void multiply_kernel(T *p_target, T f, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= width) || (y >= height)) return;

	int idx = x + y * pitch;
	T a;
	a = p_target[idx];
	T c = a * f;
	p_target[idx] = c;
}

template <typename T>
void multiply(void * d_target, T f, unsigned int width, unsigned int height, size_t pitch)
{
	if (f == 0.0f)
	{
		cudaMemset2D(d_target, pitch, 0, width * sizeof(T), height);
		return;
	}
	dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;

	multiply_kernel << <grid_size, block_size >> >((T *)d_target, f, width, height, (int)(pitch / sizeof(T)));
	CHECK_LAUNCH_ERROR();
}

template <typename T> __global__ void add_kernel(T *p_target, T *p_a, T *p_b, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= width) || (y >= height)) return;

	int idx = x + y * pitch;
	T a = p_a[idx];
	T b = p_b[idx];
	T c = a + b;
	p_target[idx] = c;
}

template <typename T>
void add(void * d_target, const void * d_a, const void * d_b, unsigned int width, unsigned int height, size_t pitch)
{
	dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;

	add_kernel << <grid_size, block_size >> >((T *)d_target, (T *)d_a, (T *)d_b, width, height, (int)(pitch / sizeof(T)));
	CHECK_LAUNCH_ERROR();
}

template <typename T> __global__ void subtract_kernel(T *p_target, T *p_a, T *p_b, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= width) || (y >= height)) return;

	int idx = x + y * pitch;
	T a = p_a[idx];
	T b = p_b[idx];
	T c = a - b;
	p_target[idx] = c;
}

template <typename T>
void subtract(void * d_target, const void * d_a, const void * d_b, unsigned int width, unsigned int height, size_t pitch)
{
	dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;

	subtract_kernel<<<grid_size, block_size>>>((T *)d_target, (T *)d_a, (T *)d_b, width, height, (int)(pitch / sizeof(T)));
	CHECK_LAUNCH_ERROR();
}

template void add<float>(void* d_target, float f, unsigned int width, unsigned int height, size_t pitch);
template void multiply<float>(void* d_target, float f, unsigned int width, unsigned int height, size_t pitch);
template void add<float>(void* d_target, const void* d_a, const void* d_b, unsigned int width, unsigned int height, size_t pitch);
template void subtract<float>(void* d_target, const void* d_a, const void* d_b, unsigned int width, unsigned int height, size_t pitch);
template void add<double>(void* d_target, double f, unsigned int width, unsigned int height, size_t pitch);
template void multiply<double>(void* d_target, double f, unsigned int width, unsigned int height, size_t pitch);
template void add<double>(void* d_target, const void* d_a, const void* d_b, unsigned int width, unsigned int height, size_t pitch);
template void subtract<double>(void* d_target, const void* d_a, const void* d_b, unsigned int width, unsigned int height, size_t pitch);
