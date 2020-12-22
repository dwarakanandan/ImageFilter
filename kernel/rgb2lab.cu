#include "rgb2lab.h"

template <typename T> __device__ inline T LabF(const T t)
{
	const T cut = (6.0f / 29.0f) * (6.0f / 29.0f) * (6.0f / 29.0f);
	if (t > 16.0f * 16.0f * 16.0f) return 16.0f;
	if (t > cut) return expf(logf(t) / 3.0f);
	if (t < 0.0f) return 4.0f / 29.0f;
	return (29.0f / 6.0f) * (29.0f / 6.0f) * (t / 3.0f) + (4.0f / 29.0f);
}

template <typename T> __device__ inline T sRGBC(const T c)
{
	const T cut = 0.04045f;
	if (c <= cut) return c / 12.92f;
	return powf((c + 0.055f) / 1.055f, 2.4f);
}

template <typename T> __global__ void rgb2Lab_T_kernel(T *p_red, T *p_green, T *p_blue, T *p_L, T *p_a, T *p_b, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= width) || (y >= height)) return;

	int idx = x + y * pitch;

	T red, green, blue;
	red   = p_red  [idx];
	green = p_green[idx];
	blue  = p_blue [idx];

	T X, Y, Z;

	// sRGB
	red = sRGBC(red); green = sRGBC(green); blue = sRGBC(blue);
	X = (0.4124f * red + 0.3576f * green + 0.1805f * blue);
	Y = (0.2126f * red + 0.7152f * green + 0.0722f * blue);
	Z = (0.0193f * red + 0.1192f * green + 0.9505f * blue);

	T fX, fY, fZ;

	fX = LabF(X);
	fY = LabF(Y);
	fZ = LabF(Z);

	T L, a, b;

	L = 116.0f * fY - 16.0f;
	a = 500.0f * (fX - fY);
	b = 200.0f * (fY - fZ);
	p_L[idx] = L;
	p_a[idx] = a;
	p_b[idx] = b;
}

template <typename T> __global__ void rgb2Lab_T_kernel(T *p_grey, T *p_L, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= width) || (y >= height)) return;

	int idx = x + y * pitch;

	T Y;
	Y = p_grey[idx];

	// sRGB
	Y = sRGBC(Y);

	T fY;

	fY = LabF(Y);

	T L = 116.0f * fY - 16.0f;

	p_L[idx] = L;
}

template <typename T>
void rgb2Lab(void * d_red, void * d_green, void * d_blue, void * d_L, void * d_a, void * d_b, unsigned int width, unsigned int height, size_t pitch)
{
	dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;
	
	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;

	rgb2Lab_T_kernel<<<grid_size, block_size>>>((T *) d_red, (T *) d_green, (T *) d_blue, (T *) d_L, (T *) d_a, (T *) d_b, width, height, (int)(pitch / sizeof(T)));
	CHECK_LAUNCH_ERROR();
}

template <typename T>
void rgb2Lab(void * d_grey, void * d_L, unsigned int width, unsigned int height, size_t pitch)
{
	dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;
	
	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;

	rgb2Lab_T_kernel<<<grid_size, block_size>>>((T *) d_grey, (T *) d_L, width, height, (int)(pitch / sizeof(T)));
	CHECK_LAUNCH_ERROR();
}

template void rgb2Lab<float>(void* d_red, void* d_green, void* d_blue, void* d_L, void* d_a, void* d_b, unsigned int width, unsigned int height, size_t pitch);
template void rgb2Lab<float>(void* d_grey, void* d_L, unsigned int width, unsigned int height, size_t pitch);
template void rgb2Lab<double>(void* d_red, void* d_green, void* d_blue, void* d_L, void* d_a, void* d_b, unsigned int width, unsigned int height, size_t pitch);
template void rgb2Lab<double>(void* d_grey, void* d_L, unsigned int width, unsigned int height, size_t pitch);
