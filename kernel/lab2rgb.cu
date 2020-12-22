#include "lab2rgb.h"

template <typename T> __device__ inline T LabInvF(const T t)
{
	const T cut = 6.0f / 29.0f;
	if (t > 16.0f) return 16.0f * 16.0f * 16.0f;
	if (t > cut) return t * t * t;
	if (t < 2.0f / 29.0f) return 0.0f;
	return 3.0f * (6.0f / 29.0f) * (6.0f / 29.0f) * (t - (4.0f / 29.0f));
}

template <typename T> __device__ inline T sRGBInvC(const T c)
{
	const T cut = 0.0031308f;
	if (c <= cut) return c * 12.92f;
	return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

template <typename T> __global__ void Lab2rgb_T_kernel(T *p_red, T *p_green, T *p_blue, T *p_L, T *p_a, T *p_b, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= width) || (y >= height)) return;
	
	int idx = x + y * pitch;
	T L = p_L[idx];
	T a = p_a[idx];
	T b = p_b[idx];

	T X, Y, Z;

	X = LabInvF((L + 16.0f) / 116.0f + a / 500.0f);
	Y = LabInvF((L + 16.0f) / 116.0f);
	Z = LabInvF((L + 16.0f) / 116.0f - b / 200.0f);

	//T X = (0.49f    * red + 0.31f    * green + 0.20f    * blue);
	//T Y = (0.17697f * red + 0.81240f * green + 0.01063f * blue);
	//T Z = (                 0.01f    * green + 0.99f    * blue);

	T red, green, blue;

	red   = __saturatef( 3.2406f * X - 1.5372f * Y - 0.4986f * Z);
	green = __saturatef(-0.9689f * X + 1.8758f * Y + 0.0415f * Z);
	blue  = __saturatef( 0.0557f * X - 0.2040f * Y + 1.0570f * Z);

	red = sRGBInvC(red); green = sRGBInvC(green); blue = sRGBInvC(blue);

	p_red  [idx] = red;
	p_green[idx] = green;
	p_blue [idx] = blue;
}

template <typename T> __global__ void Lab2rgb_T_kernel(T *p_grey, T *p_L, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= width) || (y >= height)) return;

	int idx = x + y * pitch;
	T L = p_L[idx];

	T Y;

	Y = LabInvF((L + 16.0f) / 116.0f);

	T grey;

	grey = __saturatef(Y);

	p_grey[idx] = grey;
}

template <typename T> __global__ void combine_Lab2rgb_T_kernel(T *p_red, T *p_green, T *p_blue, T *p_L, T *p_a, T *p_b, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= width) || (y >= height)) return;

	int idx = x + y * pitch;
	T L = p_L[idx];
	T a = p_a[idx];
	T b = p_b[idx];

	T X, Y, Z;

	X = LabInvF((L + 16.0f) / 116.0f + a / 500.0f);
	Y = LabInvF((L + 16.0f) / 116.0f);
	Z = LabInvF((L + 16.0f) / 116.0f - b / 200.0f);

	//T X = (0.49f    * red + 0.31f    * green + 0.20f    * blue);
	//T Y = (0.17697f * red + 0.81240f * green + 0.01063f * blue);
	//T Z = (                 0.01f    * green + 0.99f    * blue);

	T red, green, blue;

	red   = p_red  [idx];
	green = p_green[idx];
	blue  = p_blue [idx];

	red   = __saturatef(0.5f * (red   + 2.364613847f * X - 0.896540571f * Y - 0.468073276f * Z));
	green = __saturatef(0.5f * (green - 0.515166208f * X + 1.426408103f * Y + 0.088758105f * Z));
	blue  = __saturatef(0.5f * (blue  + 0.005203699f * X - 0.014408163f * Y + 1.009204464f * Z));

	p_red  [idx] = red;
	p_green[idx] = green;
	p_blue [idx] = blue;
}

template <typename T> __global__ void combine_Lab2rgb_T_kernel(T *p_grey, T *p_L, int width, int height, int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= width) || (y >= height)) return;

	int idx = x + y * pitch;
	T L = p_L[idx];

	T Y;

	Y = LabInvF((L + 16.0f) / 116.0f);

	T grey;

	grey = p_grey[idx];

	grey = __saturatef(0.5f * (grey + Y));

	p_grey[idx] = grey;
}

template <typename T>
void Lab2rgb(void * d_L, void * d_a, void * d_b, void * d_red, void * d_green, void * d_blue, unsigned int width, unsigned int height, size_t pitch)
{
	dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;
	
	dim3 grid_size;
	grid_size.x = (width + + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;

	Lab2rgb_T_kernel<<<grid_size, block_size>>>((T *) d_red, (T *) d_green, (T *) d_blue, (T *) d_L, (T *) d_a, (T *) d_b, width, height, (int)(pitch / sizeof(T)));
	CHECK_LAUNCH_ERROR();
}

template <typename T>
void Lab2rgb(void * d_L, void * d_grey, unsigned int width, unsigned int height, size_t pitch)
{
	dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;
	
	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;

	Lab2rgb_T_kernel<<<grid_size, block_size>>>((T *) d_grey, (T *) d_L, width, height, (int)(pitch / sizeof(T)));
	CHECK_LAUNCH_ERROR();
}

template <typename T>
void combineLab2rgb(void * d_L, void * d_a, void * d_b, void * d_red, void * d_green, void * d_blue, unsigned int width, unsigned int height, size_t pitch)
{
	dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;
	
	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;

	combine_Lab2rgb_T_kernel<<<grid_size, block_size>>>((T *) d_red, (T *) d_green, (T *) d_blue, (T *) d_L, (T *) d_a, (T *) d_b, width, height, (int)(pitch / sizeof(T)));
	CHECK_LAUNCH_ERROR();
}

template <typename T> 
void combineLab2rgb(void * d_L, void * d_grey, unsigned int width, unsigned int height, size_t pitch)
{
	dim3 block_size;
	block_size.x = 16;
	block_size.y = 16;
	block_size.z = 1;
	
	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;

	combine_Lab2rgb_T_kernel<<<grid_size, block_size>>>((T *) d_grey, (T *) d_L, width, height, (int)(pitch / sizeof(T)));
	CHECK_LAUNCH_ERROR();
}

template void Lab2rgb<float>(void* d_L, void* d_a, void* d_b, void* d_red, void* d_green, void* d_blue, unsigned int width, unsigned int height, size_t pitch);
template void Lab2rgb<float>(void* d_L, void* d_grey, unsigned int width, unsigned int height, size_t pitch);
template void combineLab2rgb<float>(void* d_L, void* d_a, void* d_b, void* d_red, void* d_green, void* d_blue, unsigned int width, unsigned int height, size_t pitch);
template void combineLab2rgb<float>(void* d_L, void* d_grey, unsigned int width, unsigned int height, size_t pitch);
template void Lab2rgb<double>(void* d_L, void* d_a, void* d_b, void* d_red, void* d_green, void* d_blue, unsigned int width, unsigned int height, size_t pitch);
template void Lab2rgb<double>(void* d_L, void* d_grey, unsigned int width, unsigned int height, size_t pitch);
template void combineLab2rgb<double>(void* d_L, void* d_a, void* d_b, void* d_red, void* d_green, void* d_blue, unsigned int width, unsigned int height, size_t pitch);
template void combineLab2rgb<double>(void* d_L, void* d_grey, unsigned int width, unsigned int height, size_t pitch);
