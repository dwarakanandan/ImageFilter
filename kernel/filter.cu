
#include "filter.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define BLOCK_SIZE 32
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

//------------------------------------------------------------------------------------------------------------//

template <typename T>
struct CudaAtomicAdd {
    __device__ T AtomicAdd(T* ref, T value)
	{
		extern __device__ void error(void);
		error(); // Ensure that we won't compile any un-specialized types
		return NULL;
	}
};

template <>
struct CudaAtomicAdd <int>
{
    __device__ unsigned int AtomicAdd(int* ref, int value)
	{
		return atomicAdd(ref,value);
	}
};

template <>
struct CudaAtomicAdd <float>
{
    __device__ float AtomicAdd(float* ref, float value)
	{
		return atomicAdd(ref,value);
	}
};

template <>
struct CudaAtomicAdd <double>
{
    __device__ double AtomicAdd(double* ref, double value)
	{
		#if __CUDA_ARCH__ < 600
		unsigned long long int* address_as_ull = (unsigned long long int*)ref;
		unsigned long long int old = *address_as_ull, assumed;
		do {
			assumed = old;
			old = atomicCAS(address_as_ull, 
    			    assumed,
    			    __double_as_longlong(
    			        value + __longlong_as_double(assumed)
    			    )
    			);
		} while (assumed != old);
		return __longlong_as_double(old);
	#else
		return atomicAdd(ref, value);
	#endif
	}
};


template <typename T>
__global__ void GaussianSplatSTX_GPU_kernel(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    int bound = (int)floor(GAUSSIAN_RANGE * scale);
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    CudaAtomicAdd<T> caa;
	// if (!add) std::fill(target, target + width * height, T(0));

	if (boundary == BoundaryCondition::Renormalize)
	{
		// for (int x = 0; x < width; x++)
		// {
			T weight = gaussian(d, scale);
			for (int x0 = 1; x0 <= bound; x0++)
			{
				T ga = gaussian(-x0 + d, scale);
				T gb = gaussian(x0 + d, scale);
				int xa = x - x0;
				int xb = x + x0;
				if (xa >= 0) weight += ga;
				if (xb < width) weight += gb;
			}
			T r_weight = 1.0f / weight;
			for (int y = 0; y < height; y++)
			{
				T t = source[x + y * width] * r_weight;
				for (int x0 = 1; x0 <= bound; x0++)
				{
					T ga = gaussian(-x0 + d, scale);
					T gb = gaussian(x0 + d, scale);
					int xa = x - x0;
					int xb = x + x0;
					if (xa >= 0) target[xa + y * width] += t * ga;
					if (xb < width) target[xb + y * width] += t * gb;
				}
				T g = gaussian(d, scale);
				target[x + y * width] += t * g;
			}
		// }
	}
	else // if (boundary == BoundaryCondition::Border)
	{
		T weight = gaussian(d, scale);
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian(-x0 + d, scale);
			T gb = gaussian(x0 + d, scale);
			weight += ga + gb;
		}
		T r_weight = 1.0f / weight;
		// for (int y = 0; y < height; y++)
		// {
		// 	for (int x = 0; x < width; x++)
		// 	{
				
		// 	}
        // }
        // printf("X,WeightDone,%d,%d",x,y);
        // fflush(stdout);
        T t = source[x + y * width] * r_weight;
        for (int x0 = 1; x0 <= bound; x0++)
        {
            T ga = gaussian(-x0 + d, scale);
            T gb = gaussian(x0 + d, scale);
            int xa = max(0, x - x0);
            int xb = min(x + x0, width - 1);
            caa.AtomicAdd(&target[xa + y * width],t*ga);
            caa.AtomicAdd(&target[xb + y * width],t*gb);
            // target[xa + y * width] += t * ga;
            // target[xb + y * width] += t * gb;
        }
        T g = gaussian(d, scale);
        caa.AtomicAdd(&target[x + y * width],t*g);
        // target[x + y * width] += t * g;
	}
}


template <typename T>
void GaussianSplatSTX_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;
    //std::cout<<"Start Splatx";
	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    
    GaussianSplatSTX_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add);
    CHECK_LAUNCH_ERROR();
}



template <typename T>
__global__ void GaussianSplatSTY_GPU_kernel(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    int bound = (int)floor(GAUSSIAN_RANGE * scale);
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	// if (!add) std::fill(target, target + width * height, T(0));
    CudaAtomicAdd<T> caa;
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
    //printf("Y,WeightDone,%d,%d",x,y);
    // fflush(stdout);
	T t = source[x + y * width] * r_weight;
    for (int y0 = 1; y0 <= bound; y0++)
    {
        T ga = gaussian(-y0 + d, scale);
        T gb = gaussian(y0 + d, scale);
        int ya = max(0, y - y0);
        int yb = min(y + y0, height - 1);
        caa.AtomicAdd(&target[x + ya * width],t*ga);
        caa.AtomicAdd(&target[x + yb * width],t*gb);
        // target[x + ya * width] += t * ga;
        // target[x + yb * width] += t * gb;
    }
    T g = gaussian(d, scale);
    caa.AtomicAdd(&target[x + y * width],t*g);
    // target[x + y * width] += t * g;


	// for (int y = 0; y < height; y++)
	// {
	// 	if (boundary == BoundaryCondition::Renormalize)
	// 	{
	// 		weight = gaussian(d, scale);

	// 		for (int y0 = 1; y0 <= bound; y0++)
	// 		{
	// 			T ga = gaussian(-y0 + d, scale);
	// 			T gb = gaussian(y0 + d, scale);
	// 			int ya = y - y0;
	// 			int yb = y + y0;
	// 			if (ya >= 0) weight += ga;
	// 			if (yb < height) weight += gb;
	// 		}
	// 		r_weight = 1.0f / weight;
	// 	}
	// 	for (int x = 0; x < width; x++)
	// 	{

	// 		if (boundary == BoundaryCondition::Renormalize)
	// 		{
	// 			for (int y0 = 1; y0 <= bound; y0++)
	// 			{
	// 				T ga = gaussian(-y0 + d, scale);
	// 				T gb = gaussian(y0 + d, scale);
	// 				int ya = y - y0;
	// 				int yb = y + y0;
	// 				if (ya >= 0) target[x + ya * width] += t * ga;
	// 				if (yb < height) target[x + yb * width] += t * gb;
	// 			}
	// 		}
	// 		else // if (boundary == BoundaryCondition::Border)
	// 		{
				
	// 		}

	// 	}
    // }
    
    
}



template <typename T>
void GaussianSplatSTY_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    
	GaussianSplatSTY_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add);
	CHECK_LAUNCH_ERROR();
}

template void GaussianSplatSTY_GPU<float>(float* target, float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add);
template void GaussianSplatSTY_GPU<double>(double* target, double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add);

template void GaussianSplatSTX_GPU<float>(float* target, float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add);
template void GaussianSplatSTX_GPU<double>(double* target, double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add);
