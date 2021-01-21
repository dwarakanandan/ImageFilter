#include "filter.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BLOCK_SIZE 32
#define GAUSSIAN_RANGE 3
#define COMPUTE_TYPE double

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* ref, double value) 
{ 
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
}
#endif

__constant__ COMPUTE_TYPE r_weight_const_X;
__constant__ COMPUTE_TYPE r_weight_const_Y;
__constant__ COMPUTE_TYPE g_const_x;
__constant__ COMPUTE_TYPE g_const_y;

double gaussian_host_cu(double x0, double size)
{
	double d = x0 / size;
	return exp(-0.5 * d * d);
}

template <typename T>
void setConstantMemWeight_X_GPU(T d,T scale){
	T weight = gaussian_host_cu(d, scale);
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian_host_cu(-x0 + d, scale);
			T gb = gaussian_host_cu(x0 + d, scale);
			weight += ga + gb;
		}
	T r_weight = 1.0f / weight;
	cudaMemcpyToSymbol(r_weight_const_X,&r_weight,sizeof(T),0,cudaMemcpyHostToDevice);
	
	T g = gaussian_host_cu(d, scale);
	cudaMemcpyToSymbol(g_const_x,&g,sizeof(T),0,cudaMemcpyHostToDevice);
}
template void setConstantMemWeight_X_GPU<float>(float d,float scale);
template void setConstantMemWeight_X_GPU<double>(double d,double scale);

template <typename T>
void setConstantMemWeight_Y_GPU(T d,T scale){
	T weight = gaussian_host_cu(d, scale);
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian_host_cu(-x0 + d, scale);
			T gb = gaussian_host_cu(x0 + d, scale);
			weight += ga + gb;
		}
	
	T r_weight = 1.0f / weight;
	cudaMemcpyToSymbol(r_weight_const_Y,&r_weight,sizeof(T),0,cudaMemcpyHostToDevice);

	T g = gaussian_host_cu(d, scale);
	cudaMemcpyToSymbol(g_const_y,&g,sizeof(T),0,cudaMemcpyHostToDevice);
	
}
template void setConstantMemWeight_Y_GPU<float>(float d,float scale);
template void setConstantMemWeight_Y_GPU<double>(double d,double scale);

template <typename T> void CudaImage<T>::SetConstants(T dx,T dy,T scale){
	setConstantMemWeight_X_GPU(dx,scale);
	setConstantMemWeight_Y_GPU(dy,scale);
}

template class CudaImage<float>;
template class CudaImage<double>;

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

//********************************************************************************************************************//

template <typename T>
__global__ void GaussianFilterSTY_GPU_kernel(T* __restrict__ target, T* __restrict__ source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* __restrict__ gaussian_array) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int guard = ((bound / height) + 1) * height;

    T weight = T(0);
	T r_weight = T(1);

	if (boundary == BoundaryCondition::Renormalize)
	{
		weight = gaussian(d, scale);

		for (int y0 = 1; y0 <= bound; y0++)
		{
			T ga = gaussian(-y0 + d, scale);
			T gb = gaussian(y0 + d, scale);
			int ya = y - y0;
			int yb = y + y0;
			if (ya >= 0) weight += ga;
			if (yb < height) weight += gb;
		}
		r_weight = 1.0f / weight;
	}

	if (boundary != BoundaryCondition::Renormalize)
	{
		r_weight = r_weight_const_Y;
    }

	T t = source[x + y * width] * g_const_y;

	if ((boundary == BoundaryCondition::Zero) || (boundary == BoundaryCondition::Renormalize))
	{
		for (int y0 = 1; y0 <= bound; y0++)
		{
			T ga = gaussian_array[y0 - 1];
			T gb = gaussian_array[y0 - 1 + bound];
			int ya = y - y0;
			int yb = y + y0;
			if (ya >= 0) t += source[x + ya * width] * ga;
			if (yb < height) t += source[x + yb * width] * gb;
		}
	}
	else if (boundary == BoundaryCondition::Repeat)
	{
		for (int y0 = 1; y0 <= bound; y0++)
		{
			T ga = gaussian_array[y0 - 1];
			T gb = gaussian_array[y0 - 1 + bound];
			int ya = (y - y0 + guard) % height;
			int yb = (y + y0) % height;
			t += source[x + ya * width] * ga + source[x + yb * width] * gb;
		}
	}
	else // if (boundary == BoundaryCondition::Border)
	{
		for (int y0 = 1; y0 <= bound; y0++)
		{
			T ga = gaussian_array[y0 - 1];
			T gb = gaussian_array[y0 - 1 + bound];
			int ya = _max(0, y - y0);
			int yb = _min(y + y0, height - 1);
			t += source[x + ya * width] * ga + source[x + yb * width] * gb;
		}
	}

    if (add) target[x + y * width] += t * r_weight;
    else target[x + y * width] = t * r_weight;
}

template <typename T>
void GaussianFilterSTY_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* gaussian_array_y) {
    dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    
	GaussianFilterSTY_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add, (T *)gaussian_array_y);
	CHECK_LAUNCH_ERROR();
}

template <typename T>
__global__ void GaussianFilterSTX_GPU_kernel(T* __restrict__ target, T* __restrict__ source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* __restrict__ gaussian_array) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int guard = ((bound / width) + 1) * width;

	T weight = T(0);
	T r_weight = T(1);

	if (boundary == BoundaryCondition::Renormalize)
	{
		weight = gaussian(d, scale);
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian(-x0 + d, scale);
			T gb = gaussian(x0 + d, scale);
			int xa = x - x0;
			int xb = x + x0;
			if (xa >= 0) weight += ga;
			if (xb < width) weight += gb;
		}
		r_weight = 1.0f / weight;
	}

	if (boundary != BoundaryCondition::Renormalize)
	{
		r_weight = r_weight_const_X;
    }

	T t = source[x + y * width] * g_const_x;
	
	if ((boundary == BoundaryCondition::Zero) || (boundary == BoundaryCondition::Renormalize))
	{
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian_array[x0 - 1];
			T gb = gaussian_array[x0 + bound - 1];
			int xa = x - x0;
			int xb = x + x0;
			if (xa >= 0) t += source[xa + y * width] * ga;
			if (xb < width) t += source[xb + y * width] * gb;
		}
	}
	else if (boundary == BoundaryCondition::Repeat)
	{
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian_array[x0 - 1];
			T gb = gaussian_array[x0 + bound - 1];
			int xa = (x - x0 + guard) % width;
			int xb = (x + x0) % width;
			t += source[xa + y * width] * ga + source[xb + y * width] * gb;
		}
	}
	else // if (boundary == BoundaryCondition::Border)
	{
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian_array[x0 - 1];
			T gb = gaussian_array[x0 + bound - 1];
			int xa = _max(0, x - x0);
			int xb = _min(x + x0, width - 1);
			t += source[xa + y * width] * ga + source[xb + y * width] * gb;
		}
	}

    if (add) target[x + y * width] += t * r_weight;
    else target[x + y * width] = t * r_weight;
}

template <typename T>
void GaussianFilterSTX_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* gaussian_array_x) {
    dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    
    GaussianFilterSTX_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add, (T* )gaussian_array_x);
    CHECK_LAUNCH_ERROR();
}

template void GaussianFilterSTY_GPU<float>(float* target, float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add, float* gaussian_array_y);
template void GaussianFilterSTY_GPU<double>(double* target, double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add, double* gaussian_array_y);

template void GaussianFilterSTX_GPU<float>(float* target, float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add, float* gaussian_array_x);
template void GaussianFilterSTX_GPU<double>(double* target, double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add, double* gaussian_array_x);

//********************************************************************************************************************//

template <typename T>
__global__ void GaussianSplatSTX_GPU_kernel(T* __restrict__ target,const T* __restrict__ source,int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* __restrict__ gaussian_array) {
    int bound = (int)floor(GAUSSIAN_RANGE * scale);
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	// if (!add) std::fill(target, target + width * height, T(0));

	if (boundary == BoundaryCondition::Renormalize)
	{
			T weight = g_const_x;
			for (int x0 = 1; x0 <= bound; x0++)
			{
				// T ga = gaussian(-x0 + d, scale);
				T ga = gaussian_array[x0-1];
				// T gb = gaussian(x0 + d, scale);
				T gb = gaussian_array[x0+bound-1];
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
					// T ga = gaussian(-x0 + d, scale);
					T ga = gaussian_array[x0-1];
					// T gb = gaussian(x0 + d, scale);
					T gb = gaussian_array[x0+bound-1];
					int xa = x - x0;
					int xb = x + x0;
					if (xa >= 0) atomicAdd(&target[xa + y * width],t * ga);
					if (xb < width) atomicAdd(&target[xb + y * width],t * gb);
				}
				// T g = gaussian(d, scale);
				atomicAdd(&target[x + y * width],t * g_const_x);
			}
		// }
	}
	else // if (boundary == BoundaryCondition::Border)
	{
		T t = source[x + y * width] * r_weight_const_X;
        for (int x0 = 1; x0 <= bound; x0++)
        {
			// T ga = gaussian(-x0 + d, scale);
			T ga = gaussian_array[x0-1];
			// T gb = gaussian(x0 + d, scale);
			T gb = gaussian_array[x0+bound-1];
            int xa = max(0, x - x0);
            int xb = min(x + x0, width - 1);
            atomicAdd(&target[xa + y * width],t*ga);
            atomicAdd(&target[xb + y * width],t*gb);
        }
        atomicAdd(&target[x + y * width],t*g_const_x);
        
	}
}

template <typename T>
void GaussianSplatSTX_GPU(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add,T* gaussian_array_x) {
    dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;
	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
	GaussianSplatSTX_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height,(T)scale, (T)d, boundary, add,(T*) gaussian_array_x);
    CHECK_LAUNCH_ERROR();
}

template <typename T>
__global__ void GaussianSplatSTY_GPU_kernel(T* __restrict__ target,const T* __restrict__ source,int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* __restrict__ gaussian_array) {
    int bound = (int)floor(GAUSSIAN_RANGE * scale);
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (boundary != BoundaryCondition::Renormalize){
		T t = source[x + y * width] * r_weight_const_Y;
		for (int y0 = 1; y0 <= bound; y0++)
		{
			// T ga = gaussian(-y0 + d, scale);
			T ga = gaussian_array[y0 - 1];
			// T gb = gaussian(y0 + d, scale);
			T gb = gaussian_array[y0 - 1 + bound];
			int ya = max(0, y - y0);
			int yb = min(y + y0, height - 1);
			atomicAdd(&target[x + ya * width],t*ga);
			atomicAdd(&target[x + yb * width],t*gb);
		}
		// T g = gaussian(d, scale);
		atomicAdd(&target[x + y * width],t*g_const_y);
	}else{
		
		T weight = g_const_y;
		for (int y0 = 1; y0 <= bound; y0++)
			{
				T ga = gaussian(-y0 + d, scale);
				T gb = gaussian(y0 + d, scale);
				int ya = y - y0;
				int yb = y + y0;
				if (ya >= 0) weight += ga;
				if (yb < height) weight += gb;
			}
		T r_weight = 1.0f / weight;
		T t = source[x + y * width] * r_weight;
		for (int y0 = 1; y0 <= bound; y0++)
			{
				// T ga = gaussian(-y0 + d, scale);
				T ga = gaussian_array[y0 - 1];
				// T gb = gaussian(y0 + d, scale);
				T gb = gaussian_array[y0 - 1 + bound];
				int ya = y - y0;
				int yb = y + y0;
				if (ya >= 0) target[x + ya * width] += t * ga;
				if (yb < height) target[x + yb * width] += t * gb;
			}
		atomicAdd(&target[x + y * width],t*g_const_y);

	}
}

template <typename T>
void GaussianSplatSTY_GPU(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add,T* gaussian_array_y) {
    dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
	GaussianSplatSTY_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height,(T)scale, (T)d, boundary, add,(T *) gaussian_array_y);
	CHECK_LAUNCH_ERROR();
}

template void GaussianSplatSTY_GPU<float>(float* target,const float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add,float* gaussian_array_y);
template void GaussianSplatSTY_GPU<double>(double* target,const double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add,double* gaussian_array_y);

template void GaussianSplatSTX_GPU<float>(float* target,const float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add,float* gaussian_array_x);
template void GaussianSplatSTX_GPU<double>(double* target,const double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add,double* gaussian_array_x);

//********************************************************************************************************************//

template <typename T>
__global__ void Gamma_GPU_Kernel(T* target,int target_width,T gamma)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	T v = target[x + y * target_width];
	if(v>0.0f) target[x + y * target_width] = pow(v, T(1)/gamma);
}

template <typename T> 
void Gamma_GPU(T* target, int target_width,int target_height,T gamma)
{
	dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (target_width + block_size.x - 1) / block_size.x;
	grid_size.y = (target_height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;
	Gamma_GPU_Kernel<<<grid_size,block_size>>>((T*)target,target_width,(T)gamma);
	CHECK_LAUNCH_ERROR(); 
}

template void Gamma_GPU<float>(float* target,  int target_width, int target_height,float gamma);
template void Gamma_GPU<double>(double* target,  int target_width, int target_height,double gamma);