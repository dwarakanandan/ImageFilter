
#include "filter.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define BLOCK_SIZE 32
#define GAUSSIAN_RANGE 3
#define COMPUTE_TYPE double
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

//********************************************************************************************************************//

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
//********************************************************************************************************************//
double gaussian_host(double x0, double size)
{
	double d = x0 / size;
	return exp(-0.5 * d * d);
}

__constant__ COMPUTE_TYPE r_weight_const_X;
__constant__ COMPUTE_TYPE r_weight_const_Y;

template <typename T>
void setConstantMemWeight_X_GPU(T d,T scale){
	T weight = gaussian_host(d, scale);
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian_host(-x0 + d, scale);
			T gb = gaussian_host(x0 + d, scale);
			weight += ga + gb;
		}
	T r_weight = 1.0f / weight;
	cudaMemcpyToSymbol(r_weight_const_X,&r_weight,sizeof(T),0,cudaMemcpyHostToDevice);
	// std::cout<<"ConstantMem Done";
}
template void setConstantMemWeight_X_GPU<float>(float d,float scale);
template void setConstantMemWeight_X_GPU<double>(double d,double scale);

template <typename T>
void setConstantMemWeight_Y_GPU(T d,T scale){
	T weight = gaussian_host(d, scale);
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian_host(-x0 + d, scale);
			T gb = gaussian_host(x0 + d, scale);
			weight += ga + gb;
		}
	T r_weight = 1.0f / weight;
	cudaMemcpyToSymbol(r_weight_const_Y,&r_weight,sizeof(T),0,cudaMemcpyHostToDevice);
	// std::cout<<"ConstantMem Done";
}
template void setConstantMemWeight_Y_GPU<float>(float d,float scale);
template void setConstantMemWeight_Y_GPU<double>(double d,double scale);

//********************************************************************************************************************//
template <typename T>
__global__ void GaussianSplatSTX_GPU_kernel(T* __restrict__ target,const T* __restrict__ source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
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
					if (xa >= 0) caa.AtomicAdd(&target[xa + y * width],t * ga);
					if (xb < width) caa.AtomicAdd(&target[xb + y * width],t * gb);
				}
				T g = gaussian(d, scale);
				caa.AtomicAdd(&target[x + y * width],t * g);
			}
		// }
	}
	else // if (boundary == BoundaryCondition::Border)
	{
		// T weight = gaussian(d, scale);
		// for (int x0 = 1; x0 <= bound; x0++)
		// {
		// 	T ga = gaussian(-x0 + d, scale);
		// 	T gb = gaussian(x0 + d, scale);
		// 	weight += ga + gb;
		// }
		// T r_weight = 1.0f / weight;
		// for (int y = 0; y < height; y++)
		// {
		// 	for (int x = 0; x < width; x++)
		// 	{
				
		// 	}
        // }
        // printf("X,WeightDone,%d,%d",x,y);
        // fflush(stdout);
        T t = source[x + y * width] * r_weight_const_X;
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
void GaussianSplatSTX_GPU(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;
    //std::cout<<"Start Splatx";
	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    setConstantMemWeight_X_GPU((T)d,(T)scale);
    GaussianSplatSTX_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add);
    CHECK_LAUNCH_ERROR();
}

template <typename T>
__global__ void GaussianSplatSTY_GPU_kernel(T* __restrict__ target,const T* __restrict__ source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    int bound = (int)floor(GAUSSIAN_RANGE * scale);
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	// if (!add) std::fill(target, target + width * height, T(0));
    CudaAtomicAdd<T> caa;
	// T weight = T(0);
	// T r_weight = T(1);
	// if (boundary != BoundaryCondition::Renormalize)
	// {
	// 	weight = gaussian(d, scale);

	// 	for (int y0 = 1; y0 <= bound; y0++)
	// 	{
	// 		T ga = gaussian(-y0 + d, scale);
	// 		T gb = gaussian(y0 + d, scale);
	// 		weight += ga + gb;
	// 	}
	// 	r_weight = 1.0f / weight;
    // }
    //printf("Y,WeightDone,%d,%d",x,y);
    // fflush(stdout);
	T t = source[x + y * width] * r_weight_const_Y;
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
	// if (boundary == BoundaryCondition::Renormalize)
	// {


	// }

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
void GaussianSplatSTY_GPU(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    setConstantMemWeight_Y_GPU((T)d,(T)scale);
	GaussianSplatSTY_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add);
	CHECK_LAUNCH_ERROR();
}

template void GaussianSplatSTY_GPU<float>(float* target,const float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add);
template void GaussianSplatSTY_GPU<double>(double* target,const double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add);

template void GaussianSplatSTX_GPU<float>(float* target,const float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add);
template void GaussianSplatSTX_GPU<double>(double* target,const double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add);

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

//********************************************************************************************************************//


template <typename T> 
__global__ void ScaleFast_GPU_Kernel(T* target, T* source, int target_width, int target_height, int source_width,  int source_height)
{
#if 1
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// std::cout << "Fast scaling image " << source_width << "x" << source_height << " -> " << m_width << "x" << m_height << std::endl;
	if (source_width > target_width)
	{
		unsigned int y0 = _min(y << 1, source_height - 1);
		unsigned int y1 = _min(y0 + 1, source_height - 1);
		unsigned int x0 = _min(x << 1, source_width - 1);
		unsigned int x1 = _min(x0 + 1, source_width - 1);
		T s = 0.25f * (source[x0 + y0*source_width] + source[x1 + y0*source_width] + source[x0 + y1*source_width] + source[x1 + y1*source_width]);
		// SetValue(x, y, c, s);
		target[x+y*target_width] = s;
	}
	else
	{
		unsigned int y0 = _min(y >> 1, source_height - 1);
		unsigned int x0 = _min(x >> 1, source_width - 1);
		// SetValue(x, y, c, source.GetValue(x0, y0, c));
		target[x+y*target_width] = source[x0 + y0*source_width];
	}
#else
	T scale_x = (T)(source_width) / (T)(target_width);
	T scale_y = (T)(source_height) / (T)(m_height);

	if (scale_x > 1.0f) scale_x = ceilf(scale_x);
	else if (scale_x < 1.0f) scale_x = 1.0f / ceilf(1.0f / scale_x);
	if (scale_y > 1.0f) scale_y = ceilf(scale_y);
	else if (scale_y < 1.0f) scale_y = 1.0f / ceilf(1.0f / scale_y);

	// unsigned int comp = source.GetComponents();
	// SetComponents(comp);
	// SetMin(0.0f);
	// SetMax(1.0f);
	// std::cout << "Fast scaling image with scale " << scale_x << ", " << scale_y << " (" << source_width << "x" << source_height << ") -> (" << target_width << "x" << m_height << ")" << std::endl;
			T ys = _max(0.0f, _min(((T)y + 0.5f) * scale_y - 0.5f, source_height - 1.0f));
			int y0 = (int)floorf(ys);
			T yw = ys - y0;
			for (unsigned int x = 0; x < target_width; x++)
			{
				T xs = _max(0.0f, _min(((T)x + 0.5f) * scale_x - 0.5f, source_width - 1.0f));
				int x0 = (int)floorf(xs);
				T xw = xs - x0;
				T s = source[x0 + y0*source_width] * (1.0f - xw) * (1.0f - yw);
				if (x0 + 1 < (int)source_width) s += source[(x0+1) + y0*source_width] * xw * (1.0f - yw);
				if (y0 + 1 < (int)source_height) s += source[x0 + (y0+1)*source_width] * (1.0f - xw) * yw;
				if ((x0 + 1 < (int)source_width) && (y0 + 1 < (int)source_height)) s += source[(x0+1) + (y0+1)*source_width] * xw * yw;
				// SetValue(x, y, c, s);
				target[x+y*target_width] = s;
			}
#endif
}

template <typename T>
void ScaleFast_GPU(T* target, T* source, int target_width, int target_height,  int source_width,  int source_height)
{
	dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (target_width + block_size.x - 1) / block_size.x;
	grid_size.y = (target_height + block_size.y - 1) / block_size.y;
	grid_size.z = 1;
	ScaleFast_GPU_Kernel<<<grid_size,block_size>>>((T*)target,(T*)source,target_width,target_height,source_width,source_height);
	CHECK_LAUNCH_ERROR();
}

template void ScaleFast_GPU<float>(float* target,float* source, int target_width, int target_height, int source_width, int source_height);
template void ScaleFast_GPU<double>(double* target, double* source, int target_width, int target_height, int source_width, int source_height);
//********************************************************************************************************************//