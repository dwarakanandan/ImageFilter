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
__global__ void GaussianFilterSTY_GPU_kernel(T* __restrict__ target, T* __restrict__ source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add,const T* __restrict__ gaussian_array) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    int bound = (int)floor(GAUSSIAN_RANGE * scale);
    T weight = T(0);
	T r_weight = T(1);
	// if(x==0&&y==0) printf("Y - Pixel:%f \n",source[x + y * width]);
	// if(x==100&&y==0) printf("Y - Pixel:%f \n",source[x + y * width]);
	// if(x==0&&y==100) printf("Y - Pixel:%f \n",source[x + y * width]);
	// if(x==width-1&&y==height-1) printf("Y - Pixel:%f \n",source[x + y * width]);

	if (boundary == BoundaryCondition::Renormalize)
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
void GaussianFilterSTY_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* gaussian_array_y) {
	dim3 block_size;
		block_size.x = BLOCK_SIZE;
		block_size.y = BLOCK_SIZE;
		block_size.z = 1;

		dim3 grid_size;
		grid_size.x = (width + block_size.x - 1) / block_size.x;
		grid_size.y = (height + block_size.y - 1) / block_size.y;
		grid_size.z = 1;
		
		GaussianFilterSTY_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add, (T const*)gaussian_array_y);
		//checkCudaErrors(cudaStreamSynchronize(0));
	CHECK_LAUNCH_ERROR();
}

template <typename T>
__global__ void GaussianFilterSTX_GPU_kernel(T* __restrict__ target, T* __restrict__ source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add,const T* __restrict__ gaussian_array) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    int bound = (int)floor(GAUSSIAN_RANGE * scale);

	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int guard = ((bound / width) + 1) * width;

	T weight = T(0);
	T r_weight = T(1);
	// if(x==0&&y==0) printf("X - Pixel:%f \n",source[x + y * width]);
	// if(x==100&&y==0) printf("X - Pixel:%f \n",source[x + y * width]);
	// if(x==0&&y==100) printf("X - Pixel:%f \n",source[x + y * width]);
	// if(x==width-1&&y==height-1) printf("X - Pixel:%f \n",source[x + y * width]);

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

    if (add) target[x + y * width] += t * r_weight;
    else target[x + y * width] = t * r_weight;
}

template <typename T>
void GaussianFilterSTX_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add) {
    dim3 block_size;
	block_size.x = BLOCK_SIZE;
	block_size.y = BLOCK_SIZE;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (width + block_size.x - 1) / block_size.x;
	grid_size.y = (height + block_size.y - 1) / block_size.y;
    grid_size.z = 1;
    
	GaussianFilterSTX_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add, (T const*)gaussian_array_x);
	//checkCudaErrors(cudaStreamSynchronize(0));
    CHECK_LAUNCH_ERROR();
}

template void GaussianFilterSTY_GPU<float>(float* target, float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add);
template void GaussianFilterSTY_GPU<double>(double* target, double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add);

template void GaussianFilterSTX_GPU<float>(float* target, float* source, int width, int height, float scale, float d, BoundaryCondition boundary, bool add);
template void GaussianFilterSTX_GPU<double>(double* target, double* source, int width, int height, double scale, double d, BoundaryCondition boundary, bool add);

//********************************************************************************************************************//
double gaussian_host_cu(double x0, double size)
{
	double d = x0 / size;
	return exp(-0.5 * d * d);
}

__constant__ COMPUTE_TYPE r_weight_const_X;
__constant__ COMPUTE_TYPE r_weight_const_Y;
__constant__ COMPUTE_TYPE g_const_x;
__constant__ COMPUTE_TYPE g_const_y;

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
	// std::cout<<"ConstantMem Done";
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

//********************************************************************************************************************//
template <typename T>
__global__ void GaussianSplatSTX_GPU_kernel(T* __restrict__ target,const T* __restrict__ source,int width, int height, T scale, T d, BoundaryCondition boundary, bool add,const T* __restrict__ gaussian_array) {
    int bound = (int)floor(GAUSSIAN_RANGE * scale);
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	// if (!add) std::fill(target, target + width * height, T(0));
	if(!add){
		target[x + y * width] = T(0);
		__syncthreads();
	}

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
            int xa = _max(0, x - x0);
            int xb = _min(x + x0, width - 1);
            atomicAdd(&target[xa + y * width],t*ga);
            atomicAdd(&target[xb + y * width],t*gb);
        }
        atomicAdd(&target[x + y * width],t*g_const_x);
        
        
	}
	// if(0 == x && 0==y) {
	// 	printf("target value = %lf\n", target[x + y * width]);
	// 	printf("target value1 = %lf\n", target[x+1 + y * width]);
	// 	// printf("intended target value = %lf\n", target[minXIndex + y * width]);
	// 	// printf("shared value = %lf\n", shared_target_pixels[threadIdx.y][j]);
	// }
}


template <typename T>
__global__ void GaussianSplatSTX_GPU_kernel_shared(T* __restrict__ target, T* __restrict__ source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add,const T* __restrict__ gaussian_array){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// int shared_size = blockDim.y*(blockDim.x+bound*2);

	extern __shared__ char array[];
	T* values = reinterpret_cast<T*>(array);
	// int* index = (int*)&values[shared_size];
	int block_start = blockDim.x*blockIdx.x;
	int block_end = blockDim.x*(blockIdx.x+1) -1;
	int size_row = blockDim.x + bound*2;
	T t = source[x + y * width] * r_weight_const_X;
	values[ty*size_row + x-block_start + bound] = 0;
	// if(tx==0)
	// {
	// 	#pragma unroll
	// 	for(int x0 = 0; x0 < size_row; x0++)
	// 	{
	// 		values[ty*size_row + x0] = 0;
	// 	}
	// }
	if(tx==0){
		for(int x0 = 0; x0 < bound; x0++)
		{
			values[ty*size_row + x0] = 0;
		}
	}else if (tx==blockDim.x-1){
		for(int x0 = size_row-bound; x0 < size_row; x0++)
		{
			values[ty*size_row + x0] = 0;
		}
	}
	__syncthreads();
	int row_start = block_start;
	// int x_temp = x-row_start;
	for (int x0 = 1; x0 <= bound; x0++)
	{
		// T ga = gaussian(-x0 + d, scale);
		T ga = gaussian_array[x0-1];
		// T gb = gaussian(x0 + d, scale);
		T gb = gaussian_array[x0+bound-1];
		int xa = _max(0, x - x0);
		int xb = _min(x + x0, width - 1);
		atomicAdd(&values[ty*size_row + (xa-row_start+bound)],t*ga);
		atomicAdd(&values[ty*size_row + (xb-row_start+bound)],t*gb);
	}
	__syncthreads();
	// int curr_pixel_index = x+y*width;
	// T sum = ;
	atomicAdd(&target[x + y * width],t*g_const_x + values[ty*size_row + x-block_start + bound]);

	if(tx==0){
		if(x-bound<block_start && x-bound>0){
			
			for (int x0 = 1; x0 <= bound; x0++)
			{
				int curr_edge_pixel_index = (x-x0) + y*width;
				int xa = x-x0;
				atomicAdd(&target[curr_edge_pixel_index],values[ty*size_row + (xa-row_start+bound)]);
			}
		}
	}else if (tx==blockDim.x-1){
		if(x+bound>block_end && x+bound<width-1){
			for (int x0 = 1; x0 <= bound; x0++)
			{
				int curr_edge_pixel_index = (x+x0) + y*width;
				int xb = x+x0;
				atomicAdd(&target[curr_edge_pixel_index],values[ty*size_row + (xb - row_start + bound)]);
			}
		}
	}
	// __syncthreads();
	// if(0 == x && 0==y) {
	// 	printf("target value = %lf\n", target[x + y * width]);
	// 	printf("target value1 = %lf\n", target[x + 1 + y * width]);
	// 	// printf("shared value = %lf , %f\n", values[ty*size_row + x-block_start + bound]);
	// 	// printf("shared value1 = %lf\n", values[ty*size_row + x + 1-block_start + bound]);
	// 	// printf("shared value = %lf\n", shared_target_pixels[threadIdx.y][j]);
	// }
}



template <typename T>
void GaussianSplatSTX_GPU(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add,T* gaussian_array_x) {
	dim3 block_size;
	int block_size_shared = 32;
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int size_shared_mem = (block_size_shared*(block_size_shared+bound*2))*sizeof(T); 
	// std::cout<<"Shared MemSize:"<<size_shared_mem<<std::endl;
	if(size_shared_mem<=65535){
		block_size.x = block_size_shared;
		block_size.y = block_size_shared;
		block_size.z = 1;

		dim3 grid_size;
		grid_size.x = (width + block_size.x - 1) / block_size.x;
		grid_size.y = (height + block_size.y - 1) / block_size.y;
		grid_size.z = 1;
		// std::cout<<"Grid Size:"<<grid_size.x<<" "<<grid_size.y<<std::endl;
		// std::cout<<"Width: "<<width<<", Height: "<<height;
		// cudaFuncSetCacheConfig(GaussianSplatSTX_GPU_kernel_shared,cudaFuncCachePreferShared);
		GaussianSplatSTX_GPU_kernel_shared<<<grid_size,block_size,size_shared_mem>>>((T *)target, (T *)source, width, height, (T)scale, (T)d, boundary, add, (T const*)gaussian_array_x);
	}else{	
		block_size.x = BLOCK_SIZE;
		block_size.y = BLOCK_SIZE;
		block_size.z = 1;
		dim3 grid_size;
		grid_size.x = (width + block_size.x - 1) / block_size.x;
		grid_size.y = (height + block_size.y - 1) / block_size.y;
		grid_size.z = 1;
		GaussianSplatSTX_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height,(T)scale, (T)d, boundary, add,(T const*) gaussian_array_x);
		//checkCudaErrors(cudaStreamSynchronize(0));
	}
	//checkCudaErrors(cudaStreamSynchronize(0));
    CHECK_LAUNCH_ERROR();
}

template <typename T>
__global__ void GaussianSplatSTY_GPU_kernel_shared(T* __restrict__ target, T* __restrict__ source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add,const T* __restrict__ gaussian_array){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// int shared_size = blockDim.y*(blockDim.x+bound*2);

	extern __shared__ char array[];
	T* values = reinterpret_cast<T*>(array);
	
	// int* index = (int*)&values[shared_size];
	int block_start = blockDim.y*blockIdx.y;
	int block_end = blockDim.y*(blockIdx.y+1) -1;
	int size_column = blockDim.y + bound*2;
	int size_row = blockDim.x;
	// T value[size_column][blockDim.x];

	T t = source[x + y * width] * r_weight_const_Y;
	values[(y-block_start + bound)*size_row + tx ] = 0;
	// if(tx==0)
	// {
	// 	#pragma unroll
	// 	for(int x0 = 0; x0 < size_row; x0++)
	// 	{
	// 		values[ty*size_row + x0] = 0;
	// 	}
	// }
	if(ty==0){
		for(int y0 = 0; y0 < bound; y0++)
		{
			values[y0*size_row + tx] = 0;
		}
	}else if (ty==blockDim.y-1){
		for(int y0 = size_column-bound; y0 < size_column; y0++)
		{
			values[y0*size_row + tx] = 0;
		}
	}
	__syncthreads();
	int row_start = block_start;
	// int x_temp = x-row_start;
	for (int y0 = 1; y0 <= bound; y0++)
	{
		// T ga = gaussian(-y0 + d, scale);
		T ga = gaussian_array[y0-1];
		// T gb = gaussian(y0 + d, scale);
		T gb = gaussian_array[y0+bound-1];
		int ya = _max(0, y - y0);
		int yb = _min(y + y0, height - 1);
		atomicAdd(&values[(ya-row_start+bound)*size_row + tx],t*ga);
		atomicAdd(&values[(yb-row_start+bound)*size_row + tx],t*gb);

		// atomicAdd(&values[ty*size_row + (xa-row_start+bound)],t*ga);
		// atomicAdd(&values[ty*size_row + (xb-row_start+bound)],t*gb);
	}
	__syncthreads();
	// int curr_pixel_index = x+y*width;
	// T sum = ;
	atomicAdd(&target[x + y * width],t*g_const_x + values[(y-block_start + bound)*size_row + tx ]);

	if(ty==0){
		if(y-bound<block_start && y-bound>0){
			
			for (int y0 = 1; y0 <= bound; y0++)
			{
				int curr_edge_pixel_index = x + (y-y0)*width;
				int ya = y-y0;
				atomicAdd(&target[curr_edge_pixel_index],values[(ya-row_start+bound)*size_row + tx]);
			}
		}
	}else if (ty==blockDim.y-1){
		if(y+bound>block_end && y+bound<height-1){
			for (int y0 = 1; y0 <= bound; y0++)
			{
				int curr_edge_pixel_index = x + (y+y0)*width;
				int yb = y+y0;
				atomicAdd(&target[curr_edge_pixel_index],values[(yb-row_start+bound)*size_row + tx]);
			}
		}
	}
	// __syncthreads();
	// if(0 == x && 0==y) {
	// 	printf("target value = %lf\n", target[x + y * width]);
	// 	printf("target value1 = %lf\n", target[x + 1 + y * width]);
	// 	// printf("shared value = %lf , %f\n", values[ty*size_row + x-block_start + bound]);
	// 	// printf("shared value1 = %lf\n", values[ty*size_row + x + 1-block_start + bound]);
	// 	// printf("shared value = %lf\n", shared_target_pixels[threadIdx.y][j]);
	// }
}

template <typename T>
__global__ void GaussianSplatSTY_GPU_kernel(T* __restrict__ target,const T* __restrict__ source,int width, int height, T scale, T d, BoundaryCondition boundary, bool add,const T* __restrict__ gaussian_array) {
    int bound = (int)floor(GAUSSIAN_RANGE * scale);
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(!add){
		target[x + y * width] = T(0);
		__syncthreads();
	}
	
	if (boundary != BoundaryCondition::Renormalize){
		T t = source[x + y * width] * r_weight_const_Y;
		for (int y0 = 1; y0 <= bound; y0++)
		{
			// T ga = gaussian(-y0 + d, scale);
			T ga = gaussian_array[y0 - 1];
			// T gb = gaussian(y0 + d, scale);
			T gb = gaussian_array[y0 - 1 + bound];
			int ya = _max(0, y - y0);
			int yb = _min(y + y0, height - 1);
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
	int block_size_shared = 32;
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int size_shared_mem = (block_size_shared*(block_size_shared+bound*2))*sizeof(T); 
	if(size_shared_mem<=65535){
		// std::cout<<"Shared MemSize:"<<size_shared_mem<<std::endl;
		block_size.x = block_size_shared;
		block_size.y = block_size_shared;
		block_size.z = 1;

		dim3 grid_size;
		grid_size.x = (width + block_size.x - 1) / block_size.x;
		grid_size.y = (height + block_size.y - 1) / block_size.y;
		grid_size.z = 1;
		// std::cout<<"Grid Size:"<<grid_size.x<<" "<<grid_size.y<<std::endl;
		// std::cout<<"Width: "<<width<<", Height: "<<height;
		// cudaFuncSetCacheConfig(GaussianSplatSTX_GPU_kernel_shared,cudaFuncCachePreferShared);
		GaussianSplatSTY_GPU_kernel_shared<<<grid_size, block_size,size_shared_mem>>>((T *)target, (T *)source, width, height,(T)scale, (T)d, boundary, add,(T const*) gaussian_array_y);
	}else{	
		block_size.x = BLOCK_SIZE;
		block_size.y = BLOCK_SIZE;
		block_size.z = 1;

		dim3 grid_size;
		grid_size.x = (width + block_size.x - 1) / block_size.x;
		grid_size.y = (height + block_size.y - 1) / block_size.y;
		grid_size.z = 1;
		GaussianSplatSTY_GPU_kernel<<<grid_size, block_size>>>((T *)target, (T *)source, width, height,(T)scale, (T)d, boundary, add,(T const*) gaussian_array_y);
	}
	//checkCudaErrors(cudaStreamSynchronize(0));
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


// template <typename T>
// __global__ void GaussianSplatSTY_GPU_kernel_shared(T* __restrict__ target,const T* __restrict__ source,int width, int height, T scale, T d, BoundaryCondition boundary, bool add,const T* __restrict__ gaussian_array){
// 	int x = threadIdx.x + blockIdx.x * blockDim.x;
// 	int y = threadIdx.y + blockIdx.y * blockDim.y;
// 	int bound = (int)floor(GAUSSIAN_RANGE * scale);
// 	int tx = threadIdx.x;
// 	int ty = threadIdx.y;
// 	int shared_size = blockDim.y*(blockDim.x+bound*2);
// 	extern __shared__ char array[];
// 	T* values = reinterpret_cast<T*>(array);
// 	int* index = (int*)&values[shared_size];
// 	int block_start = blockDim.y*blockIdx.y;
// 	int block_end = blockDim.y*(blockIdx.y+1) -1;
// 	int size_row = blockDim.x + bound*2;
// 	T t = source[x + y * width] * r_weight_const_X;
// 	int sharedMem_index =0;
// 	if(ty==0)
// 	{
// 		for(int y0 = (block_start-bound); y0 < (block_end+bound); y0++)
// 		{
// 			if((y0>=0) && y0<=(height-1))
// 				index[tx*blockDim.x + sharedMem_index] = x+y0*width;
// 			else
// 				index[tx*blockDim.x + sharedMem_index] = -1;
// 			sharedMem_index++;
// 		}
		
		
// 	}
// 	__syncthreads();
// 	for (int y0 = 1; y0 <= bound; y0++)
// 	{
// 		// T ga = gaussian(-y0 + d, scale);
// 		T ga = gaussian_array[y0-1];
// 		// T gb = gaussian(y0 + d, scale);
// 		T gb = gaussian_array[y0+bound-1];
// 		int ya = max(0, y - y0);
// 		int yb = min(y + y0, height - 1);
// 		int ia=-1,ib=-1;
// 		for(int i=0;i<size_row;i++){
// 			if(index[ty*blockDim.y + i]==(x+ya*width))
// 				ia=i;
// 			if(index[ty*blockDim.y + i]==(x+yb*width))
// 				ib=i;
// 			if(ia>-1&&ib>-1) break;
// 		}
// 		if(ia!=-1)
// 			atomicAdd(&values[tx*blockDim.x + ia],t*ga);
// 		if(ib!=-1)
// 			atomicAdd(&values[tx*blockDim.x + ib],t*gb);
// 		//atomicAdd(&target[xa + y * width],t*ga);
// 		//atomicAdd(&target[xb + y * width],t*gb);
// 	}
// 	__syncthreads();
// 	int curr_pixel_index = x+y*width;
// 	T sum = T(0);
// 	for(int i=0;i<size_row;i++){
// 			if(index[tx*blockDim.x + i] == curr_pixel_index){
// 				sum = values[tx*blockDim.x + i];
// 				break;
// 			}
// 		}
	
// 	atomicAdd(&target[x + y * width],sum);
// 	if(ty==0){
// 		if(y-bound<block_start && y-bound>0){
			
// 			for (int y0 = 1; y0 <= bound; y0++)
// 			{
// 				int curr_edge_pixel_index = x + (y-y0)*width;
// 				T sum_edge = T(0);
// 				for(int i=0;i<size_row;i++){
// 						if(index[tx*blockDim.x + i] == curr_edge_pixel_index)
// 							sum_edge += values[tx*blockDim.x + i];
// 				}
// 				atomicAdd(&target[curr_edge_pixel_index],sum);
// 			}
// 		}
// 	}else if (ty==blockDim.y-1){
// 		if(y+bound>block_end && y+bound<height-1){
// 			for (int y0 = 1; y0 <= bound; y0++)
// 			{
// 				int curr_edge_pixel_index = x + (y+y0)*width;
// 				T sum_edge = T(0);
// 				for(int i=0;i<size_row;i++){
// 						if(index[ty*blockDim.y + i] == curr_edge_pixel_index)
// 							sum_edge += values[ty*blockDim.y + i];
// 				}
// 				atomicAdd(&target[curr_edge_pixel_index],sum);
// 			}
// 		}
// 	}
// }
