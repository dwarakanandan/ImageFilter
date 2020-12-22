#include "sample.h"

__global__
void init_linear_kernel(int *linear, int count)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (x >= count) return;

	linear[x] = x;
}

void initLinear(void *linear, int count)
{
	dim3 block_size;
	block_size.x = 256;
	block_size.y = 1;
	block_size.z = 1;

	dim3 grid_size;
	grid_size.x = (count + block_size.x - 1) / block_size.x;
	grid_size.z = 1;

	init_linear_kernel<<<grid_size, block_size>>>((int *)linear, count);
	CHECK_LAUNCH_ERROR();
}
