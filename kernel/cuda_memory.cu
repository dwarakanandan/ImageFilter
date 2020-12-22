#include "cuda_memory.h"
#include "kernel_cu.h"

void freeDeviceData(cudaArray_t &d_ptr)
{
	if (d_ptr == (cudaArray_t)NULL) return;
	allocationLogger.free(d_ptr);
	{
		checkCudaErrors(cudaFreeArray(d_ptr));
	}
	d_ptr = (cudaArray_t)NULL;
}
