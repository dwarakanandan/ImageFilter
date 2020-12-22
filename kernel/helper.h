#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cuda_runtime_api.h>
#include <thread>
#include <chrono>

//#define CALL_LOG
//#define KERNEL_TIMINGS

#ifdef CALL_LOG
#include <iostream>
#endif

#ifdef KERNEL_TIMINGS
#include <iostream>
#include <chrono>
#define TIMER_START(A) \
	checkCudaErrors(cuCtxSynchronize()); \
	auto start_time = std::chrono::high_resolution_clock::now(); \
	std::cout << A << std::endl
#define TIMER_END() \
	checkCudaErrors(cuCtxSynchronize()); \
	auto end_time = std::chrono::high_resolution_clock::now(); \
	std::cout << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl
#else
#define TIMER_START(A)
#define TIMER_END()
#endif

//#define CHECK_LAUNCH_ERROR()
#define CHECK_LAUNCH_ERROR() checkCudaErrors(cudaDeviceSynchronize())

class DebugStreamBuffer : public std::filebuf
{
private:
	bool cout_active;
public:
	DebugStreamBuffer() { cout_active = true; std::filebuf::open("NUL", std::ios::out); }
	void open(const char fname[])
	{
		close();
		std::filebuf::open(fname ? fname : "NUL", std::ios::out);
	}
	void append(const char fname[])
	{
		close();
		std::filebuf::open(fname ? fname : "NUL", std::ios::out | std::ios::app);
	}
	void close() { std::filebuf::close(); }
	virtual int sync()
	{
		if (cout_active)
		{
			size_t out_waiting = pptr() - pbase();
			std::string s(pbase(), out_waiting);
			std::cout << s.c_str();
		}
		return std::filebuf::sync();
	}
	void deactivate_cout() { cout_active = false; }
	void activate_cout() { cout_active = true; }
};

#ifndef _NOEXCEPT
#define _NOEXCEPT
#endif

class DebugStream : public std::ostream
{
public:
	DebugStream() : std::ios(0), std::ostream(new DebugStreamBuffer()) {}
	~DebugStream() _NOEXCEPT { std::ostream::flush(); close(); delete rdbuf(); }
	void open(const char fname[] = 0) { ((DebugStreamBuffer*)rdbuf())->open(fname); }
	void append(const char fname[] = 0) { ((DebugStreamBuffer*)rdbuf())->append(fname); }
	void close() { ((DebugStreamBuffer*)rdbuf())->close(); }

	void deactivate_cout() { ((DebugStreamBuffer*)rdbuf())->deactivate_cout(); }
	void activate_cout() { ((DebugStreamBuffer*)rdbuf())->activate_cout(); }
};

extern DebugStream logger;

#include <stdio.h>
#include <sstream>
#include <driver_types.h>
#include <iostream>
#include <algorithm>

#include <iostream>
#include <iomanip>
#include <fstream>

#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// Error Code string definitions here
typedef struct
{
    char const *error_string;
    int  error_id;
} s_CudaErrorStr;

/**
 * Error codes
 */
static s_CudaErrorStr sCudaErrorString[] =
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cudaEventQuery() and ::cudaStreamQuery()).
     */
	{ "cudaSuccess", 0},
  
    /**
     * The device function being invoked (usually via ::cudaLaunch()) was not
     * previously configured via the ::cudaConfigureCall() function.
     */
    { "cudaErrorMissingConfiguration", 1 },
  
    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    { "cudaErrorMemoryAllocation", 2 },
  
    /**
     * The API call failed because the CUDA driver and runtime could not be
     * initialized.
     */
    { "cudaErrorInitializationError", 3 },
  
    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The device cannot be used until
     * ::cudaThreadExit() is called. All existing device memory allocations
     * are invalid and must be reconstructed if the program is to continue
     * using CUDA.
     */
    { "cudaErrorLaunchFailure", 4 },
  
    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    { "cudaErrorPriorLaunchFailure", 5 },
  
    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device property
     * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
     * for more information. The device cannot be used until ::cudaThreadExit()
     * is called. All existing device memory allocations are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    { "cudaErrorLaunchTimeout", 6 },
  
    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::cudaErrorInvalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    { "cudaErrorLaunchOutOfResources", 7 },
  
    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    { "cudaErrorInvalidDeviceFunction", 8 },
  
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See ::cudaDeviceProp for more
     * device limitations.
     */
    { "cudaErrorInvalidConfiguration", 9 },
  
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    { "cudaErrorInvalidDevice", 10 },
  
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    { "cudaErrorInvalidValue", 11 },
  
    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    { "cudaErrorInvalidPitchValue", 12 },
  
    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    { "cudaErrorInvalidSymbol", 13 },
  
    /**
     * This indicates that the buffer object could not be mapped.
     */
    { "cudaErrorMapBufferObjectFailed", 14 },
  
    /**
     * This indicates that the buffer object could not be unmapped.
     */
    { "cudaErrorUnmapBufferObjectFailed", 15 },
  
    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     */
    { "cudaErrorInvalidHostPointer", 16 },
  
    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     */
    { "cudaErrorInvalidDevicePointer", 17 },
  
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    { "cudaErrorInvalidTexture", 18 },
  
    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
     */
    { "cudaErrorInvalidTextureBinding", 19 },
  
    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
     */
    { "cudaErrorInvalidChannelDescriptor", 20 },
  
    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::cudaMemcpyKind.
     */
    { "cudaErrorInvalidMemcpyDirection", 21 },
  
    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the CUDA 3.1 release.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * ::cudaGetSymbolAddress().
     */
    { "cudaErrorAddressOfConstant", 22 },
  
    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    { "cudaErrorTextureFetchFailed", 23 },
  
    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    { "cudaErrorTextureNotBound", 24 },
  
    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    { "cudaErrorSynchronizationError", 25 },
  
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by CUDA.
     */
    { "cudaErrorInvalidFilterSetting", 26 },
  
    /**
     * This indicates that an attempt was made to read a non-float texture as a
     * normalized float. This is not supported by CUDA.
     */
    { "cudaErrorInvalidNormSetting", 27 },
  
    /**
     * Mixing of device and device emulation code was not allowed.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    { "cudaErrorMixedDeviceExecution", 28 },
  
    /**
     * This indicates that a CUDA Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * CUDA driver has been unloaded.
     */
    { "cudaErrorCudartUnloading", 29 },
  
    /**
     * This indicates that an unknown internal error has occurred.
     */
    { "cudaErrorUnknown", 30 },

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of CUDA will never return this error.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    { "cudaErrorNotYetImplemented", 31 },
  
    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    { "cudaErrorMemoryValueTooLarge", 32 },
  
    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::cudaStream_t and
     * ::cudaEvent_t.
     */
    { "cudaErrorInvalidResourceHandle", 33 },
  
    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::cudaSuccess (which indicates completion). Calls that
     * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
     */
    { "cudaErrorNotReady", 34 },
  
    /**
     * This indicates that the installed NVIDIA CUDA driver is older than the
     * CUDA runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    { "cudaErrorInsufficientDriver", 35 },
  
    /**
     * This indicates that the user has called ::cudaSetValidDevices(),
     * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
     * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
     * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::CUcontext active on the
     * host thread.
     */
    { "cudaErrorSetOnActiveProcess", 36 },
  
    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    { "cudaErrorInvalidSurface", 37 },
  
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    { "cudaErrorNoDevice", 38 },
  
    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    { "cudaErrorECCUncorrectable", 39 },
  
    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    { "cudaErrorSharedObjectSymbolNotFound", 40 },
  
    /**
     * This indicates that initialization of a shared object failed.
     */
    { "cudaErrorSharedObjectInitFailed", 41 },
  
    /**
     * This indicates that the ::cudaLimit passed to the API call is not
     * supported by the active device.
     */
    { "cudaErrorUnsupportedLimit", 42 },
  
    /**
     * This indicates that multiple global or constant variables (across separate
     * CUDA source files in the application) share the same string name.
     */
    { "cudaErrorDuplicateVariableName", 43 },
  
    /**
     * This indicates that multiple textures (across separate CUDA source
     * files in the application) share the same string name.
     */
    { "cudaErrorDuplicateTextureName", 44 },
  
    /**
     * This indicates that multiple surfaces (across separate CUDA source
     * files in the application) share the same string name.
     */
    { "cudaErrorDuplicateSurfaceName", 45 },
  
    /**
     * This indicates that all CUDA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long
     * running CUDA kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active CUDA work being performed.
     */
    { "cudaErrorDevicesUnavailable", 46 },
  
    /**
     * This indicates that the device kernel image is invalid.
     */
    { "cudaErrorInvalidKernelImage", 47 },
  
    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    { "cudaErrorNoKernelImageForDevice", 48 },
  
    /**
     * This indicates that the current context is not compatible with this
     * the CUDA Runtime. This can only occur if you are using CUDA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version 
     * of the API, because the Runtime API call expects a primary driver 
     * context and the Driver context is not primary, or because the Driver 
     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
     * with the CUDA Driver API" for more information.
     */
    { "cudaErrorIncompatibleDriverContext", 49 },
      
    /**
     * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    { "cudaErrorPeerAccessAlreadyEnabled", 50 },
    
    /**
     * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to 
     * disable peer addressing which has not been enabled yet via 
     * ::cudaDeviceEnablePeerAccess().
     */
    { "cudaErrorPeerAccessNotEnabled", 51 },
    
    /**
     * This indicates that a call tried to access an exclusive-thread device that 
     * is already in use by a different thread.
     */
    { "cudaErrorDeviceAlreadyInUse", 54 },

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    { "cudaErrorProfilerDisabled", 55 },

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cudaProfilerStart or
     * ::cudaProfilerStop without initialization.
     */
    { "cudaErrorProfilerNotInitialized", 56 },

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStart() when profiling is already enabled.
     */
    { "cudaErrorProfilerAlreadyStarted", 57 },

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStop() when profiling is already disabled.
     */
     { "cudaErrorProfilerAlreadyStopped", 58 },

    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again until ::cudaThreadExit() is called. All existing 
     * allocations are invalid and must be reconstructed if the program is to
     * continue using CUDA. 
     */
    { "cudaErrorAssert", 59 },
  
    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::cudaEnablePeerAccess().
     */
    { "cudaErrorTooManyPeers", 60 },
  
    /**
     * This error indicates that the memory range passed to ::cudaHostRegister()
     * has already been registered.
     */
    { "cudaErrorHostMemoryAlreadyRegistered", 61 },
        
    /**
     * This error indicates that the pointer passed to ::cudaHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    { "cudaErrorHostMemoryNotRegistered", 62 },

    /**
     * This error indicates that an OS call failed.
     */
    { "cudaErrorOperatingSystem", 63 },

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    { "cudaErrorPeerAccessUnsupported", 64 },

    /**
     * This error indicates that a device runtime grid launch did not occur 
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches. 
     */
    { "cudaErrorLaunchMaxDepthExceeded", 65 },

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped textures which are unsupported by the device runtime. 
     * Kernels launched via the device runtime only support textures created with 
     * the Texture Object API's.
     */
    { "cudaErrorLaunchFileScopedTex", 66 },

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    { "cudaErrorLaunchFileScopedSurf", 67 },

    /**
     * This error indicates that a call to ::cudaDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device 
     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on 
     * launched grids at a greater depth successfully, the maximum nested 
     * depth at which ::cudaDeviceSynchronize will be called must be specified 
     * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime. 
     * Keep in mind that additional levels of sync depth require the runtime 
     * to reserve large amounts of device memory that cannot be used for 
     * user allocations.
     */
    { "cudaErrorSyncDepthExceeded", 68 },

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    { "cudaErrorLaunchPendingCountExceeded", 69 },
    
    /**
     * This error indicates the attempted operation is not permitted.
     */
    { "cudaErrorNotPermitted", 70 },

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    { "cudaErrorNotSupported", 71 },

	/**
	* Device encountered an error in the call stack during kernel execution,
	* possibly due to stack corruption or exceeding the stack size limit.
	* The context cannot be used, so it must be destroyed (and a new one should be created).
	* All existing device memory allocations from this context are invalid
	* and must be reconstructed if the program is to continue using CUDA.
	*/
	{ "cudaErrorHardwareStackError", 72 },

	/**
	* The device encountered an illegal instruction during kernel execution
	* The context cannot be used, so it must be destroyed (and a new one should be created).
	* All existing device memory allocations from this context are invalid
	* and must be reconstructed if the program is to continue using CUDA.
	*/
	{ "cudaErrorIllegalInstruction", 73 },

	/**
	* The device encountered a load or store instruction
	* on a memory address which is not aligned.
	* The context cannot be used, so it must be destroyed (and a new one should be created).
	* All existing device memory allocations from this context are invalid
	* and must be reconstructed if the program is to continue using CUDA.
	*/
	{ "cudaErrorMisalignedAddress", 74 },

	/**
	* While executing a kernel, the device encountered an instruction
	* which can only operate on memory locations in certain address spaces
	* (global, shared, or local), but was supplied a memory address not
	* belonging to an allowed address space.
	* The context cannot be used, so it must be destroyed (and a new one should be created).
	* All existing device memory allocations from this context are invalid
	* and must be reconstructed if the program is to continue using CUDA.
	*/
	{ "cudaErrorInvalidAddressSpace", 75 },

	/**
	* The device encountered an invalid program counter.
	* The context cannot be used, so it must be destroyed (and a new one should be created).
	* All existing device memory allocations from this context are invalid
	* and must be reconstructed if the program is to continue using CUDA.
	*/
	{ "cudaErrorInvalidPc", 76 },

	/**
	* The device encountered a load or store instruction on an invalid memory address.
	* The context cannot be used, so it must be destroyed (and a new one should be created).
	* All existing device memory allocations from this context are invalid
	* and must be reconstructed if the program is to continue using CUDA.
	*/
	{ "cudaErrorIllegalAddress", 77 },

	/**
	* A PTX compilation failed. The runtime may fall back to compiling PTX if
	* an application does not contain a suitable binary for the current device.
	*/
	{ "cudaErrorInvalidPtx", 78 },

	/**
	* This indicates an error with the OpenGL or DirectX context.
	*/
	{ "cudaErrorInvalidGraphicsContext", 79 },

    /**
     * This indicates an internal startup failure in the CUDA runtime.
     */
    { "cudaErrorStartupFailure", 0x7f },

    /**
     * Any unhandled CUDA driver error is added to this value and returned via
     * the runtime. Production releases of CUDA should not return such errors.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    { "cudaErrorApiFailureBase", 10000 },

    { NULL, -1 }
};

inline const char *getCudaErrorString(cudaError_t error_id)
{
    int index = 0;

    while (sCudaErrorString[index].error_id != error_id &&
           sCudaErrorString[index].error_id != -1)
    {
        index++;
    }

    if (sCudaErrorString[index].error_id == error_id)
        return (const char *)sCudaErrorString[index].error_string;
    else
        return (const char *)"CUDA_ERROR not found!";
}

inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
    if ((cudaError_t) 0 != err)
    {
        std::cerr << "checkCudaErrors() error = " << err << " \"" << getCudaErrorString(err) << "\" from file <" << file << ">, line " << line << "." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(100));;
		exit(EXIT_FAILURE);
    }
}

#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x10, 8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11, 8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12, 8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13, 8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM1xx class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM2xx class
		{ 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
		{ 0x61, 128 }, // Pascal Generation (SM 6.1) GP10x class
		{ -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
	return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
	int current_device = 0, sm_per_multiproc = 0;
	float max_compute_perf = 0.0f;
	int max_perf_device = 0;
	int device_count = 0, best_SM_arch = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&device_count);

	checkCudaErrors(cudaGetDeviceCount(&device_count));

	if (device_count == 0)
	{
		fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	// Find the best major SM Architecture GPU device
	while (current_device < device_count)
	{
		cudaGetDeviceProperties(&deviceProp, current_device);

		// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
		if (deviceProp.computeMode != cudaComputeModeProhibited)
		{
			if (deviceProp.major > 0 && deviceProp.major < 9999)
			{
				best_SM_arch = std::max(best_SM_arch, deviceProp.major);
			}
		}

		current_device++;
	}

	// Find the best CUDA capable GPU device
	current_device = 0;

	while (current_device < device_count)
	{
		cudaGetDeviceProperties(&deviceProp, current_device);

		// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
		if (deviceProp.computeMode != cudaComputeModeProhibited)
		{
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)
			{
				sm_per_multiproc = 1;
			}
			else
			{
				sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
			}

			float compute_perf = (float) deviceProp.multiProcessorCount * (float) sm_per_multiproc * (float) deviceProp.clockRate;

			std::cout << "Found Device " << current_device << " SM" << deviceProp.major << "." << deviceProp.minor << " computer perf = " << compute_perf << std::endl;

			if (compute_perf  > max_compute_perf)
			{
				// If we find GPU with SM major > 2, search only these
				if (best_SM_arch > 2)
				{
					// If our device==dest_SM_arch, choose this, or else pass
					if (deviceProp.major == best_SM_arch)
					{
						max_compute_perf = compute_perf;
						max_perf_device = current_device;
					}
				}
				else
				{
					max_compute_perf = compute_perf;
					max_perf_device = current_device;
				}
			}
		}

		++current_device;
	}

	std::cout << "Selected Device " << max_perf_device << std::endl;

	return max_perf_device;
}

inline int getCoreCount()
{
	int current_device;
	cudaDeviceProp deviceProp;

	cudaGetDevice(&current_device);
	cudaGetDeviceProperties(&deviceProp, current_device);

	return deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
}
