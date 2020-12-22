#pragma once
#include <string.h>

#include <deque>
#include <vector>
#include <mutex>
#include "cuda_memory.h"

class AllocationLogger
{
	std::deque<void *> allocated, allocated2;
	std::deque<size_t> size, size2;
	std::deque<char *> alloc_file, alloc2_file;
	std::deque<int> alloc_line, alloc2_line;
	size_t peakDevice, peakHost;
	size_t currentDevice, currentHost;
	std::mutex lock;
private:
	std::string commify(size_t n)
	{
		std::string s;
		int cnt = 0;
		do
		{
			s.insert(0, 1, char('0' + n % 10));
			n /= 10;
			if (++cnt == 3 && n)
			{
				s.insert(0, 1, ',');
				cnt = 0;
			}
		} while (n);
		return s;
	}

public:
	AllocationLogger() { peakDevice = peakHost = currentDevice = currentHost = (size_t)0;  }
	~AllocationLogger() {}
	void destroy()
	{
		logger << "Peak memory usage:" << std::endl;
		logger << " device: " << commify(peakDevice) << " bytes" << std::endl;
		logger << " system: " << commify(peakHost) << " bytes" << std::endl;
		if (allocated.empty() && allocated2.empty()) return;
		logger << "Memory leak list:" << std::endl;
		while (!allocated.empty())
		{
			logger << "  leaked " << commify(size.front()) << " bytes at 0x" << std::hex << (size_t)allocated.front() << std::dec << " (device): " << alloc_file.front() << ":" << alloc_line.front() << std::endl;
			size.pop_front();
			allocated.pop_front();
			alloc_file.pop_front();
			alloc_line.pop_front();
		}
		while (!allocated2.empty())
		{
			logger << "  leaked " << commify(size2.front()) << " bytes at 0x" << std::hex << (size_t)allocated2.front() << std::dec << " (system): " << alloc2_file.front() << ":" << alloc2_line.front() << std::endl;
			size2.pop_front();
			allocated2.pop_front();
			alloc2_file.pop_front();
			alloc2_line.pop_front();
		}
	}

	template <class T>
	void free(T a)
	{
		std::lock_guard<std::mutex> guard(lock);
		for (size_t i = 0; i < allocated.size(); i++)
		{
			if ((void *)a == allocated[i])
			{
				//logger << "Freeing device memory at 0x" << std::hex << (size_t)a << std::dec << std::endl;
				currentDevice -= size[i];
				allocated[i] = allocated.back();
				allocated.pop_back();
				size[i] = size.back();
				size.pop_back();
				alloc_line[i] = alloc_line.back();
				alloc_line.pop_back();
				alloc_file[i] = alloc_file.back();
				alloc_file.pop_back();
				return;
			}
		}
		for (size_t i = 0; i < allocated2.size(); i++)
		{
			if ((void *)a == allocated2[i])
			{
				//logger << "Freeing system memory at 0x" << std::hex << (size_t)a << std::dec << std::endl;
				currentHost -= size2[i];
				allocated2[i] = allocated2.back();
				allocated2.pop_back();
				size2[i] = size2.back();
				size2.pop_back();
				alloc2_line[i] = alloc2_line.back();
				alloc2_line.pop_back();
				alloc2_file[i] = alloc2_file.back();
				alloc2_file.pop_back();
				return;
			}
		}
	}

	template <class T>
	void alloc(T a, size_t s, const char *file, const int line)
	{
		std::lock_guard<std::mutex> guard(lock);
		//logger << "Allocating " << s * sizeof(T) << " bytes of device memory at 0x" << std::hex << (size_t)a << std::dec << " \"" << file << ":" << line << std::endl;
		currentDevice += s;
		peakDevice = std::max(peakDevice, currentDevice);
		allocated.push_back((void *)a);
		size.push_back(s);
		alloc_file.push_back((char *)file);
		alloc_line.push_back(line);
	}

	template <class T>
	void alloc2(T a, size_t s, const char *file, const int line)
	{
		std::lock_guard<std::mutex> guard(lock);
		//logger << "Allocating " << s * sizeof(T) << " bytes of system memory at 0x" << std::hex << (size_t)a << std::dec << " \"" << file << ":" << line << std::endl;
		currentHost += s;
		peakHost = std::max(peakHost, currentHost);
		allocated2.push_back(a);
		size2.push_back(s);
		alloc2_file.push_back((char *)file);
		alloc2_line.push_back(line);
	}

	long long getCurrentDevice() { return currentDevice; }
	long long getCurrentHost() { return currentHost; }
};

extern AllocationLogger allocationLogger;

template <typename T>
void upload2DData(T *d_ptr, size_t pitch, T *h_ptr, unsigned int width, unsigned int height)
{
	checkCudaErrors(cudaMemcpy2DAsync((void *)d_ptr, pitch, h_ptr, width * sizeof(T), width * sizeof(T), height, cudaMemcpyHostToDevice));
}

template <typename T>
void download2DData(T *h_ptr, T *d_ptr, const size_t pitch, unsigned int width, unsigned int height)
{
	checkCudaErrors(cudaMemcpy2D(h_ptr, width * sizeof(T), (void *)d_ptr, pitch, width * sizeof(T), height, cudaMemcpyDeviceToHost));
}

template <typename T>
void downloadData(T *h_ptr, T *d_ptr, unsigned int width)
{
	checkCudaErrors(cudaMemcpy(h_ptr, (void *)d_ptr, width * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void uploadData(T * d_ptr, T *h_ptr, unsigned int width)
{
	checkCudaErrors(cudaMemcpyAsync((void *)d_ptr, h_ptr, width * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void allocDeviceData(T *&d_ptr, size_t &pitch, unsigned int width, unsigned int height, const char *file, const int line, bool fail_for_system)
{
	pitch = width * sizeof(T);
	cudaError_t r = cudaMalloc((void **)&d_ptr, width * height * sizeof(T));
	if (r == 0)
	{
		allocationLogger.alloc(d_ptr, pitch * height, file, line);
	}
	else if (!fail_for_system)
	{
		pitch = (((((width + 3) >> 2) << 2) * sizeof(T) + 511) >> 9) << 9;
		void *h_ptr;
		checkCudaErrors(cudaMallocHost(&h_ptr, pitch * height));
		checkCudaErrors(cudaHostGetDevicePointer((void **)&d_ptr, h_ptr, 0));
		allocationLogger.alloc2(d_ptr, pitch * height, file, line);
	}
	else
	{
		d_ptr = NULL;
	}
}

template <typename T>
void allocDeviceData(T *&d_ptr, size_t width, const char *file, const int line, bool fail_for_system)
{
	cudaError_t r = cudaMalloc((void **)&d_ptr, width * sizeof(T));
	if (r == 0)
	{
		allocationLogger.alloc(d_ptr, width * sizeof(T), file, line);
	}
	else if (!fail_for_system)
	{
		void *h_ptr;
		checkCudaErrors(cudaMallocHost(&h_ptr, width * sizeof(T)));
		checkCudaErrors(cudaHostGetDevicePointer((void **)&d_ptr, h_ptr, 0));
		allocationLogger.alloc2(d_ptr, width * sizeof(T), file, line);
	}
	else
	{
		d_ptr = NULL;
	}
}

template <typename T>
void allocPinnedData(T *&d_ptr, size_t &pitch, unsigned int width, unsigned int height, const char *file, const int line)
{
	pitch = width * sizeof(T);
	cudaError_t r = cudaMallocHost((void **)&d_ptr, width * height * sizeof(T));
	if (r == 0)
	{
		allocationLogger.alloc2(d_ptr, pitch * height, file, line);
	}
	else
	{
		d_ptr = NULL;
	}
}

template <typename T>
void allocPinnedData(T *&d_ptr, size_t width, const char *file, const int line)
{
	cudaError_t r = cudaMallocHost((void **)&d_ptr, width * sizeof(T));
	if (r == 0)
	{
		allocationLogger.alloc2(d_ptr, width * sizeof(T), file, line);
	}
	else
	{
		d_ptr = NULL;
	}
}

template <typename T>
void allocHostData(T * &h_ptr, size_t width, const char *file, const int line)
{
	if (width == 0)
	{
		h_ptr = 0;
		return;
	}
	h_ptr = new T[width]; // this one is allowed
	allocationLogger.alloc2(h_ptr, width * sizeof(T), file, line);
}

template <typename T>
void freeHostData(T *&h_ptr)
{
	if (h_ptr == (T *)NULL) return;
	allocationLogger.free(h_ptr);
	delete[] h_ptr; // this one is allowed
	h_ptr = (T *)NULL;
}

template <typename T, class Allocator>
void freeHostData(std::vector<T*, Allocator> &h_ptr)
{
	while (!h_ptr.empty())
	{
		allocationLogger.free(h_ptr.back());
		freeHostData(h_ptr.back());
		h_ptr.pop_back();
	}
}

template <typename T, class Allocator>
void freeDeviceData(std::vector<T *, Allocator> &d_ptr)
{
	while (!d_ptr.empty())
	{
		allocationLogger.free(d_ptr.back());
		freeDeviceData(d_ptr.back());
		d_ptr.pop_back();
	}
}

template <class Allocator>
void freeDeviceData(std::vector<cudaArray_t, Allocator> &d_ptr)
{
	while (!d_ptr.empty())
	{
		allocationLogger.free(d_ptr.back());
		freeDeviceData(d_ptr.back());
		d_ptr.pop_back();
	}
}

template <typename T, class Allocator>
void freePinnedData(std::vector<T *, Allocator> &d_ptr)
{
	while (!d_ptr.empty())
	{
		allocationLogger.free(d_ptr.back());
		freePinnedData(d_ptr.back());
		d_ptr.pop_back();
	}
}

template <class T>
struct LoggingAllocator {
	typedef T value_type;
	LoggingAllocator() = default;
	template <class U> constexpr LoggingAllocator(const LoggingAllocator<U>&) noexcept {}
	T* allocate(std::size_t n) {
		if (n > std::size_t(-1) / sizeof(T)) throw std::bad_alloc();
		T *p;
		allocHostData<T>(p, n, __FILE__, __LINE__);
		if (p) return p;
		throw std::bad_alloc();
	}
	void deallocate(T* p, std::size_t) noexcept { freeHostData(p); }
};
template <class T, class U>
bool operator==(const LoggingAllocator<T>&, const LoggingAllocator<U>&) { return true; }
template <class T, class U>
bool operator!=(const LoggingAllocator<T>&, const LoggingAllocator<U>&) { return false; }

template <typename T>
void freeDeviceData(T *&d_ptr)
{
	if (d_ptr == 0) return;
	allocationLogger.free(d_ptr);
	struct cudaPointerAttributes attributes;
	cudaError_t err = cudaPointerGetAttributes(&attributes, (const void *)d_ptr);
	checkCudaErrors(err);
	if (attributes.type == cudaMemoryTypeDevice)
	{
		checkCudaErrors(cudaFree(d_ptr));
	}
	else
	{
		checkCudaErrors(cudaFreeHost(attributes.hostPointer));
	}
	d_ptr = 0;
}

template <typename T>
void freePinnedData(T *&d_ptr)
{
	if (d_ptr == 0) return;
	allocationLogger.free(d_ptr);
	checkCudaErrors(cudaFreeHost(d_ptr));
	d_ptr = 0;
}

