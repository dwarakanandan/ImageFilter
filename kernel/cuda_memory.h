#pragma once

#include "helper.h"
#include "cuda_runtime_api.h"
#include <vector>

template <typename T>
void upload2DData(T *d_ptr, size_t pitch, T *h_ptr, unsigned int width, unsigned int height);

template <typename T>
void uploadData(T *d_ptr, T *h_ptr, unsigned int width);

template <typename T>
void download2DData(T *h_ptr, T *d_ptr, size_t pitch, unsigned int width, unsigned int height);

template <typename T>
void downloadData(T *h_ptr, T *d_ptr, unsigned int width);

template <typename T>
void freeDeviceData(T *&d_ptr);

template <typename T>
void freeHostData(T *&h_ptr);

template <typename T, class Allocator>
void freeDeviceData(std::vector<T*, Allocator> &d_ptr);

template <typename T, class Allocator>
void freeHostData(std::vector<T*, Allocator> &h_ptr);

void freeDeviceData(cudaArray_t &d_ptr);

template <class Allocator>
void freeDeviceData(std::vector<cudaArray_t, Allocator> &d_ptr);

template <typename T>
void allocDeviceData(T *&d_ptr, size_t &pitch, unsigned int width, unsigned int height, const char *file, const int line, bool fail_for_system = false);

template <typename T>
void allocDeviceData(T *&d_ptr, size_t width, const char *file, const int line, bool fail_for_system = false);

template <typename T>
void allocHostData(T *&h_ptr, size_t width, const char *file, const int line);
