#pragma once

#include "../kernel/kernel.h"
#include "float_image.h"
#include <vector>
#include <algorithm>
#define GAUSSIAN_RANGE 3
template <typename T>
class CudaImage {
public:
	CudaImage() { m_width = m_height = m_components = 0; m_min = T(0); m_max = T(0); m_mapped = false; }
	~CudaImage() { dealloc(); }

protected:
	unsigned int m_width;
	unsigned int m_height;
    unsigned int m_components;
	T m_min;
	T m_max;
	std::vector<T*, LoggingAllocator<void *>> m_data;
	std::vector<size_t, LoggingAllocator<size_t>> m_pitch;
	bool m_mapped;
	T* gaussian_array_x;
	T* gaussian_array_y;
	// __device__ __constant__ T r_weight_const_X;
	// __device__ __constant__ T r_weight_const_Y;
	// __device__ __constant__ T g_const_x;
	// __device__ __constant__ T g_const_y;

protected:
	void dealloc();
	
	void alloc();

	bool check(const CudaImage<T>& source);

public:
	void CreateMapped(void *linear, unsigned int width, unsigned int height, unsigned int components, unsigned int c_offset = 0)
	{
		dealloc();
		m_mapped = true;
		m_width = width;
		m_height = height;
		m_components = components;
		m_data.resize(components);
		m_pitch.resize(components);
		m_min = 0.0;
		m_max = 1.0;
		int stride = ((width * height + 63) >> 6) << 6;
		for (unsigned int cc = 0; cc < components; cc++)
		{
			m_pitch[cc] = m_width * sizeof(T);
			m_data[cc] = &(((T *)linear)[(cc + c_offset) * stride]);
			// initialize to something sensible
		}
	}

	void SetDimensions(unsigned int width, unsigned int height, unsigned int components);

	void Upload(FloatImage<T> &source);

	void Download(FloatImage<T> &dest) const;

	// calculate this = a - b
	void SubtractImage(const CudaImage<T> &a, const CudaImage<T> &b);

	// calculate this = a + b
	void AddImage(const CudaImage<T> &a, const CudaImage<T> &b);

	// calculate this = f * this
	void Attenuate(T f);

	void AttenuateImage(T L, T a, T b);

	void Clear();

	T*& GetArray(unsigned int c) { return m_data[c]; }

	T const* GetArrayConst(unsigned int c) const { return m_data[c]; }

	void ConvertLab2Rgb(const CudaImage<T> &Lab);

	void ConvertRgb2Lab(const CudaImage<T> &RGB);

	void Copy(const CudaImage<T> &source);

	void AddLab(T L, T a, T b);

	void Add(T f);

	// we need this :/
	void resize(const CudaImage<T> &source);
	void resize(unsigned int width, unsigned int height, unsigned int componenets);

	unsigned int GetComponents() { return m_components;  }

	T GetMin() { return m_min; }

	T GetMax() { return m_max; }

	void SetMin(T m) { m_min = m; }

	void SetMax(T m) { m_max = m; }

	unsigned int GetWidth() const { return m_width; }

	unsigned int GetHeight() const { return m_height; }

	void * GetData(unsigned int c) { return m_data[c]; }

	size_t GetPitch(unsigned int c) { return m_pitch[c]; }

	void Release() { SetDimensions(0, 0, 0); }

	void Gamma(T gamma);

	void ScaleFast(const CudaImage<T>& source);

	void GaussianFilter(const CudaImage<T>& source, CudaImage<T>& scratch, T scale, T dx, T dy, BoundaryCondition boundary, bool add = false);

	void GaussianSplat(const CudaImage<T>& source, CudaImage<T>& scratch, T scale, T dx, T dy, BoundaryCondition boundary, bool add = false);

	void GaussianWeightedFilter(CudaImage<T>& msk, const CudaImage<T>& source, CudaImage<T>& scratch, const CudaImage<T>& mask, T scale, T dx, T dy, BoundaryCondition boundary, bool add = false);

public:
	void GaussianFilterSTX(int target, const CudaImage<T>& source, int component, T scale, T dx, BoundaryCondition boundary, bool add);

	void GaussianFilterSTY(int target, const CudaImage<T>& source, int component, T scale, T dy, BoundaryCondition boundary, bool add);

	void GaussianSplatSTX(int target, const CudaImage<T>& source, int component, T scale, T dx, BoundaryCondition boundary, bool add);

	void GaussianSplatSTY(int target, const CudaImage<T>& source, int component, T scale, T dy, BoundaryCondition boundary, bool add);

	void CopyComponent(int target, const CudaImage<T>& source, int component);

	void SetGaussianArrays(T dx,T dy,T scale);

	void SetGaussianArrays(T* gaussian_array_x,T* gaussian_array_y);

	T* GetGaussianArrayX() { return gaussian_array_x;}

	T* GetGaussianArrayY() { return gaussian_array_y;}

	void SetConstants(T dx,T dy,T scale);

private:
	void GaussianFilterSTX(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

	void GaussianFilterSTY(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

	void GaussianSplatSTX(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

	void GaussianSplatSTY(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);
};
