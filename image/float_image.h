#pragma once

#include "image.h"
#include <vector>

#include "../kernel/kernel_cu.h"

enum class BoundaryCondition { Repeat, Zero, Border, Renormalize};

template <typename T>
class FloatImage {
public:
	FloatImage() { m_width = m_height = m_components = 0; m_min = T(0); m_max = T(0); m_mapped = false; }
	~FloatImage() { if (!m_mapped) dealloc(); }

protected:
	unsigned int m_width;
	unsigned int m_height;
    unsigned int m_components;
	T m_min;
	T m_max;
	std::vector<T *, LoggingAllocator<T *>> m_data;
	bool m_mapped;

private:
	void dealloc();

	void alloc();

	bool check(const FloatImage<T> &source);

public:
	void CreateMapped(T* linear, unsigned int width, unsigned int height, unsigned int components, unsigned int c_offset = 0)
	{
		dealloc();
		m_mapped = true;
		m_width = width;
		m_height = height;
		m_components = components;
		m_data.resize(components);
		m_min = 0.0;
		m_max = 1.0;
		for (unsigned int cc = 0; cc < components; cc++)
		{
			m_data[cc] = &(linear[(cc + c_offset) * width * height]);
		}
	}

	bool ReadImage(const char fileName[], bool raw = false);

	bool WriteImage(const char fileName[]);

	unsigned int GetWidth() const { return m_width; }

	unsigned int GetHeight() const { return m_height; }

	unsigned int GetComponents() const { return m_components; }

	T GetMin() const { return m_min; }

	void SetMin(T m) { m_min = m; }

	T GetMax() const { return m_max; }

	void SetMax(T m) { m_max = m; }

	void SetWidth(unsigned int width) { m_width = width; alloc(); }

	void SetHeight(unsigned int height) { m_height = height; alloc(); }

	void SetComponents(unsigned int components) { m_components = components; alloc(); }

	void Clear() { for (unsigned int c = 0; c < m_components; c++) memset(m_data[c], 0, (size_t)m_width * (size_t)m_height * sizeof(T)); }

	void Clear(unsigned int c) { memset(m_data[c], 0, (size_t)m_width * (size_t)m_height * sizeof(T)); }

	T *&GetArray(unsigned int c) { return m_data[c]; }

	T const *GetArrayConst(unsigned int c) const { return m_data[c]; }

	void SetValue(unsigned int x, unsigned int y, unsigned int c, T v) { m_data[c][x + y * m_width] = v; }

	T GetValue(unsigned int x, unsigned int y, unsigned int c) const { return m_data[c][x + y * m_width]; }

	void CalculateMinMax();

public:
	void Copy(const FloatImage<T> &source);

	void CopyReplicate(const FloatImage<T>& source, unsigned int components);

	void Attenuate(const T f);

	void Attenuate(int c, const T f);

	void AttenuateAdd(int c_dst, int c_src, const T f);

	void Add(int c_dst, int c_src);

	void AddSquared(int c_dst, int c_src);

	void Square();

	void SquareRoot();

	void Square(int c);

	void SquareRoot(int c);

	void AddImage(int c_dst, const FloatImage<T> &source, int c_src);

	void Attenuate(const FloatImage<T> &source, const T f);

	void Lerp(const FloatImage<T> &source, const T f);

	void Upload(const FloatImage<T> &source) { Copy(source); }

	void Download(FloatImage<T> &target) const { target.Copy(*this); }

	void Scale(const FloatImage<T> &source, const T scale);

	void Scale(const FloatImage<T> &source);

	void ScaleFast(const FloatImage<T> &source);

	void ScaleMasked(FloatImage<T> &mask_small, const FloatImage<T> &src, const FloatImage<T> &mask, bool fast);

	void Copy(const Image &source);

	void CopyAlpha(const Image &source);

	void Maximize(const char *name = NULL, T min = 0.0f, T max = 0.0f);

	void Maximize(T *min, T *max);

	void Gamma(T gamma);

	void Clamp();

	void Scale(const Image &source, const T scale);

	void Scale(const Image &source, const T scale, const bool alt_scale);

	void CopyIntoImage(Image &target, bool maximize = false, bool retarget_alpha = false) const;

	void IncreaseComponents();
public:
	void resize(const FloatImage<T> &source);

	void resize(unsigned int width, unsigned int height, unsigned int componenets);

	void resizeExtra(unsigned int width, unsigned int height, unsigned int componenets);

	void SubtractImage(const FloatImage<T> &a, const FloatImage<T> &b);

	void AddImage(const FloatImage<T> &a, const FloatImage<T> &b);

	void MultiplyImage(const FloatImage<T> &a, const FloatImage<T> &b);

	void DivideImage(const FloatImage<T> &a, const FloatImage<T> &b);

	void AddImageMasked(const FloatImage<T> &a, const FloatImage<T> &b, const FloatImage<T> &mask);

	void MaskImage(const FloatImage<T> &a, const FloatImage<T> &b);

	T GetError(unsigned int c);

	void AddLab(T L, T a, T b);

	void Add(T f);

	void AttenuateImage(T L, T a, T b);

	void ConvertLab2Rgb(const FloatImage<T> &Lab);

	void ConvertRgb2Lab(const FloatImage<T> &RGB);

	void FillDiffusion(const FloatImage<T> &source, FloatImage<T> &mask);

	void AnisotropicDiffusion(const FloatImage<T>& source, FloatImage<T>& mask);

	void GaussianFilter(const FloatImage<T>& source, FloatImage<T>& scratch, T scale, T dx, T dy, BoundaryCondition boundary, bool add = false);

	void GaussianSplat(const FloatImage<T> &source, FloatImage<T> &scratch, T scale, T dx, T dy, BoundaryCondition boundary, bool add = false);

	void GaussianWeightedFilter(FloatImage<T> &msk, const FloatImage<T> &source, FloatImage<T> &scratch, const FloatImage<T> &mask, T scale, T dx, T dy, BoundaryCondition boundary, bool add = false);

public:
	void GaussianFilterSTX(int target, const FloatImage<T>& source, int component, T scale, T dx, BoundaryCondition boundary, bool add);

	void GaussianFilterSTY(int target, const FloatImage<T>& source, int component, T scale, T dy, BoundaryCondition boundary, bool add);

	void GaussianSplatSTX(int target, const FloatImage<T>& source, int component, T scale, T dx, BoundaryCondition boundary, bool add);

	void GaussianSplatSTY(int target, const FloatImage<T>& source, int component, T scale, T dy, BoundaryCondition boundary, bool add);

	void CopyComponent(int target, const FloatImage<T>& source, int component);

private:
	void GaussianFilterSTX(T *target, T *source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

	void GaussianFilterSTY(T *target, T *source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

	void GaussianSplatSTX(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

	void GaussianSplatSTY(T *target, T *source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);
};
