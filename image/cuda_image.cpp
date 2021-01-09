#include "cuda_image.h"
#include "../kernel/kernel_cu.h"
#include "../kernel/filter.h"

template <typename T> void CudaImage<T>::dealloc()
{
	if (m_mapped) return;
	freeDeviceData(m_data);
	m_data.resize(0);
	m_pitch.resize(0);
}

template <typename T> void CudaImage<T>::alloc()
{
	m_pitch.resize(m_components);
	m_data.resize(m_components);
	for (unsigned int c = 0; c < m_components; c++)
	{
		allocDeviceData<T>(m_data[c], m_pitch[c], m_width, m_height, __FILE__, __LINE__);
	}
}

template <typename T> void CudaImage<T>::resize(const CudaImage<T> &source)
{
	if (!check(source))
	{
		if (m_mapped)
		{
			std::cout << "Trying to resize mapped linear image." << std::endl;
			exit(-1);
		}

		SetDimensions(source.m_width, source.m_height, source.m_components);

		m_data.resize(m_components);
		m_pitch.resize(m_components);
		for (unsigned int c = 0; c < m_components; c++)
			allocDeviceData<T>(m_data[c], m_pitch[c], m_width, m_height, __FILE__, __LINE__);
	}
}

template <typename T> void CudaImage<T>::resize(unsigned int width, unsigned int height, unsigned int components)
{
	if (!((m_width == width) && (m_height == height) && (m_components == components)))
	{
		if (m_mapped)
		{
			std::cout << "Trying to resize mapped linear image." << std::endl;
			exit(-1);
		}

		SetDimensions(width, height, components);

		m_data.resize(m_components);
		m_pitch.resize(m_components);
		for (unsigned int c = 0; c < m_components; c++)
			allocDeviceData<T>(m_data[c], m_pitch[c], m_width, m_height, __FILE__, __LINE__);
	}
}

template <typename T> bool CudaImage<T>::check(const CudaImage<T> &source)
{
	return ((m_width == source.m_width) && (m_height == source.m_height) && (m_components == source.m_components));
}

template <typename T> void CudaImage<T>::SetDimensions(unsigned int width, unsigned int height, unsigned int components)
{
	if ((m_width == width) && (m_height == height) && (m_components == components)) return;
	if (m_mapped)
	{
		std::cout << "Trying to resize mapped linear image." << std::endl;
		exit(-1);
	}
	dealloc();
	m_width = width;
	m_height = height;
	m_components = components;
}

template <typename T> void CudaImage<T>::Upload(FloatImage<T> &source)
{
	unsigned int components = source.GetComponents();
	resize(source.GetWidth(), source.GetHeight(), source.GetComponents());

	m_min = source.GetMin();
	m_max = source.GetMax();

	if (m_mapped)
	{
		for (unsigned int c = 0; c < m_components; c++)
		{
			uploadData<T>(m_data[c], source.GetArray(c), m_width * m_height);
		}
	}
	else
	{
		for (unsigned int c = 0; c < components; c++)
		{
			upload2DData<T>(m_data[c], m_pitch[c], source.GetArray(c), m_width, m_height);
		}
	}

	CHECK_LAUNCH_ERROR();
}

template <typename T> void CudaImage<T>::Download(FloatImage<T> &dest) const
{
	unsigned int components = m_components;

	dest.resize(m_width, m_height, m_components);
	dest.SetMin(m_min);
	dest.SetMax(m_max);

	if (m_mapped)
	{
		for (unsigned int c = 0; c < m_components; c++)
		{
			downloadData<T>(dest.GetArray(c), m_data[c], m_width * m_height);
		}
	}
	else
	{
		for (unsigned int c = 0; c < m_components; c++)
		{
			download2DData<T>(dest.GetArray(c), m_data[c], m_pitch[c], m_width, m_height);
		}
	}


	CHECK_LAUNCH_ERROR();
}

// calculate this = a - b
template <typename T> void CudaImage<T>::SubtractImage(const CudaImage<T> &a, const CudaImage<T> &b)
{
	resize(a);
	if (!check(b)) 
		exit(-1);

	for (unsigned int i = 0; i < m_data.size(); i++)
	{
		subtract<T>(m_data[i], a.m_data[i], b.m_data[i], m_width, m_height, m_pitch[i]);
	}

	CHECK_LAUNCH_ERROR();
}

// calculate this = a + b
template <typename T> void CudaImage<T>::AddImage(const CudaImage<T> &a, const CudaImage<T> &b)
{
	resize(a);
	if (!check(b)) 
		exit(-1);

	for (unsigned int i = 0; i < m_data.size(); i++)
	{
		add<T>(m_data[i], a.m_data[i], b.m_data[i], m_width, m_height, m_pitch[i]);
	}

	CHECK_LAUNCH_ERROR();
}

// calculate this = f * this
template <typename T> void CudaImage<T>::Attenuate(T f)
{
	for (unsigned int i = 0; i < m_data.size(); i++)
	{
		if (f != 1.0f) multiply<T>(m_data[i], f, m_width, m_height, m_pitch[i]);
	}

	CHECK_LAUNCH_ERROR();
}

template <typename T> void CudaImage<T>::AttenuateImage(T L, T a, T b)
{
	if (L != 1.0f) multiply<T>(m_data[0], L, m_width, m_height, m_pitch[0]);
	if (m_components >= 3)
	{
		if (a != 1.0f) multiply<T>(m_data[1], a, m_width, m_height, m_pitch[1]);
		if (b != 1.0f) multiply<T>(m_data[2], b, m_width, m_height, m_pitch[2]);
	}
	CHECK_LAUNCH_ERROR();
}

template <typename T> void CudaImage<T>::Clear()
{
	for (unsigned int i = 0; i < m_data.size(); i++)
	{
		cudaMemset2D(m_data[i], m_pitch[i], 0, m_width * sizeof(T), m_height);
	}

	CHECK_LAUNCH_ERROR();
}

template <typename T> void CudaImage<T>::Copy(const CudaImage<T> &source)
{
	resize(source);

	m_min = source.m_min;
	m_max = source.m_max;

	for (unsigned int i = 0; i < m_data.size(); i++)
	{
		cudaMemcpy2D(m_data[i], m_pitch[i], source.m_data[i], source.m_pitch[i], m_width * sizeof(T), m_height, cudaMemcpyDeviceToDevice);
	}

	CHECK_LAUNCH_ERROR();
}

template <typename T> void CudaImage<T>::ConvertLab2Rgb(const CudaImage<T> &Lab)
{
	resize(Lab);
	m_min = T(0);
	m_max = T(1);

	if (m_components >= 3)
		Lab2rgb<T>(Lab.m_data[0], Lab.m_data[1], Lab.m_data[2], m_data[0], m_data[1], m_data[2], m_width, m_height, m_pitch[0]);
	else 
		Lab2rgb<T>(Lab.m_data[0], m_data[0], m_width, m_height, m_pitch[0]);
	if (m_components == 2)
	{
		cudaMemcpy2D(m_data[1], m_pitch[1], Lab.m_data[1], Lab.m_pitch[1], m_width * sizeof(T), m_height, cudaMemcpyDeviceToDevice);
	}
	else if (m_components == 4)
	{
		cudaMemcpy2D(m_data[3], m_pitch[3], Lab.m_data[3], Lab.m_pitch[3], m_width * sizeof(T), m_height, cudaMemcpyDeviceToDevice);
	}
}

template <typename T> void CudaImage<T>::ConvertRgb2Lab(const CudaImage<T> &RGB)
{
	resize(RGB);
	// L
	m_min = T(0);
	m_max = T(100);
	if (m_components >= 3)
		rgb2Lab<T>(RGB.m_data[0], RGB.m_data[1], RGB.m_data[2], m_data[0], m_data[1], m_data[2], m_width, m_height, m_pitch[0]);
	else
		rgb2Lab<T>(RGB.m_data[0], m_data[0], m_width, m_height, m_pitch[0]);
	if (m_components == 2)
	{
		cudaMemcpy2D(m_data[1], m_pitch[1], RGB.m_data[1], RGB.m_pitch[1], m_width * sizeof(T), m_height, cudaMemcpyDeviceToDevice);
	}
	else if (m_components == 4)
	{
		cudaMemcpy2D(m_data[3], m_pitch[3], RGB.m_data[3], RGB.m_pitch[3], m_width * sizeof(T), m_height, cudaMemcpyDeviceToDevice);
	}
}

template <typename T> void CudaImage<T>::AddLab(T L, T a, T b)
{
	if (L != 0.0f) add<T>(m_data[0], L, m_width, m_height, m_pitch[0]);
	if (m_components >= 3)
	{
		if (a != 0.0f) add<T>(m_data[1], a, m_width, m_height, m_pitch[1]);
		if (b != 0.0f) add<T>(m_data[2], b, m_width, m_height, m_pitch[2]);
	}
}

template <typename T> void CudaImage<T>::Add(T f)
{
	if (f == 0.0f) return;
	for (unsigned int c = 0; c < m_components; c++)
	{
		add<T>(m_data[c], f, m_width, m_height, m_pitch[c]);
	}
}

template <typename T> void CudaImage<T>::Gamma(T gamma)
{
	for (unsigned int c = 0; c < m_components; c++)
	{
		Gamma_GPU(GetArray(c),m_width,m_height,gamma);
	}
	// std::cout << "Function not implemented in " << __FILE__ << ":" << __LINE__ << std::endl;
	// exit(-1);
}

template <typename T> void CudaImage<T>::ScaleFast(const CudaImage<T>& source)
{
	for (unsigned int c = 0; c < m_components; c++)
	{
		ScaleFast_GPU(GetArray(c),source.m_data[c],m_width,m_height,source.GetWidth(),source.GetHeight());
	}
	// std::cout << "Function not implemented in " << __FILE__ << ":" << __LINE__ << std::endl;
	// exit(-1);
}

template <typename T> void CudaImage<T>::GaussianFilter(const CudaImage<T>& source, CudaImage<T>& scratch, T scale, T dx, T dy, BoundaryCondition boundary, bool add)
{
	resize(source);
	scratch.resize(source);
	for (unsigned int i = 0; i < m_data.size(); i++)
	{
		GaussianFilterSTY(scratch.m_data[i], source.m_data[i], m_width, m_height, scale, dy, boundary, false);
		GaussianFilterSTX(m_data[i], scratch.m_data[i], m_width, m_height, scale, dx, boundary, add);
	}
}

template <typename T> void CudaImage<T>::GaussianSplat(const CudaImage<T>& source, CudaImage<T>& scratch, T scale, T dx, T dy, BoundaryCondition boundary, bool add)
{
	resize(source);
	scratch.resize(source);
	if ((boundary == BoundaryCondition::Repeat) || (boundary == BoundaryCondition::Zero))
	{
		for (unsigned int i = 0; i < m_data.size(); i++)
		{
			GaussianFilterSTX(scratch.m_data[i], source.m_data[i], m_width, m_height, scale, -dx, boundary, false);
			GaussianFilterSTY(m_data[i], scratch.m_data[i], m_width, m_height, scale, -dy, boundary, add);
		}
	}
	else
	{
		for (unsigned int i = 0; i < m_data.size(); i++)
		{
			GaussianSplatSTX(scratch.m_data[i], source.m_data[i], m_width, m_height, scale, dx, boundary, false);
			GaussianSplatSTY(m_data[i], scratch.m_data[i], m_width, m_height, scale, dy, boundary, add);
		}
	}
}

template <typename T> void CudaImage<T>::GaussianWeightedFilter(CudaImage<T>& msk, const CudaImage<T>& source, CudaImage<T>& scratch, const CudaImage<T>& mask, T scale, T dx, T dy, BoundaryCondition boundary, bool add)
{
	resize(source);
	msk.resize(mask);
	scratch.resize(source);
	GaussianFilterSTY(scratch.m_data[0], mask.m_data[0], m_width, m_height, scale, dy, boundary, false);
	GaussianFilterSTX(msk.m_data[0], scratch.m_data[0], m_width, m_height, scale, dx, boundary, add);
	for (unsigned int i = 0; i < m_data.size(); i++)
	{
		GaussianFilterSTY(scratch.m_data[i], source.m_data[i], m_width, m_height, scale, dy, boundary, false);
		GaussianFilterSTX(m_data[i], scratch.m_data[i], m_width, m_height, scale, dx, boundary, add);
	}
}

template <typename T> void CudaImage<T>::GaussianFilterSTX(int target, const CudaImage<T>& source, int component, T scale, T dx, BoundaryCondition boundary, bool add)
{
	GaussianFilterSTX(m_data[target], source.m_data[component], m_width, m_height, scale, dx, boundary, add);
}

template <typename T> void CudaImage<T>::GaussianFilterSTY(int target, const CudaImage<T>& source, int component, T scale, T dy, BoundaryCondition boundary, bool add)
{
	GaussianFilterSTY(m_data[target], source.m_data[component], m_width, m_height, scale, dy, boundary, add);
}

template <typename T> void CudaImage<T>::GaussianSplatSTX(int target, const CudaImage<T>& source, int component, T scale, T dx, BoundaryCondition boundary, bool add)
{
	GaussianSplatSTX(m_data[target], source.GetArrayConst(component), m_width, m_height, scale, dx, boundary, add);
}

template <typename T> void CudaImage<T>::GaussianSplatSTY(int target, const CudaImage<T>& source, int component, T scale, T dy, BoundaryCondition boundary, bool add)
{
	GaussianSplatSTY(m_data[target],source.GetArrayConst(component), m_width, m_height, scale, dy, boundary, add);
}

template <typename T> void CudaImage<T>::CopyComponent(int target, const CudaImage<T>& source, int component)
{
	std::copy(source.m_data[component], source.m_data[component] + (size_t)m_width * m_height, m_data[target]);
}

template <typename T> void CudaImage<T>::GaussianFilterSTX(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add)
{
	GaussianFilterSTX_GPU(target, source, width, height, scale, d, boundary, add);
}

template <typename T> void CudaImage<T>::GaussianFilterSTY(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add)
{
	GaussianFilterSTY_GPU(target, source, width, height, scale, d, boundary, add);
}

template <typename T> void CudaImage<T>::GaussianSplatSTX(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add)
{
	GaussianSplatSTX_GPU(target, source, width, height, scale, d, boundary, add);
	// std::cout << "Function not implemented in " << __FILE__ << ":" << __LINE__ << std::endl;
	
	// exit(-1);
}

template <typename T> void CudaImage<T>::GaussianSplatSTY(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add)
{
	GaussianSplatSTY_GPU(target, source, width, height, scale, d, boundary, add);
	// std::cout << "Function not implemented in " << __FILE__ << ":" << __LINE__ << std::endl;
	
	// exit(-1);
}

template class CudaImage<float>;
template class CudaImage<double>;
