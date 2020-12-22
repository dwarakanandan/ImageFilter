#include "float_image.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <zlib.h>
#include "../kernel/kernel_cu.h"
#include <locale>
#include "png_image.h"
#include "jpeg_image.h"
#include <omp.h>

#define CENTERED_DX

#if defined(__GNUC__)
#define __forceinline \
		__inline__ __attribute__((always_inline))
#endif

const char *FloatImageCode = "flt0";
const char* DoubleImageCode = "flt1";

template <typename T> bool FloatImage<T>::ReadImage(const char fileName[], bool raw)
{
	std::locale loc;
	std::string name = fileName; name = name.substr(name.length() - 3);
	for (unsigned int i = 0; i < name.length(); ++i) name[i] = std::tolower(name[i], loc);
	if (name.compare("png") == 0)
	{
		PngImage image;
		if (!image.ReadImage(fileName)) return false;
		Copy(image);
		if (!raw) ConvertRgb2Lab(*this);
		return true;
	}
	else if ((name.compare("jpg") == 0) || (name.compare("jpeg") == 0))
	{
		JpegImage image;
		if (!image.ReadImage(fileName)) return false;
		Copy(image);
		if (!raw) ConvertRgb2Lab(*this);
		return true;
	}
	else
	{
		logger << "Reading image \"" << fileName << "\"" << std::endl;

		gzFile in = gzopen(fileName, "rb");

		if (in == NULL) return false;

		char magic[4];

		gzread(in, magic, 4 * sizeof(FloatImageCode[0]));
		bool is_float = true;
		bool is_double = true;
		for (int i = 0; i < 4; i++)
		{
			if (magic[i] != FloatImageCode[i])
			{
				is_float = false;
			}
			if (magic[i] != DoubleImageCode[i])
			{
				is_double = false;
			}
		}
		if ((!is_float) && (!is_double))
		{
			gzclose(in);
			return false;
		}

		gzread(in, &m_width, sizeof(m_width));
		gzread(in, &m_height, sizeof(m_height));
		gzread(in, &m_components, sizeof(m_components));
		if (((is_float) && (sizeof(T) == sizeof(float))) || ((is_double) && (sizeof(T) == sizeof(double))))
		{
			gzread(in, &m_min, sizeof(T));
			gzread(in, &m_max, sizeof(T));
		}
		else if (is_float)
		{
			float t_min, t_max;
			gzread(in, &t_min, sizeof(float));
			gzread(in, &t_max, sizeof(float));
			m_min = (T)t_min;
			m_max = (T)t_max;
		}
		else
		{
			double t_min, t_max;
			gzread(in, &t_min, sizeof(float));
			gzread(in, &t_max, sizeof(float));
			m_min = (T)t_min;
			m_max = (T)t_max;
		}
		alloc();
		if (((is_float) && (sizeof(T) == sizeof(float))) || ((is_double) && (sizeof(T) == sizeof(double))))
		{
			for (int i = 0; i < (int)m_components; i++)
			{
				if ((unsigned long long)gzread(in, m_data[i], m_width * m_height * sizeof(T)) != (size_t)m_width * m_height * sizeof(T)) return false;
			}
		}
		else if (is_float)
			{
				float* tmp;
				allocHostData(tmp, (size_t)m_width * m_height, __FILE__, __LINE__);
				for (int i = 0; i < (int)m_components; i++)
				{
					if ((unsigned long long)gzread(in, tmp, m_width * m_height * sizeof(float)) != (size_t)m_width * m_height * sizeof(float)) return false;
					T* ttmp = m_data[i];
					for (unsigned int xy = 0; xy < m_width * m_height; xy++) ttmp[xy] = (T)tmp[xy];
				}
				freeHostData(tmp);
			}
		else
		{
			double* tmp;
			allocHostData(tmp, (size_t)m_width * m_height, __FILE__, __LINE__);
			for (int i = 0; i < (int)m_components; i++)
			{
				if ((unsigned long long)gzread(in, tmp, m_width * m_height * sizeof(double)) != (size_t)m_width * m_height * sizeof(double)) return false;
				T* ttmp = m_data[i];
				for (unsigned int xy = 0; xy < m_width * m_height; xy++) ttmp[xy] = (T)tmp[xy];
			}
			freeHostData(tmp);
		}

		gzclose(in);
		return true;
	}
}

template <typename T> bool FloatImage<T>::WriteImage(const char fileName[])
{
	logger << "Writing image \"" << fileName << "\"" << std::endl;

	gzFile out = gzopen(fileName, "wb9");

	if (out == NULL) return false;

	if (sizeof(T) == sizeof(float))
	{
		gzwrite(out, FloatImageCode, 4 * sizeof(FloatImageCode[0]));
	}
	else
	{
		gzwrite(out, DoubleImageCode, 4 * sizeof(FloatImageCode[0]));
	}
	gzwrite(out, &m_width, sizeof(m_width));
	gzwrite(out, &m_height, sizeof(m_height));
	gzwrite(out, &m_components, sizeof(m_components));
	gzwrite(out, &m_min, sizeof(m_min));
	gzwrite(out, &m_max, sizeof(m_max));
	for (int i = 0; i < (int)m_components; i++) gzwrite(out, m_data[i], m_width * m_height * sizeof(m_data[0][0]));
	
	gzclose(out);
	return true;
}

template <typename T> void FloatImage<T>::Copy(const FloatImage<T> &source)
{
	resize(source);
	SetMin(source.GetMin());
	SetMax(source.GetMax());
	for (unsigned int c = 0; c < m_components; c++)
	{
		memcpy(GetArray(c), source.GetArrayConst(c), (size_t)m_width * m_height * sizeof(T));
	}
}

template <typename T> void FloatImage<T>::CopyReplicate(const FloatImage<T>&  source, unsigned int components)
{
	resize(source.GetWidth(), source.GetHeight(), components);
	SetMin(source.GetMin());
	SetMax(source.GetMax());
	for (unsigned int c = 0; c < m_components; c++)
	{
		memcpy(GetArray(c), source.GetArrayConst(c % source.GetComponents()), (size_t)m_width * m_height * sizeof(T));
	}
}

template <typename T> void FloatImage<T>::Attenuate(const T f)
{
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int x = 0; x < (int)(m_width * m_height); x++)
		{
			m_data[c][x] = m_data[c][x] * f;
		}
	}
}

template <typename T> void FloatImage<T>::Attenuate(int c, const T f)
{
	for (int x = 0; x < (int)(m_width * m_height); x++)
	{
		m_data[c][x] = m_data[c][x] * f;
	}
}

template <typename T> void FloatImage<T>::AttenuateAdd(int c_dst, int c_src, const T f)
{
	for (int x = 0; x < (int)(m_width * m_height); x++)
	{
		m_data[c_dst][x] += m_data[c_src][x] * f;
	}
}

template <typename T> void FloatImage<T>::Add(int c_dst, int c_src)
{
	for (int x = 0; x < (int)(m_width * m_height); x++)
	{
		m_data[c_dst][x] += m_data[c_src][x];
	}
}

template <typename T> void FloatImage<T>::AddSquared(int c_dst, int c_src)
{
	for (int x = 0; x < (int)(m_width * m_height); x++)
	{
		m_data[c_dst][x] += m_data[c_src][x] * m_data[c_src][x];
	}
}

template <typename T> void FloatImage<T>::Square()
{
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int x = 0; x < (int)(m_width * m_height); x++)
		{
			m_data[c][x] *= m_data[c][x];
		}
	}
}

template <typename T> void FloatImage<T>::SquareRoot()
{
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int x = 0; x < (int)(m_width * m_height); x++)
		{
			m_data[c][x] = sqrt(m_data[c][x]);
		}
	}
}

template <typename T> void FloatImage<T>::Square(int c)
{
	for (int x = 0; x < (int)(m_width * m_height); x++)
	{
		m_data[c][x] *= m_data[c][x];
	}
}

template <typename T> void FloatImage<T>::SquareRoot(int c)
{
	for (int x = 0; x < (int)(m_width * m_height); x++)
	{
		m_data[c][x] = sqrt(m_data[c][x]);
	}
}

template <typename T> void FloatImage<T>::AddImage(int c_dst, const FloatImage<T> &source, int c_src)
{
	for (int x = 0; x < (int)(m_width * m_height); x++)
	{
		m_data[c_dst][x] += source.m_data[c_src][x];
	}
}

template <typename T> void FloatImage<T>::Attenuate(const FloatImage<T> &source, const T f)
{
	resize(source);
	SetMin(source.GetMin());
	SetMax(source.GetMax());
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int x = 0; x < (int)(m_width * m_height); x++)
		{
			m_data[c][x] = source.m_data[c][x] * f;
		}
	}
}

template <typename T> void FloatImage<T>::Lerp(const FloatImage<T> &source, const T f)
{
	check(source);
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int x = 0; x < (int)(m_width * m_height); x++)
		{
			m_data[c][x] += (source.m_data[c][x] - m_data[c][x]) * f;
		}
	}
}

template <typename T> void FloatImage<T>::Copy(const Image &source)
{
	SetWidth(source.GetWidth());
	SetHeight(source.GetHeight());
	unsigned int comp = source.GetComponents();
	if (comp > 3) comp = 3;
	if (comp < 3) comp = 1;
	SetComponents(comp);
	SetMin(0.0f);
	SetMax(1.0f);
	T scale = (T)source.GetMask();
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				GetArray(c)[x + y * m_width] = (T)source.VGetValue(x, y, c) / scale;
			}
		}
	}
}

template <typename T> void FloatImage<T>::CopyAlpha(const Image &source)
{
	// needs to be inverted
	SetWidth(source.GetWidth());
	SetHeight(source.GetHeight());
	unsigned int comp = source.GetComponents();
	SetComponents(1);
	SetMin(0.0f);
	SetMax(1.0f);
	T scale = (T)source.GetMask();
	for (unsigned int y = 0; y < m_height; y++)
	{
		for (unsigned int x = 0; x < m_width; x++)
		{
			GetArray(0)[x + y * m_width] = (T)source.VGetValue(x, y, comp - 1) / scale;
		}
	}
}

template <typename T> void FloatImage<T>::Maximize(const char *name, T min, T max)
{
	bool is_lab;
	for (unsigned int c = 0; c < m_components; c++)
	{
		T total_min, total_max;
		if (min == max)
		{
			total_min = total_max = GetValue(0, 0, c);
			for (unsigned int y = 0; y < m_height; y++)
			{
				for (unsigned int x = 0; x < m_width; x++)
				{
					T v = GetValue(x, y, c);
					total_min = std::min(v, total_min);
					total_max = std::max(v, total_max);
				}
			}
		}
		else
		{
			total_min = min;
			total_max = max;
		}
		if (name != NULL) logger << "stretching contrast for \"" << name << "\" [" << total_min << "," << total_max << "] ->[0.0,1.0]" << std::endl;
		if (total_min == total_max)
		{
			total_min = 0.0f;
			total_max = 1.0f;
		}
		// check if this is an LAB image
		if (c == 0)
		{
			if ((total_min >= 0.0f) && (total_max <= 100.0f) && (total_max >= 25.0f)) is_lab = true;
			else
			{
				// difference between lab
				if ((total_min <= -12.5f) && (total_max >= 12.5f)) is_lab = true;
				else is_lab = false;
			}
		}
		if (is_lab)
		{
			if (c == 0)
			{
				if (total_min >= 0.0f)
				{
					total_min = 0.0f;
					total_max = 100.0f;
				}
				else
				{
					total_min = -50.0f;
					total_max = 50.0f;
				}
			}
			else
			{
				// equal color distance
				total_min = -50.0f;
				total_max = 50.0f;
			}
		}
		if (!is_lab)
		{
			if (-total_min < 0.25f * total_max)
			{
				// uneven distribution, leave alone?
			}
			else
			{
				// even distribution
				total_max = std::max(-total_min, -total_max);
				total_min = -total_max;
			}
		}
		T scale = total_max - total_min;
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				GetArray(c)[x + y * m_width] = (GetArray(c)[x + y * m_width] - total_min) / scale;
			}
		}
	}
	m_min = 0.0f;
	m_max = 1.0f;
}

template <typename T> void FloatImage<T>::Maximize(T *min, T *max)
{
	for (unsigned int c = 0; c < m_components; c++)
	{
		T total_min, total_max;
		total_min = min[c];
		total_max = max[c];
		if (total_min == total_max)
		{
			total_min = 0.0f;
			total_max = 1.0f;
		}
		T scale = total_max - total_min;
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				GetArray(c)[x + y * m_width] = (GetArray(c)[x + y * m_width] - total_min) / scale;
			}
		}
	}
	m_min = 0.0f;
	m_max = 1.0f;
}

template <typename T> void FloatImage<T>::Gamma(T gamma)
{
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				T v = GetArray(c)[x + y * m_width];
				if (v > 0.0f) GetArray(c)[x + y * m_width] = pow(v, T(1)/gamma);
			}
		}
	}
}

template <typename T> void FloatImage<T>::Clamp()
{
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				T v = GetArray(c)[x + y * m_width];
				GetArray(c)[x + y * m_width] = std::max(T(0), v);
			}
		}
	}
}

template <typename T> void FloatImage<T>::Scale(const Image &source, const T in_scale)
{
	Scale(source, in_scale, false);
}

template <typename T> void FloatImage<T>::Scale(const Image &source, const T in_scale, const bool alt_scale)
{
	if (in_scale == 1.0f) return Copy(source);
	T scale = 1.0f / in_scale;
	SetWidth((unsigned int) floor(0.5f + source.GetWidth() / scale));
	SetHeight((unsigned int) floor(0.5f + source.GetHeight() / scale));
	T flt_scale = std::max(T(1), scale);
	int sc;
	if (alt_scale)
		sc = (int)floor(0.5 * flt_scale);
	else
		sc = (int)ceil(3.0 * flt_scale);
	unsigned int comp = source.GetComponents();
	if (comp > 3) comp = 3;
	if (comp < 3) comp = 1;
	SetComponents(comp);
	SetMin(0.0f);
	SetMax(1.0f);
	T range = (T)source.GetMask();
	std::cout << "Scaling image with scale " << in_scale << " (" << source.GetWidth() << "x" << source.GetHeight() << ") -> (" << m_width << "x" << m_height << ")" << std::endl;
	if (alt_scale)
	{
		for (int y = 0; y < (int)m_height; y++)
		{
			int ys = (int)floor(y * scale + 0.5f);
			for (unsigned int x = 0; x < m_width; x++)
			{
				int xs = (int)floor(x * scale + 0.5f);
				bool label_found = false;
				T s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
				for (int y0 = std::max(0, ys - sc); y0 < std::min((int)source.GetHeight(), ys + sc + 1); y0++)
				{
					for (int x0 = std::max(0, xs - sc); x0 < std::min((int)source.GetWidth(), xs + sc + 1); x0++)
					{
						if (label_found == false)
						{
							for (unsigned int c = 0; c < m_components; c++)
							{
								s[c] = (T)source.VGetValue(x0, y0, c) / range;
								label_found |= (s[c] > 0.0f);
							}
						}
						else
						{
							bool label_valid = true;
							for (unsigned int c = 0; c < m_components; c++)
							{
								label_valid &= (s[c] == (T)source.VGetValue(x0, y0, c) / range);
							}
							if (!label_valid)
							{
								for (unsigned int c = 0; c < m_components; c++) s[c] = 0.0f;
								x0 = (int)source.GetWidth();
								y0 = (int)source.GetHeight();
							}
						}
					}
				}
				for (unsigned int c = 0; c < m_components; c++) GetArray(c)[x + y * m_width] = s[c];
			}
		}
	}
	else
	{
		for (unsigned int c = 0; c < m_components; c++)
		{
			for (int y = 0; y < (int)m_height; y++)
			{
				int ys = (int)floor(y * scale + 0.5f);
				for (unsigned int x = 0; x < m_width; x++)
				{
					int xs = (int)floor(x * scale + 0.5f);
					T s = 0.0f;
					T w = 0.0f;
					for (int y0 = std::max(0, ys - sc); y0 < std::min((int)source.GetHeight(), ys + sc + 1); y0++)
					{
						T dy = (y0 - y * scale) / flt_scale;
						T ry = 3.1415926535897932384626433832795f * dy;
						T wy;
						if (dy == 0.0f)
						{
							wy = 1.0f;
						}
						else if (fabs(dy) < T(3))
						{
							wy = sin(ry) / ry * sin(ry / T(3)) / (ry / T(3));
						}
						else
						{
							wy = 0.0f;
						}
						if (wy != 0.0f)
						{
							for (int x0 = std::max(0, xs - sc); x0 < std::min((int)source.GetWidth(), xs + sc + 1); x0++)
							{
								T dx = (x0 - x * scale) / flt_scale;
								T rx = 3.1415926535897932384626433832795f * dx;
								T wx;
								if (dx == 0.0f)
								{
									wx = 1.0f;
								}
								else if (fabs(dx) < T(3))
								{
									wx = sin(rx) / rx * sin(rx / T(3)) / (rx / T(3));
								}
								else
								{
									wx = 0.0f;
								}
								if (wx != 0.0f)
								{
									s += wx * wy * (T)source.VGetValue(x0, y0, c) / range;
									w += wx * wy;
								}
							}
						}
					}
					if (w == 0.0f)
					{
						s = w = 1.0f;
					}

					GetArray(c)[x + y * m_width] = s / w;
				}
			}
		}
	}
}

template <typename T> void FloatImage<T>::Scale(const FloatImage<T> &source, const T in_scale)
{
	T scale = 1.0f / in_scale;
	SetWidth((unsigned int)floor(0.5f + source.GetWidth() / scale));
	SetHeight((unsigned int)floor(0.5f + source.GetHeight() / scale));
	T flt_scale = std::min(T(1), scale);
	int sc = (int)ceil(3.0f * scale);
	unsigned int comp = source.GetComponents();
	if (comp > 3) comp = 3;
	if (comp < 3) comp = 1;
	SetComponents(comp);
	SetMin(0.0f);
	SetMax(1.0f);
	std::cout << "Scaling image with scale " << scale << " (" << source.GetWidth() << "x" << source.GetHeight() << ") -> (" << m_width << "x" << m_height << ")" << std::endl;
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			int ys = (int)floor(y * scale + 0.5f);
			for (unsigned int x = 0; x < m_width; x++)
			{
				int xs = (int)floor(x * scale + 0.5f);
				T s = 0.0f;
				T w = 0.0f;
				for (int y0 = std::max(0, ys - sc); y0 < std::min((int)source.GetHeight(), ys + sc + 1); y0++)
				{
					T dy = (y0 - y * scale) / flt_scale;
					T ry = 1.57079632679489661923f * dy;
					T wy;
					if (dy == 0.0f)
					{
						wy = 1.0f;
					}
					else if (fabs(dy) < T(3))
					{
						wy = sin(ry) / ry * sin(ry / T(3)) / (ry / T(3));
					}
					else
					{
						wy = 0.0f;
					}
					if (wy > 0.0f)
					{
						for (int x0 = std::max(0, xs - sc); x0 < std::min((int)source.GetWidth(), xs + sc + 1); x0++)
						{
							T dx = (x0 - x * scale) / flt_scale;
							T rx = 1.57079632679489661923f * dx;
							T wx;
							if (dx == 0.0f)
							{
								wx = 1.0f;
							}
							else if (fabs(dx) < T(3))
							{
								wx = sin(rx) / rx * sin(rx / T(3)) / (rx / T(3));
							}
							else
							{
								wx = 0.0f;
							}
							if (wx > 0.0f)
							{
								s += wx * wy * source.GetValue(x0, y0, c);
								w += wx * wy;
							}
						}
					}
				}
				if (w == 0.0f)
				{
					s = w = 1.0f;
				}
				GetArray(c)[x + y * m_width] = s / w;
			}
		}
	}
}

template <typename T> void FloatImage<T>::Scale(const FloatImage<T> &source)
{
	T scale_x = (T)(source.GetWidth()) / (T)(m_width);
	T scale_y = (T)(source.GetHeight()) / (T)(m_height);
	T flt_scale_x = std::min(T(1), scale_x);
	T flt_scale_y = std::min(T(1), scale_y);
	int sc_x = (int)ceil(T(3) * scale_x);
	int sc_y = (int)ceil(T(3) * scale_y);
	unsigned int comp = source.GetComponents();
	//if (comp > 3) comp = 3;
	//if (comp < 3) comp = 1;
	SetComponents(comp);
	SetMin(0.0f);
	SetMax(1.0f);
	std::cout << "Scaling image with scale " << scale_x << ", " << scale_y << " (" << source.GetWidth() << "x" << source.GetHeight() << ") -> (" << m_width << "x" << m_height << ")" << std::endl;
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			int ys = (int)floor(y * scale_y + 0.5f);
			for (unsigned int x = 0; x < m_width; x++)
			{
				int xs = (int)floor(x * scale_x + 0.5f);
				T s = 0.0f;
				T w = 0.0f;
				for (int y0 = std::max(0, ys - sc_y); y0 < std::min((int)source.GetHeight(), ys + sc_y + 1); y0++)
				{
					T dy = (y0 - y * scale_y) / flt_scale_y;
					T ry = 1.57079632679489661923f * dy;
					T wy;
					if (dy == T(0))
					{
						wy = T(1);
					}
					else if (fabs(dy) < T(3))
					{
						wy = sin(ry) / ry * sin(ry / T(3)) / (ry / T(3));
					}
					else
					{
						wy = 0.0f;
					}
					if (wy > 0.0f)
					{
						for (int x0 = std::max(0, xs - sc_x); x0 < std::min((int)source.GetWidth(), xs + sc_x + 1); x0++)
						{
							T dx = (x0 - x * scale_x) / flt_scale_x;
							T rx = 1.57079632679489661923f * dx;
							T wx;
							if (dx == 0.0f)
							{
								wx = 1.0f;
							}
							else if (fabs(dx) < 3.0f)
							{
								wx = sin(rx) / rx * sin(rx / T(3)) / (rx / T(3));
							}
							else
							{
								wx = 0.0f;
							}
							if (wx > 0.0f)
							{
								s += wx * wy * source.GetValue(x0, y0, c);
								w += wx * wy;
							}
						}
					}
				}
				if (w == 0.0f)
				{
					s = w = 1.0f;
				}
				GetArray(c)[x + y * m_width] = s / w;
			}
		}
	}
}

template <typename T> void FloatImage<T>::ScaleFast(const FloatImage<T> &source)
{
#if 1
	unsigned int source_width = source.GetWidth();
	unsigned int source_height = source.GetHeight();
	std::cout << "Fast scaling image " << source_width << "x" << source_height << " -> " << m_width << "x" << m_height << std::endl;
	if (source.GetWidth() > m_width)
	{
		for (unsigned int c = 0; c < m_components; c++)
		{
			for (unsigned int y = 0; y < m_height; y++)
			{
				unsigned int y0 = std::min(y << 1, source_height - 1);
				unsigned int y1 = std::min(y0 + 1, source_height - 1);
				for (unsigned int x = 0; x < m_width; x++)
				{
					unsigned int x0 = std::min(x << 1, source_width - 1);
					unsigned int x1 = std::min(x0 + 1, source_width - 1);
					T s = 0.25f * (source.GetValue(x0, y0, c) + source.GetValue(x1, y0, c) + source.GetValue(x0, y1, c) + source.GetValue(x1, y1, c));
					SetValue(x, y, c, s);
				}
			}
		}
	}
	else
	{
		for (unsigned int c = 0; c < m_components; c++)
		{
			for (unsigned int y = 0; y < m_height; y++)
			{
				unsigned int y0 = std::min(y >> 1, source_height - 1);
				for (unsigned int x = 0; x < m_width; x++)
				{
					unsigned int x0 = std::min(x >> 1, source_width - 1);
					SetValue(x, y, c, source.GetValue(x0, y0, c));
			}
		}
	}
}
#else
	T scale_x = (T)(source.GetWidth()) / (T)(m_width);
	T scale_y = (T)(source.GetHeight()) / (T)(m_height);

	if (scale_x > 1.0f) scale_x = ceilf(scale_x);
	else if (scale_x < 1.0f) scale_x = 1.0f / ceilf(1.0f / scale_x);
	if (scale_y > 1.0f) scale_y = ceilf(scale_y);
	else if (scale_y < 1.0f) scale_y = 1.0f / ceilf(1.0f / scale_y);

	unsigned int comp = source.GetComponents();
	SetComponents(comp);
	SetMin(0.0f);
	SetMax(1.0f);
	std::cout << "Fast scaling image with scale " << scale_x << ", " << scale_y << " (" << source.GetWidth() << "x" << source.GetHeight() << ") -> (" << m_width << "x" << m_height << ")" << std::endl;
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			T ys = std::max(0.0f, std::min(((T)y + 0.5f) * scale_y - 0.5f, source.GetHeight() - 1.0f));
			int y0 = (int)floorf(ys);
			T yw = ys - y0;
			for (unsigned int x = 0; x < m_width; x++)
			{
				T xs = std::max(0.0f, std::min(((T)x + 0.5f) * scale_x - 0.5f, source.GetWidth() - 1.0f));
				int x0 = (int)floorf(xs);
				T xw = xs - x0;
				T s = source.GetValue(x0, y0, c) * (1.0f - xw) * (1.0f - yw);
				if (x0 + 1 < (int)source.GetWidth()) s += source.GetValue(x0 + 1, y0, c) * xw * (1.0f - yw);
				if (y0 + 1 < (int)source.GetHeight()) s += source.GetValue(x0, y0 + 1, c) * (1.0f - xw) * yw;
				if ((x0 + 1 < (int)source.GetWidth()) && (y0 + 1 < (int)source.GetHeight())) s += source.GetValue(x0 + 1, y0 + 1, c) * xw * yw;
				SetValue(x, y, c, s);
			}
		}
	}
#endif
}

template <typename T> void FloatImage<T>::ScaleMasked(FloatImage<T> &mask_small, const FloatImage<T> &src, const FloatImage<T> &mask, bool fast)
{
	// weighted scaling produces boundary
	if (fast)
	{
		mask_small.ScaleFast(mask);
		ScaleFast(src);
	}
	else
	{
		mask_small.Scale(mask);
		Scale(src);
	}
	T *p_mask = mask_small.GetArray(0);
	for (int x = 0; x < (int)(m_width * m_height); x++) if (p_mask[x] > 0.0f) p_mask[x] = 1.0f;
}

template <typename T> void FloatImage<T>::CopyIntoImage(Image &target, bool maximize, bool retarget_alpha) const
{
	if ((GetComponents() != 4) && (GetComponents() != 2)) retarget_alpha = false;
	target.SetWidth(GetWidth());
	target.SetHeight(GetHeight());
	target.SetComponents(GetComponents());
	int mask = (int) target.GetMask();
	T min, max, avg;
	if (maximize)
	{
		min = max = avg = 0.0f;
		for (unsigned int c = 0; c < m_components; c++)
		{
			for (unsigned int y = 0; y < m_height; y++)
			{
				for (unsigned int x = 0; x < m_width; x++)
				{
					T v = GetArrayConst(c)[x + y * m_width];
					min = std::min(min, v);
					max = std::max(max, v);
					avg += v;
				}
			}
		}
		avg /= (size_t)m_width * m_height * m_components;
		T ext = std::max(avg - min, max - avg);
		min = avg - ext;
		max = avg + ext;
	}
	else
	{
		min = m_min;
		max = m_max;
	}
	T a_sc = 1.0f;
	T g = 1.0f;
	if (retarget_alpha)
	{
		T a_max, a_avg;
		a_max = a_avg = 0.0f;
		unsigned int c = m_components - 1;
		//const T *dat = GetArrayConst(c);
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				T v = GetArrayConst(c)[x + y * m_width];
				a_max = std::max(a_max, v);
				a_avg += v;
			}
		}
		a_avg /= (size_t)m_width * m_height;
		if (a_max == 0.0f)
		{
			g = 1.0f;
			a_sc = 1.0f;
		}
		else
		{
			g = log(0.5f) / log(a_avg / a_max);
			a_sc = 1.0f / a_max;
		}
	}
	T scale = (T) target.GetMask() / (max - min);
	for (unsigned int c = 0; c < (retarget_alpha?(m_components - 1):m_components); c++)
	{
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				int v = std::max(0, std::min((int) floor((GetArrayConst(c)[x + y * m_width] - min) * scale + 0.5f), mask));
				target.VSetValue(x, y, c, (unsigned int) v);
			}
		}
	}
	if (retarget_alpha)
	{
		unsigned int c = m_components - 1;
		for (unsigned int y = 0; y < m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				int v = std::max(0, std::min((int)floor(pow(GetArrayConst(c)[x + y * m_width] * a_sc, g) * (T)target.GetMask() + T(0.5)), mask));
				target.VSetValue(x, y, c, (unsigned int)v);
			}
		}
	}
}

template <typename T> void FloatImage<T>::CalculateMinMax()
{
	m_min = m_max = m_data[0][0];
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (unsigned int i = 0; i < m_width * m_height; i++)
		{
			m_min = std::min(m_min, m_data[c][i]);
			m_max = std::max(m_max, m_data[c][i]);
		}
	}
}

template <typename T> void FloatImage<T>::IncreaseComponents()
{
	m_components++;
	m_data.resize(m_components);

	allocHostData<T>(m_data.back(), (size_t)m_width * m_height, __FILE__, __LINE__);
}

template <typename T> void FloatImage<T>::alloc()
{
	dealloc();
	if (m_width * m_height * m_components == 0) return;
	m_data.resize(m_components);
	for (unsigned int c = 0; c < m_components; c++)
	{
		allocHostData<T>(m_data[c], (size_t)m_width * m_height, __FILE__, __LINE__);
	}
}

template <typename T> void FloatImage<T>::dealloc()
{
	while (!m_data.empty())
	{
		if (m_data.back() != NULL)
		{
			freeHostData(m_data.back());
		}
		m_data.pop_back();
	}
}

template <typename T> void FloatImage<T>::SubtractImage(const FloatImage<T> &a, const FloatImage<T> &b)
{
	resize(a);
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int i = 0; i < (int)(m_width * m_height); i++)
		{
			m_data[c][i] = a.m_data[c][i] - b.m_data[c][i];
		}
	}
}

template <typename T> void FloatImage<T>::AddImage(const FloatImage<T> &a, const FloatImage<T> &b)
{
	resize(a);
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int i = 0; i < (int)(m_width * m_height); i++)
		{
			m_data[c][i] = a.m_data[c][i] + b.m_data[c][i];
		}
	}
}

template <typename T> void FloatImage<T>::MultiplyImage(const FloatImage<T> &a, const FloatImage<T> &b)
{
	resize(a);
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int i = 0; i < (int)(m_width * m_height); i++)
		{
			m_data[c][i] = a.m_data[c][i] * b.m_data[c % b.m_components][i];
		}
	}
}

template <typename T> void FloatImage<T>::DivideImage(const FloatImage<T> &a, const FloatImage<T> &b)
{
	resize(a);
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int i = 0; i < (int)(m_width * m_height); i++)
		{
			if (b.m_data[c % b.m_components][i] == 0.0f) m_data[c][i] = 0.0f;
			else m_data[c][i] = a.m_data[c][i] / b.m_data[c % b.m_components][i];
		}
	}
}

template <typename T> void FloatImage<T>::AddImageMasked(const FloatImage<T> &a, const FloatImage<T> &b, const FloatImage<T> &mask)
{
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int i = 0; i < (int)(m_width * m_height); i++)
		{
			if (mask.m_data[0][i] == 1.0f) m_data[c][i] = a.m_data[c][i];
			else m_data[c][i] = a.m_data[c][i] + b.m_data[c][i];
		}
	}
}

template <typename T> void FloatImage<T>::MaskImage(const FloatImage<T> &a, const FloatImage<T> &b)
{
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int i = 0; i < (int)(m_width * m_height); i++)
		{
			if (b.m_data[c % b.GetComponents()][i] == 0.0f)
				m_data[c][i] = 0.0f;
			else
				m_data[c][i] = a.m_data[c][i];
		}
	}
}

template <typename T> T FloatImage<T>::GetError(unsigned int c)
{
	T var = 0.0f;
	{
		T var_local = 0.0f;
		for (int i = 0; i < (int)(m_width * m_height); i++)
		{
			var_local += m_data[c][i] * m_data[c][i];
		}
		var += var_local;
	}
	return var / (T)((size_t)m_width * m_height);
}

template <typename T> void FloatImage<T>::AddLab(T L, T a, T b)
{
	for (int y = 0; y < (int)m_height; y++)
	{
		for (unsigned int x = 0; x < m_width; x++)
		{
			m_data[0][x + y * m_width] += L;
			m_data[1][x + y * m_width] += a;
			m_data[2][x + y * m_width] += b;
		}
	}
}

template <typename T> void FloatImage<T>::Add(T f)
{
	if (f == 0.0f) return;
	for (unsigned int c = 0; c < m_components; c++)
	{
		for (int y = 0; y < (int)m_height; y++)
		{
			for (unsigned int x = 0; x < m_width; x++)
			{
				m_data[c][x + y * m_width] += f;
			}
		}
	}
}

template <typename T> void FloatImage<T>::AttenuateImage(T L, T a, T b)
{
	for (int y = 0; y < (int)m_height; y++)
	{
		for (unsigned int x = 0; x < m_width; x++)
		{
			m_data[0][x + y * m_width] *= L;
			m_data[1][x + y * m_width] *= a;
			m_data[2][x + y * m_width] *= b;
		}
	}
}

template <typename T> __forceinline T LabInvF(const T t)
{
	const T cut = T(6) / T(29);
	if (t > T(16)) return T(16) * T(16) * T(16);
	if (t > cut) return t * t * t;
	if (t < T(2) / T(29)) return T(0);
	return T(3) * (T(6) / T(29)) * (T(6) / T(29)) * (t - (T(4) / T(29)));
}

__forceinline float sRGBInvC(const float c)
{
	const float cut = 0.0031308f;
	if (c <= cut) return c * 12.92f;
	return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

__forceinline double sRGBInvC(const double c)
{
	const double cut = 0.0031308;
	if (c <= cut) return c * 12.92;
	return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

template <typename T> __forceinline T saturatef(T v) { return std::max(T(0), std::min(v, T(1))); }

template <typename T> void FloatImage<T>::ConvertLab2Rgb(const FloatImage<T> &Lab)
{
	resize(Lab);
	for (int y = 0; y < (int)m_height; y++)
	{
		for (unsigned int x = 0; x < m_width; x++)
		{
			int idx = x + y * m_width;
			T L = Lab.m_data[0][idx];
			T a = Lab.m_data[1][idx];
			T b = Lab.m_data[2][idx];

			T X, Y, Z;

			X = LabInvF((L + 16.0f) / 116.0f + a / 500.0f);
			Y = LabInvF((L + 16.0f) / 116.0f);
			Z = LabInvF((L + 16.0f) / 116.0f - b / 200.0f);

			//T X = (0.49f    * red + 0.31f    * green + 0.20f    * blue);
			//T Y = (0.17697f * red + 0.81240f * green + 0.01063f * blue);
			//T Z = (                 0.01f    * green + 0.99f    * blue);

			T red, green, blue;

			red = saturatef(3.2406f * X - 1.5372f * Y - 0.4986f * Z);
			green = saturatef(-0.9689f * X + 1.8758f * Y + 0.0415f * Z);
			blue = saturatef(0.0557f * X - 0.2040f * Y + 1.0570f * Z);

			red = sRGBInvC(red); green = sRGBInvC(green); blue = sRGBInvC(blue);

			m_data[0][idx] = red;
			m_data[1][idx] = green;
			m_data[2][idx] = blue;
		}
	}
}

__forceinline float LabF(const float t)
{
	const float cut = (6.0f / 29.0f) * (6.0f / 29.0f) * (6.0f / 29.0f);
	if (t > 16.0f * 16.0f * 16.0f) return 16.0f;
	if (t > cut) return expf(logf(t) / 3.0f);
	if (t < 0.0f) return 4.0f / 29.0f;
	return (29.0f / 6.0f) * (29.0f / 6.0f) * (t / 3.0f) + (4.0f / 29.0f);
}

__forceinline float sRGBC(const float c)
{
	const float cut = 0.04045f;
	if (c <= cut) return c / 12.92f;
	return powf((c + 0.055f) / 1.055f, 2.4f);
}

__forceinline double LabF(const double t)
{
	const double cut = (6.0 / 29.0) * (6.0 / 29.0) * (6.0 / 29.0);
	if (t > 16.0 * 16.0 * 16.0) return 16.0;
	if (t > cut) return exp(log(t) / 3.0);
	if (t < 0.0) return 4.0 / 29.0;
	return (29.0 / 6.0) * (29.0 / 6.0) * (t / 3.0) + (4.0 / 29.0);
}

__forceinline double sRGBC(const double c)
{
	const double cut = 0.04045;
	if (c <= cut) return c / 12.92;
	return pow((c + 0.055) / 1.055, 2.4);
}

template <typename T> void FloatImage<T>::ConvertRgb2Lab(const FloatImage<T> &RGB)
{
	resize(RGB);
	for (int y = 0; y < (int)m_height; y++)
	{
		for (unsigned int x = 0; x < m_width; x++)
		{
			int idx = x + y * m_width;

			T red, green, blue;
			red = RGB.m_data[0][idx];
			green = RGB.m_data[1][idx];
			blue = RGB.m_data[2][idx];

			T X, Y, Z;

			// sRGB
			red = sRGBC(red); green = sRGBC(green); blue = sRGBC(blue);
			X = (0.4124f * red + 0.3576f * green + 0.1805f * blue);
			Y = (0.2126f * red + 0.7152f * green + 0.0722f * blue);
			Z = (0.0193f * red + 0.1192f * green + 0.9505f * blue);

			T fX, fY, fZ;

			fX = LabF(X);
			fY = LabF(Y);
			fZ = LabF(Z);

			T L, a, b;

			L = 116.0f * fY - 16.0f;
			a = 500.0f * (fX - fY);
			b = 200.0f * (fY - fZ);
			m_data[0][idx] = L;
			m_data[1][idx] = a;
			m_data[2][idx] = b;
		}
	}
}

#define GAUSSIAN_RANGE 3
#define GAUSSIAN_WEIGHTED_RANGE 3
//#define GAUSSIAN_DERIVATIVE_WEIGHT 1.4142135623730950488016887242097f
//#define GAUSSIAN_DERIVATIVE_WEIGHT 2.0f

__forceinline__ float gaussian(float x0, float size)
{
	float d = x0 / size;
	return expf(-0.5f * d * d);
}

__forceinline__ double gaussian(double x0, double size)
{
	double d = x0 / size;
	return exp(-0.5 * d * d);
}

template <typename T> void FloatImage<T>::GaussianFilter(const FloatImage<T>&  source, FloatImage<T>&  scratch, T scale, T dx, T dy, BoundaryCondition boundary, bool add)
{
	resize(source);
	scratch.resize(source);
	for (unsigned int i = 0; i < m_data.size(); i++)
	{
		GaussianFilterSTY(scratch.m_data[i], source.m_data[i], m_width, m_height, scale, dy, boundary, false);
		GaussianFilterSTX(m_data[i], scratch.m_data[i], m_width, m_height, scale, dx, boundary, add);
	}
}

template <typename T> void FloatImage<T>::GaussianSplat(const FloatImage<T>&  source, FloatImage<T>&  scratch, T scale, T dx, T dy, BoundaryCondition boundary, bool add)
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

template <typename T> void FloatImage<T>::GaussianWeightedFilter(FloatImage<T> &msk, const FloatImage<T> &source, FloatImage<T> &scratch, const FloatImage<T> &mask, T scale, T dx, T dy, BoundaryCondition boundary, bool add)
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

template <typename T> void FloatImage<T>::GaussianFilterSTX(int target, const FloatImage<T>&  source, int component, T scale, T dx, BoundaryCondition boundary, bool add)
{
	GaussianFilterSTX(m_data[target], source.m_data[component], m_width, m_height, scale, dx, boundary, add);
}

template <typename T> void FloatImage<T>::GaussianFilterSTY(int target, const FloatImage<T>&  source, int component, T scale, T dy, BoundaryCondition boundary, bool add)
{
	GaussianFilterSTY(m_data[target], source.m_data[component], m_width, m_height, scale, dy, boundary, add);
}

template <typename T> void FloatImage<T>::GaussianSplatSTX(int target, const FloatImage<T>&  source, int component, T scale, T dx, BoundaryCondition boundary, bool add)
{
	GaussianSplatSTX(m_data[target], source.m_data[component], m_width, m_height, scale, dx, boundary, add);
}

template <typename T> void FloatImage<T>::GaussianSplatSTY(int target, const FloatImage<T>&  source, int component, T scale, T dy, BoundaryCondition boundary, bool add)
{
	GaussianSplatSTY(m_data[target], source.m_data[component], m_width, m_height, scale, dy, boundary, add);
}

template <typename T> void FloatImage<T>::CopyComponent(int target, const FloatImage<T>&  source, int component)
{
	std::copy(source.m_data[component], source.m_data[component] + (size_t)m_width * m_height, m_data[target]);
}

template <typename T> void FloatImage<T>::GaussianFilterSTX(T *target, T *source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add)
{
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int guard = ((bound / width) + 1) * width;

	if (boundary == BoundaryCondition::Renormalize)
	{
		for (int x = 0; x < width; x++)
		{
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
				T g = gaussian(d, scale);
				T t = source[x + y * width] * g;
				for (int x0 = 1; x0 <= bound; x0++)
				{
					T ga = gaussian(-x0 + d, scale);
					T gb = gaussian(x0 + d, scale);
					int xa = x - x0;
					int xb = x + x0;
					if (xa >= 0) t += source[xa + y * width] * ga;
					if (xb < width) t += source[xb + y * width] * gb;
				}
				if (add) target[x + y * width] += t * r_weight;
				else target[x + y * width] = t * r_weight;
			}
		}
	}
	else
	{
		T weight = gaussian(d, scale);
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian(-x0 + d, scale);
			T gb = gaussian(x0 + d, scale);
			weight += ga + gb;
		}
		T r_weight = 1.0f / weight;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				T g = gaussian(d, scale);
				T t = source[x + y * width] * g;
				if (boundary == BoundaryCondition::Zero)
				{
					for (int x0 = 1; x0 <= bound; x0++)
					{
						T ga = gaussian(-x0 + d, scale);
						T gb = gaussian(x0 + d, scale);
						int xa = x - x0;
						int xb = x + x0;
						if (xa >= 0) t += source[xa + y * width] * ga;
						if (xb < width) t += source[xb + y * width] * gb;
					}
				}
				else if (boundary == BoundaryCondition::Repeat)
				{
					for (int x0 = 1; x0 <= bound; x0++)
					{
						T ga = gaussian(-x0 + d, scale);
						T gb = gaussian(x0 + d, scale);
						int xa = (x - x0 + guard) % width;
						int xb = (x + x0) % width;
						t += source[xa + y * width] * ga + source[xb + y * width] * gb;
					}
				}
				else // if (boundary == BoundaryCondition::Border)
				{
					for (int x0 = 1; x0 <= bound; x0++)
					{
						T ga = gaussian(-x0 + d, scale);
						T gb = gaussian(x0 + d, scale);
						int xa = std::max(0, x - x0);
						int xb = std::min(x + x0, width - 1);
						t += source[xa + y * width] * ga + source[xb + y * width] * gb;
					}
				}
				if (add) target[x + y * width] += t * r_weight;
				else target[x + y * width] = t * r_weight;
			}
		}
	}
}

template <typename T> void FloatImage<T>::GaussianFilterSTY(T *target, T *source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add)
{
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int guard = ((bound / height) + 1) * height;

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
	for (int y = 0; y < height; y++)
	{
		if (boundary == BoundaryCondition::Renormalize)
		{
			weight = gaussian(d, scale);

			for (int y0 = 1; y0 <= bound; y0++)
			{
				T ga = gaussian(-y0 + d, scale);
				T gb = gaussian(y0 + d, scale);
				int ya = y - y0;
				int yb = y + y0;
				if (ya >= 0) weight += ga;
				if (yb < height) weight += gb;
			}
			r_weight = 1.0f / weight;
		}
		for (int x = 0; x < width; x++)
		{
			T g = gaussian(d, scale);
			T t = source[x + y * width] * g;
			if ((boundary == BoundaryCondition::Zero) || (boundary == BoundaryCondition::Renormalize))
			{
				for (int y0 = 1; y0 <= bound; y0++)
				{
					T ga = gaussian(-y0 + d, scale);
					T gb = gaussian(y0 + d, scale);
					int ya = y - y0;
					int yb = y + y0;
					if (ya >= 0) t += source[x + ya * width] * ga;
					if (yb < height) t += source[x + yb * width] * gb;
				}
			}
			else if (boundary == BoundaryCondition::Repeat)
			{
				for (int y0 = 1; y0 <= bound; y0++)
				{
					T ga = gaussian(-y0 + d, scale);
					T gb = gaussian(y0 + d, scale);
					int ya = (y - y0 + guard) % height;
					int yb = (y + y0) % height;
					t += source[x + ya * width] * ga + source[x + yb * width] * gb;
				}
			}
			else // if (boundary == BoundaryCondition::Border)
			{
				for (int y0 = 1; y0 <= bound; y0++)
				{
					T ga = gaussian(-y0 + d, scale);
					T gb = gaussian(y0 + d, scale);
					int ya = std::max(0, y - y0);
					int yb = std::min(y + y0, height - 1);
					t += source[x + ya * width] * ga + source[x + yb * width] * gb;
				}
			}
			if (add) target[x + y * width] += t * r_weight;
			else target[x + y * width] = t * r_weight;
		}
	}
}

template <typename T> void FloatImage<T>::GaussianSplatSTX(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add)
{
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int guard = ((bound / width) + 1) * width;

	if (!add) std::fill(target, target + width * height, T(0));

	if (boundary == BoundaryCondition::Renormalize)
	{
		for (int x = 0; x < width; x++)
		{
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
					if (xa >= 0) target[xa + y * width] += t * ga;
					if (xb < width) target[xb + y * width] += t * gb;
				}
				T g = gaussian(d, scale);
				target[x + y * width] += t * g;
			}
		}
	}
	else // if (boundary == BoundaryCondition::Border)
	{
		T weight = gaussian(d, scale);
		for (int x0 = 1; x0 <= bound; x0++)
		{
			T ga = gaussian(-x0 + d, scale);
			T gb = gaussian(x0 + d, scale);
			weight += ga + gb;
		}
		T r_weight = 1.0f / weight;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				T t = source[x + y * width] * r_weight;
				for (int x0 = 1; x0 <= bound; x0++)
				{
					T ga = gaussian(-x0 + d, scale);
					T gb = gaussian(x0 + d, scale);
					int xa = std::max(0, x - x0);
					int xb = std::min(x + x0, width - 1);
					target[xa + y * width] += t * ga;
					target[xb + y * width] += t * gb;
				}
				T g = gaussian(d, scale);
				target[x + y * width] += t * g;
			}
		}
	}
}

template <typename T> void FloatImage<T>::GaussianSplatSTY(T *target, T *source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add)
{
	int bound = (int)floor(GAUSSIAN_RANGE * scale);
	int guard = ((bound / height) + 1) * height;

	if (!add) std::fill(target, target + width * height, T(0));

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
	for (int y = 0; y < height; y++)
	{
		if (boundary == BoundaryCondition::Renormalize)
		{
			weight = gaussian(d, scale);

			for (int y0 = 1; y0 <= bound; y0++)
			{
				T ga = gaussian(-y0 + d, scale);
				T gb = gaussian(y0 + d, scale);
				int ya = y - y0;
				int yb = y + y0;
				if (ya >= 0) weight += ga;
				if (yb < height) weight += gb;
			}
			r_weight = 1.0f / weight;
		}
		for (int x = 0; x < width; x++)
		{
			T t = source[x + y * width] * r_weight;
			if (boundary == BoundaryCondition::Renormalize)
			{
				for (int y0 = 1; y0 <= bound; y0++)
				{
					T ga = gaussian(-y0 + d, scale);
					T gb = gaussian(y0 + d, scale);
					int ya = y - y0;
					int yb = y + y0;
					if (ya >= 0) target[x + ya * width] += t * ga;
					if (yb < height) target[x + yb * width] += t * gb;
				}
			}
			else // if (boundary == BoundaryCondition::Border)
			{
				for (int y0 = 1; y0 <= bound; y0++)
				{
					T ga = gaussian(-y0 + d, scale);
					T gb = gaussian(y0 + d, scale);
					int ya = std::max(0, y - y0);
					int yb = std::min(y + y0, height - 1);
					target[x + ya * width] += t * ga;
					target[x + yb * width] += t * gb;
				}
			}
			T g = gaussian(d, scale);
			target[x + y * width] += t * g;
		}
	}
}

template <typename T> void FloatImage<T>::resize(const FloatImage<T> &source)
{
	if (!check(source))
	{
		if (m_mapped)
		{
			std::cout << "Trying to resize mapped linear image." << std::endl;
			exit(-1);
		}
		dealloc();

		m_width = source.m_width;
		m_height = source.m_height;
		m_components = source.m_components;

		alloc();
	}
}

template <typename T> void FloatImage<T>::resize(unsigned int width, unsigned int height, unsigned int components)
{
	if (!((m_width == width) && (m_height == height) && (m_components == components)))
	{
		if (m_mapped)
		{
			std::cout << "Trying to resize mapped linear image." << std::endl;
			exit(-1);
		}
		dealloc();

		m_width = width;
		m_height = height;
		m_components = components;

		alloc();
	}
}

template <typename T> void FloatImage<T>::resizeExtra(unsigned int width, unsigned int height, unsigned int components)
{
	if (!((m_width == width) && (m_height == height) && (m_components >= components)))
	{
		if (m_mapped)
		{
			std::cout << "Trying to resize mapped linear image." << std::endl;
			exit(-1);
		}
		dealloc();

		m_width = width;
		m_height = height;
		m_components = components;

		alloc();
	}
}

template <typename T> bool FloatImage<T>::check(const FloatImage<T> &source)
{
	return ((m_width == source.m_width) && (m_height == source.m_height) && (m_components == source.m_components));
}

template <typename T> void FloatImage<T>::FillDiffusion(const FloatImage<T> &source, FloatImage<T> &mask)
{
	resize(source);
	// mask is binary
#pragma omp parallel for
	for (int p = 0; p < (int)(mask.GetWidth() * mask.GetHeight()); p++) if (mask.m_data[0][p] != 1.0f) mask.m_data[0][p] = 0.0f;

	FloatImage<T> *scratch[4];
	for (int i = 0; i < 4; i++) scratch[i] = new FloatImage<T>();
	scratch[0]->Copy(source);
	scratch[1]->Copy(source);
	scratch[2]->Copy(mask);
	scratch[3]->Copy(mask);

	bool complete = false;

	// gaussian with sigma = 1
	const double w0 = 0.25 / (1.0 + exp(-0.5));
	const double w1 = 0.25 - w0;

	int iter = 0;
#pragma omp parallel
	{
#pragma omp for
		for (int p = 0; p < (int)(m_width * m_height); p++)
		{
			if (mask.m_data[0][p] != 1.0f)
			{
				for (unsigned int c = 0; c < source.GetComponents(); c++)
				{
					scratch[0]->m_data[c][p] = 0.0f;
				}
			}
		}

		int t = omp_get_thread_num();
		double new_v[3] = { 0.0, 0.0, 0.0 };
		double new_m = 0.0;
		double total_w = 0.0;
		while (!complete)
		{
#pragma omp barrier
			if (t == 0) complete = true;
			bool my_complete = true;
#pragma omp for
			for (int p = 0; p < (int)(m_width * m_height); p++)
			{
				if (mask.m_data[0][p] != 1.0f)
				{
					int x = p % m_width;
					int y = p / m_width;
					for (unsigned int i = 0; i < m_data.size(); i++) new_v[i] = 0.0;
					new_m = 0.0;
					total_w = 0.0;
					for (int y0 = std::max(0, y - 1); y0 <= std::min((int)m_height - 1, y + 1); y0++)
					{
						for (int x0 = std::max(0, x - 1); x0 <= std::min((int)m_width - 1, x + 1); x0++)
						{
							if ((x != x0) || (y != y0))
							{
								int p0 = x0 + y0 * m_width;
								double w = exp(-0.5 * (((size_t)x - x0) * ((size_t)x - x0) + ((size_t)y - y0) * ((size_t)y - y0)));
								new_m += w * scratch[2]->m_data[0][p0];
								for (unsigned int i = 0; i < m_data.size(); i++) new_v[i] += w * scratch[0]->m_data[i][p0];
								total_w += w;
							}
						}
					}
					if (total_w != 0.0)
					{
						new_m /= total_w;
						for (unsigned int i = 0; i < m_data.size(); i++) new_v[i] /= total_w;
					}
					scratch[3]->m_data[0][p] = (T)new_m;
					for (unsigned int i = 0; i < m_data.size(); i++) scratch[1]->m_data[i][p] = (T)new_v[i];
					if ((my_complete) && (iter < 50000))
					{
						my_complete = ((scratch[3]->m_data[0][p] != T(0)) && (scratch[2]->m_data[0][p] != T(0)) && (fabs(scratch[3]->m_data[0][p] - scratch[2]->m_data[0][p]) < T(1e-6)));
						if ((my_complete) && (iter < 10000))
						{
							double d = 0.0;
							for (unsigned int i = 0; i < m_data.size(); i++)
							{
								double delta = ((double)scratch[1]->m_data[i][p] / (double)scratch[3]->m_data[0][p]) - ((double)scratch[0]->m_data[i][p] / (double)scratch[2]->m_data[0][p]);
								d += delta * delta;
							}
							my_complete &= (d < 1e-12);
						}
					}
				}
			}
#pragma omp barrier
			if (!my_complete) complete = false;
#pragma omp barrier
			if (t == 0)
			{
				std::swap(scratch[0], scratch[1]);
				std::swap(scratch[2], scratch[3]);
				iter++;
			}
		}
	}
#pragma omp parallel for
	for (int i = 0; i < (int)m_data.size(); i++)
	{
		for (int p = 0; p < (int)(m_width * m_height); p++)
		{
			m_data[i][p] = scratch[0]->m_data[i][p] / scratch[2]->m_data[0][p];
		}
	}

	for (int i = 0; i < 4; i++) delete scratch[i];
}

template <typename T> void FloatImage<T>::AnisotropicDiffusion(const FloatImage<T>&  source, FloatImage<T>&  mask)
{
	resize(source);

	FloatImage<T>* scratch[2];
	for (int i = 0; i < 2; i++) scratch[i] = new FloatImage<T>();
	scratch[0]->Copy(source);
	scratch[1]->Copy(source);

	bool complete = false;

	// gaussian with sigma = 1
	const double w0 = 0.25 / (1.0 + exp(-0.5));
	const double w1 = 0.25 - w0;

	int iter = 0;
#pragma omp parallel
	{

		int t = omp_get_thread_num();
		double new_v[3] = { 0.0, 0.0, 0.0 };
		double old_v[3] = { 0.0, 0.0, 0.0 };

		const double lambda = 4.0;
		const double k = 0.25;

		while (!complete)
		{
#pragma omp barrier
			if (t == 0) complete = true;
			bool my_complete = true;
#pragma omp for
			for (int p = 0; p < (int)(m_width * m_height); p++)
			{
				if (mask.m_data[0][p] != 1.0f)
				{
					int x = p % m_width;
					int y = p / m_width;
					for (unsigned int i = 0; i < m_data.size(); i++) new_v[i] = old_v[i] = scratch[0]->m_data[i][p];
					for (int y0 = std::max(0, y - 1); y0 <= std::min((int)m_height - 1, y + 1); y0++)
					{
						for (int x0 = std::max(0, x - 1); x0 <= std::min((int)m_width - 1, x + 1); x0++)
						{
							if ((x != x0) || (y != y0))
							{
								int p0 = x0 + y0 * m_width;
								double w = exp(-0.5 * (((size_t)x - x0) * ((size_t)x - x0) + ((size_t)y - y0) * ((size_t)y - y0)));
								if (mask.m_data[0][p0] == 1.0f) w *= 10.0;
								for (unsigned int i = 0; i < m_data.size(); i++)
								{
									double nabla = scratch[0]->m_data[i][p0] - old_v[i];
									double c = exp(-(nabla * nabla) / (k * k));
									new_v[i] += lambda * w * c * nabla;
								}
							}
						}
					}
					for (unsigned int i = 0; i < m_data.size(); i++) scratch[1]->m_data[i][p] = (T)new_v[i];
					if ((my_complete) && (iter < 10000))
					{
						double d = 0.0;
						for (unsigned int i = 0; i < m_data.size(); i++)
						{
							double delta = new_v[i] - old_v[i];
							d += delta * delta;
						}
						my_complete &= (d < 1e-12);
					}
				}
			}
#pragma omp barrier
			if (!my_complete) complete = false;
#pragma omp barrier
			if (t == 0)
			{
				std::swap(scratch[0], scratch[1]);
				iter++;
			}
		}
	}
#pragma omp parallel for
	for (int i = 0; i < (int)m_data.size(); i++)
	{
		for (int p = 0; p < (int)(m_width * m_height); p++)
		{
			m_data[i][p] = scratch[0]->m_data[i][p];
		}
	}

	for (int i = 0; i < 2; i++) delete scratch[i];
}

template class FloatImage<float>;
template class FloatImage<double>;
