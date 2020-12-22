#pragma once

#define _CRT_SECURE_NO_WARNINGS

// make visual studio happy

#define PNG_DEBUG 3
#include <png.h>
#include <zlib.h>
#include <string.h>
#include <vector>

#ifndef override
#define override
#endif

#define CUDA_ARRAY_UINT8  0
#define CUDA_ARRAY_UINT16 1
#define CUDA_ARRAY_UINT32 2
#define CUDA_ARRAY_UINT64 3
#define CUDA_ARRAY_FLOAT  8
#define CUDA_ARRAY_DOUBLE 9

// Virtual image base class
class Image {
public:
	Image() { m_width = m_height = m_components = m_mask = 0; }
	virtual ~Image() {}

protected:
	unsigned int m_width;
	unsigned int m_height;
    unsigned int m_components;
	unsigned int m_mask;

public:
	virtual bool ReadImage(const char fileName[]) = 0;

	virtual bool WriteImage(const char fileName[]) = 0;

	unsigned int GetWidth() const { return m_width; }

	unsigned int GetHeight() const { return m_height; }

	unsigned int GetComponents() const { return m_components; }

	unsigned int GetMask() const { return m_mask; }

	void setMask(unsigned int mask) { m_mask = mask; }

	virtual void SetWidth(unsigned int width) = 0;

	virtual void SetHeight(unsigned int height) = 0;

	virtual void SetComponents(unsigned int components) = 0;

	virtual void VSetValue(unsigned int x, unsigned int y, unsigned int c, unsigned int v) = 0;

	virtual unsigned int VGetValue(unsigned int x, unsigned int y, unsigned int c) const = 0;

public:
	void CopyImage(const Image &source);
};
