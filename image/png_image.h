#pragma once

#include "image.h"

// compressed integer image format (usually 8 bit per channel; specified in depth)
class PngImage : public Image {
private:
	unsigned int m_bitDepth;
	png_bytep *m_rowPointer;

private:
	void Cleanup();

	bool Alloc();

public:
	PngImage() { m_rowPointer = NULL; m_bitDepth = 8; m_mask = 255; }

	virtual ~PngImage() { Cleanup(); }

public:
	virtual bool ReadImage(const char fileName[]) override;
	
	void SetBitDepth(unsigned int bit_depth);
	
	virtual bool WriteImage(const char fileName[]) override;

	virtual void SetWidth(unsigned int width) override { Cleanup(); m_width = width; Alloc(); }

	virtual void SetHeight(unsigned int height) override { Cleanup(); m_height = height; Alloc(); }

	virtual void SetComponents(unsigned int components) override { Cleanup(); m_components = components; Alloc(); }

	virtual void VSetValue(unsigned int x, unsigned int y, unsigned int c, unsigned int v) override;

	virtual unsigned int VGetValue(unsigned int x, unsigned int y, unsigned int c) const override;
};
