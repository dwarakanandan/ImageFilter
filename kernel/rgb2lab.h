#pragma once

#include "helper.h"

template <typename T>
void rgb2Lab(void *d_red, void *d_green, void *d_blue,
	void *d_L, void *d_a, void *d_b,
	unsigned int width, unsigned int height, size_t pitch);

template <typename T>
void rgb2Lab(void *d_grey, void *d_L,
	unsigned int width, unsigned int height, size_t pitch);

