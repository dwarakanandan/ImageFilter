#pragma once

#include "helper.h"

template <typename T>
void Lab2rgb(void *d_L, void *d_a, void *d_b,
	void *d_red, void *d_green, void *d_blue,
	unsigned int width, unsigned int height, size_t pitch);

template <typename T>
void combineLab2rgb(void *d_L, void *d_a, void *d_b,
	void *d_red, void *d_green, void *d_blue,
	unsigned int width, unsigned int height, size_t pitch);

template <typename T>
void Lab2rgb(void *d_L, void *d_grey,
	unsigned int width, unsigned int height, size_t pitch);

template <typename T>
void combineLab2rgb(void *d_L, void *d_grey,
	unsigned int width, unsigned int height, size_t pitch);

