#pragma once

#include "helper.h"

template <typename T>
void add(void *d_target, T f,
	unsigned int width, unsigned int height, size_t pitch);

template <typename T>
void multiply(void *d_target, T f,
	unsigned int width, unsigned int height, size_t pitch);

template <typename T>
void add(void *d_target, const void *d_a, const void *d_b,
	unsigned int width, unsigned int height, size_t pitch);

template <typename T>
void subtract(void *d_target, const void *d_a, const void *d_b,
	unsigned int width, unsigned int height, size_t pitch);
