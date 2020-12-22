#pragma once

#include "helper.h"
#include <vector>

static const float min_L = -16.0f;
static const float max_L =  100.0f;
static const float min_a = -262.73169f;
static const float max_a =  286.25195f;
static const float min_b = -155.35591f;
static const float max_b = 156.20012f;

#include "cuda_memory.h"
#include "filter.h"
#include "lab2rgb.h"
#include "math_kernel.h"
#include "rgb2lab.h"
#include "sample.h"

// include this only when needed
//#include "kernel_cu.h"
