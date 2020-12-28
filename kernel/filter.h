#pragma once

#include "helper.h"
#include "../image/float_image.h"

template <typename T> void GaussianFilterSTX_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

template <typename T> void GaussianFilterSTY_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);