#pragma once

#include "helper.h"
#include "../image/float_image.h"
#include "../image/cuda_image.h"

template <typename T> void GaussianFilterSTX_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* gaussian_array_x);

template <typename T> void GaussianFilterSTY_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* gaussian_array_y);

template <typename T> void GaussianSplatSTX_GPU(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* gaussian_array_x);

template <typename T> void GaussianSplatSTY_GPU(T *target,const T *source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add, T* gaussian_array_y);

template <typename T> void Gamma_GPU(T* target, int target_width,int target_height,T gamma);
