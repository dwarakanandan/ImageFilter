#pragma once

#include "helper.h"
#include "../image/float_image.h"

template <typename T> void GaussianFilterSTX_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

template <typename T> void GaussianFilterSTY_GPU(T* target, T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

template <typename T> void GaussianSplatSTX_GPU(T* target,const T* source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

template <typename T> void GaussianSplatSTY_GPU(T *target,const T *source, int width, int height, T scale, T d, BoundaryCondition boundary, bool add);

template <typename T> void Gamma_GPU(T* target, int target_width,int target_height,T gamma);

template <typename T> void ScaleFast_GPU(T* target,T* source, int target_width,int target_height, int source_width, int source_height);
