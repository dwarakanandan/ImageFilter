#include "../image/png_image.h"
#include "../image/jpeg_image.h"
#include "../image/float_image.h"
#include "../image/cuda_image.h"
#include "../image/lsmrWrapper.h"
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <locale>
#include <cmath>

DebugStream logger;
AllocationLogger allocationLogger;
long long g_max_memory = 4294967296ll;

#define COMPUTE_TYPE double
//#define COMPUTE_TYPE float

bool Init(int device)
{
	int device_count;
	if (cudaGetDeviceCount(&device_count) != 0)
	{
		std::cout << "CUDA not found." << std::endl;
		return false;
	}
	if (device_count == 0)
	{
		std::cout << "No CUDA devices found." << std::endl;
		return false;
	}
	int cuda_device;
	if (device < 0)
		cuda_device = gpuGetMaxGflopsDeviceId();
	else
	{
		std::cout << "Selected Device " << device << std::endl;
		cuda_device = device;
	}
	checkCudaErrors(cudaSetDevice(cuda_device));
	return true;
}

template <class ComputeImage>
void processImage_internal(ComputeImage& fimage, bool raw_input, COMPUTE_TYPE gamma_input, bool signed_input, bool raw_output, COMPUTE_TYPE gamma_output, bool signed_output, COMPUTE_TYPE scale, COMPUTE_TYPE blur, COMPUTE_TYPE damp, COMPUTE_TYPE dx, COMPUTE_TYPE dy, BoundaryCondition boundary, bool cuda_usage, int device)
{
	ComputeImage scratch;
	if (gamma_input != COMPUTE_TYPE(1)) fimage.Gamma(gamma_input);
	if (raw_input)
	{
		if (signed_input)
		{
			fimage.Attenuate(COMPUTE_TYPE(2));
			fimage.Add(COMPUTE_TYPE(-1));
		}
	}
	else
	{
		fimage.ConvertRgb2Lab(fimage);
	}
	if (scale != COMPUTE_TYPE(1))
	{
		scratch.resize((int)std::floor(fimage.GetWidth() * scale + (0.5)), (int)std::floor(fimage.GetHeight() * scale + (0.5)), fimage.GetComponents());
		scratch.ScaleFast(fimage);
		fimage.Copy(scratch);
	}
	if (blur > COMPUTE_TYPE(0))
	{
		fimage.GaussianFilter(fimage, scratch, blur, dx, dy, boundary);
	}
	else if (blur < COMPUTE_TYPE(0))
	{
		
		ComputeImage base, scratch0, mod;
		base.GaussianFilter(fimage, scratch, -blur, dx, dy, boundary);
		base.SubtractImage(fimage, base);
		mod.resize(fimage);
		unsigned int m, n;
		m = n = fimage.GetWidth() * fimage.GetHeight();
		unsigned int component = 0;
		scratch0.resize(fimage.GetWidth(), fimage.GetHeight(), 1);
		auto l1 = [&](unsigned int m, unsigned int n, const double* x, double* y)
		{
			if (sizeof(COMPUTE_TYPE) == sizeof(double))
			{
				ComputeImage mappedX, mappedY;
				mappedX.CreateMapped((double*)x, fimage.GetWidth(), fimage.GetHeight(), 1);
				mappedY.CreateMapped((double*)y, fimage.GetWidth(), fimage.GetHeight(), 1);
				mappedY.GaussianFilter(mappedX, scratch0, -blur, dx, dy, boundary, true);
			}
			else
			{
				std::cout << "Function not implemented in " << __FILE__ << ":" << __LINE__ << std::endl;
				exit(-1);
			}
		};
		auto l2 = [&](unsigned int m, unsigned int n, double* x, const double* y)
		{
			if (sizeof(COMPUTE_TYPE) == sizeof(double))
			{
				ComputeImage mappedX, mappedY;
				mappedX.CreateMapped((double*)x, fimage.GetWidth(), fimage.GetHeight(), 1);
				mappedY.CreateMapped((double*)y, fimage.GetWidth(), fimage.GetHeight(), 1);
				mappedX.GaussianSplat(mappedY, scratch0, -blur, dx, dy, boundary, true);
			}
			else
			{
				std::cout << "Function not implemented in " << __FILE__ << ":" << __LINE__ << std::endl;
				exit(-1);
			}
		};
		lsmrWrapper<decltype(l1), decltype(l2)> lsmr(l1, l2);
		lsmr.SetOutputStream(std::cout);
		lsmr.SetMaximumNumberOfIterations(4 * n);
#ifndef USE_LSQR
		lsmr.SetLocalSize(10000);
#endif
		// effectively set tolerances to epsilon
		lsmr.SetToleranceA(1e-12);
		lsmr.SetToleranceB(1e-12);
		lsmr.SetEpsilon(1.0e-20);
		lsmr.SetDamp(damp);

		for (unsigned int c = 0; c < fimage.GetComponents(); c++)
		{
			lsmr.Solve(m, n, base.GetArray(c), mod.GetArray(c));
		}

		fimage.AddImage(fimage, mod);
	}
	if (raw_output)
	{
		if (signed_output)
		{
			fimage.Attenuate(COMPUTE_TYPE(0.5));
			fimage.Add(COMPUTE_TYPE(0.5));
		}
	}
	else
	{
		fimage.ConvertLab2Rgb(fimage);
	}
	if (gamma_output != COMPUTE_TYPE(1)) fimage.Gamma(COMPUTE_TYPE(1) / gamma_output);
}

template <class ComputeImage>
void processImage_internal_cuda(ComputeImage& fimage_input, bool raw_input, COMPUTE_TYPE gamma_input, bool signed_input, bool raw_output, COMPUTE_TYPE gamma_output, bool signed_output, COMPUTE_TYPE scale, COMPUTE_TYPE blur, COMPUTE_TYPE damp, COMPUTE_TYPE dx, COMPUTE_TYPE dy, BoundaryCondition boundary, bool cuda_usage, int device)
{
	CudaImage<COMPUTE_TYPE> fimage;
	fimage.Upload(fimage_input);
	CudaImage<COMPUTE_TYPE> scratch;
	if (gamma_input != COMPUTE_TYPE(1)) fimage.Gamma(gamma_input);
	if (raw_input)
	{
		if (signed_input)
		{
			fimage.Attenuate(COMPUTE_TYPE(2));
			fimage.Add(COMPUTE_TYPE(-1));
		}
	}
	else
	{
		fimage.ConvertRgb2Lab(fimage);
	}
	if (scale != COMPUTE_TYPE(1))
	{
		scratch.resize((int)std::floor(fimage.GetWidth() * scale + (0.5)), (int)std::floor(fimage.GetHeight() * scale + (0.5)), fimage.GetComponents());
		scratch.ScaleFast(fimage);
		fimage.Copy(scratch);
	}
	if (blur > COMPUTE_TYPE(0))
	{
		fimage.GaussianFilter(fimage, scratch, blur, dx, dy, boundary);
	}
	else if (blur < COMPUTE_TYPE(0))
	{
		
		ComputeImage base, scratch0, mod;
		CudaImage<COMPUTE_TYPE> base_cuda,scratch0_cuda,mod_cuda;
		base_cuda.SetGaussianArrays(dx,dy,-blur);
		base_cuda.GaussianFilter(fimage, scratch, -blur, dx, dy, boundary);
		base_cuda.SubtractImage(fimage, base_cuda);
		mod_cuda.resize(fimage);
		unsigned int m, n;
		m = n = fimage.GetWidth() * fimage.GetHeight();
		unsigned int component = 0;
		scratch0_cuda.resize(fimage.GetWidth(), fimage.GetHeight(), 1);
		auto l1 = [&](unsigned int m, unsigned int n, const double* x, double* y)
		{
			if (sizeof(COMPUTE_TYPE) == sizeof(double))
			{
				ComputeImage mappedX, mappedY;
				CudaImage<COMPUTE_TYPE> mappedX_cuda, mappedY_cuda;
				mappedX.CreateMapped((double*)x, fimage.GetWidth(), fimage.GetHeight(), 1);
				mappedY.CreateMapped((double*)y, fimage.GetWidth(), fimage.GetHeight(), 1);
				mappedX_cuda.Upload(mappedX);
				mappedY_cuda.Upload(mappedY);
				mappedY_cuda.SetGaussianArrays(base_cuda.GetGaussianArrayX(),base_cuda.GetGaussianArrayY());
				//std::cout<<"Pitch0:"<<mappedX_cuda.GetPitch(0)<<",Pitch1:"<<mappedX_cuda.GetPitch(1)<<",Pitch2:"<<mappedX_cuda.GetPitch(2)<<",Width:"<<mappedX_cuda.GetWidth();
				mappedY_cuda.GaussianFilter(mappedX_cuda, scratch0_cuda, -blur, dx, dy, boundary, true);
				mappedY_cuda.Download(mappedY);

			}
			else
			{
				std::cout << "Function not implemented in " << __FILE__ << ":" << __LINE__ << std::endl;
				exit(-1);
			}
		};
		auto l2 = [&](unsigned int m, unsigned int n, double* x, const double* y)
		{
			if (sizeof(COMPUTE_TYPE) == sizeof(double))
			{
				ComputeImage mappedX, mappedY;
				CudaImage<COMPUTE_TYPE> mappedX_cuda, mappedY_cuda;
				mappedX.CreateMapped((double*)x, fimage.GetWidth(), fimage.GetHeight(), 1);
				mappedY.CreateMapped((double*)y, fimage.GetWidth(), fimage.GetHeight(), 1);
				mappedX_cuda.Upload(mappedX);
				mappedY_cuda.Upload(mappedY);
				mappedX_cuda.SetGaussianArrays(base_cuda.GetGaussianArrayX(),base_cuda.GetGaussianArrayY());
				mappedX_cuda.GaussianSplat(mappedY_cuda, scratch0_cuda, -blur, dx, dy, boundary, true);
				mappedX_cuda.Download(mappedX);
			}
			else
			{
				std::cout << "Function not implemented in " << __FILE__ << ":" << __LINE__ << std::endl;
				exit(-1);
			}
		};
		lsmrWrapper<decltype(l1), decltype(l2)> lsmr(l1, l2);
		lsmr.SetOutputStream(std::cout);
		lsmr.SetMaximumNumberOfIterations(4 * n);
#ifndef USE_LSQR
		lsmr.SetLocalSize(10000);
#endif
		// effectively set tolerances to epsilon
		lsmr.SetToleranceA(1e-12);
		lsmr.SetToleranceB(1e-12);
		lsmr.SetEpsilon(1.0e-20);
		lsmr.SetDamp(damp);
		base_cuda.Download(base);
		mod_cuda.Download(mod);
		// #pragma omp parallel for
		for (unsigned int c = 0; c < fimage.GetComponents(); c++)
		{
			lsmr.Solve(m, n, base.GetArray(c), mod.GetArray(c));
		}

		mod_cuda.Upload(mod);
		fimage.AddImage(fimage, mod_cuda);
	}
	if (raw_output)
	{
		if (signed_output)
		{
			fimage.Attenuate(COMPUTE_TYPE(0.5));
			fimage.Add(COMPUTE_TYPE(0.5));
		}
	}
	else
	{
		fimage.ConvertLab2Rgb(fimage);
	}
	if (gamma_output != COMPUTE_TYPE(1)) fimage.Gamma(COMPUTE_TYPE(1) / gamma_output);


	fimage.Download(fimage_input);
}






void processImage(char *sName, char *dName, bool raw_image, COMPUTE_TYPE gamma, bool signed_image, COMPUTE_TYPE scale, COMPUTE_TYPE blur, COMPUTE_TYPE damp, COMPUTE_TYPE dx, COMPUTE_TYPE dy, BoundaryCondition boundary, bool cuda_usage, int device)
{
	FloatImage<COMPUTE_TYPE> fimage;
	fimage.ReadImage(sName, true);

	std::locale loc;
	std::string sname = sName; sname = sname.substr(sname.length() - 3);
	for (unsigned int i = 0; i < sname.length(); ++i) sname[i] = std::tolower(sname[i], loc);
	std::string dname = dName; dname = dname.substr(dname.length() - 3);
	for (unsigned int i = 0; i < dname.length(); ++i) dname[i] = std::tolower(dname[i], loc);

	bool fixed_input = ((sname.compare("png") == 0) || (sname.compare("jpg") == 0) || (sname.compare("jpeg") == 0));
	bool fixed_output = ((dname.compare("png") == 0) || (dname.compare("jpg") == 0) || (dname.compare("jpeg") == 0));

	if (cuda_usage)
	{
		CudaImage<COMPUTE_TYPE> cimage, scratch;
		// cimage.Upload(fimage);
		processImage_internal_cuda(fimage, 
			raw_image || (!fixed_input), fixed_input ? gamma : COMPUTE_TYPE(1), signed_image && fixed_input, 
			raw_image || (!fixed_output), fixed_output ? gamma : COMPUTE_TYPE(1), signed_image && fixed_output, 
			scale, blur, damp, dx, dy, boundary, cuda_usage, device);
		// cimage.Download(fimage);
	}
	else
	{
		FloatImage<COMPUTE_TYPE> scratch;
		processImage_internal(fimage,
			raw_image || (!fixed_input), fixed_input ? gamma : COMPUTE_TYPE(1), signed_image && fixed_input,
			raw_image || (!fixed_output), fixed_output ? gamma : COMPUTE_TYPE(1), signed_image && fixed_output,
			scale, blur, damp, dx, dy, boundary, cuda_usage, device);
	}

	if (dname.compare("png") == 0)
	{
		PngImage image;
		fimage.CopyIntoImage(image);
		image.WriteImage(dName);
	}
	else if ((dname.compare("jpg") == 0) || (dname.compare("jpeg") == 0))
	{
		JpegImage image;
		fimage.CopyIntoImage(image);
		image.WriteImage(dName);
	}
	else
	{
		fimage.WriteImage(dName);
	}
}

int main(int argc, char* argv[])
{
	auto start_time = std::chrono::high_resolution_clock::now();

	char *dstName = NULL;
	char *srcName = NULL;
	COMPUTE_TYPE scale = COMPUTE_TYPE(1);
	COMPUTE_TYPE gamma = COMPUTE_TYPE(1);
	bool signed_image = false;
	int device = -1;
	bool raw_image = false;
	int start = -1;
	int end = -1;
	bool cuda_usage = true;
	COMPUTE_TYPE blur = COMPUTE_TYPE(0);
	COMPUTE_TYPE damp = COMPUTE_TYPE(0);
	COMPUTE_TYPE dx = COMPUTE_TYPE(0);
	COMPUTE_TYPE dy = COMPUTE_TYPE(0);
	BoundaryCondition boundary = BoundaryCondition::Border;

	for (int i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "-device"))
		{
			device = atoi(argv[i + 1]);
		}
		if (!strcmp(argv[i], "-start"))
		{
			start = atoi(argv[i + 1]);
		}
		if (!strcmp(argv[i], "-end"))
		{
			end = atoi(argv[i + 1]);
		}
		if (!strcmp(argv[i], "-dst"))
		{
			dstName = argv[i + 1];
		}
		if (!strcmp(argv[i], "-src"))
		{
			srcName = argv[i + 1];
		}
		if (!strcmp(argv[i], "-l"))
		{
			logger.open(argv[i + 1]);
			logger.deactivate_cout();
			logger << "cmd:";
			for (int j = 0; j < argc; j++)
			{
				logger << " " << argv[j];
			}
			logger << std::endl;
			logger << std::endl;
			logger.activate_cout();
		}
		if (!strcmp(argv[i], "-signed"))
		{
			signed_image = true;
		}
		if (!strcmp(argv[i], "-raw"))
		{
			raw_image = true;
		}
		if (!strcmp(argv[i], "-scale"))
		{
			scale = (COMPUTE_TYPE)atof(argv[i + 1]);
		}
		if (!strcmp(argv[i], "-gamma"))
		{
			gamma = (COMPUTE_TYPE)atof(argv[i + 1]);
		}
		if (!strcmp(argv[i], "-no_cuda"))
		{
			cuda_usage = false;
		}
		if (!strcmp(argv[i], "-blur"))
		{
			blur = (COMPUTE_TYPE)atof(argv[i + 1]);
		}
		if (!strcmp(argv[i], "-damp"))
		{
			damp = (COMPUTE_TYPE)atof(argv[i + 1]);
		}
		if (!strcmp(argv[i], "-dx"))
		{
			dx = (COMPUTE_TYPE)atof(argv[i + 1]);
		}
		if (!strcmp(argv[i], "-dy"))
		{
			dy = (COMPUTE_TYPE)atof(argv[i + 1]);
		}
		if (!strcmp(argv[i], "-boundary_repeat"))
		{
			boundary = BoundaryCondition::Repeat;
		}
		if (!strcmp(argv[i], "-boundary_zero"))
		{
			boundary = BoundaryCondition::Zero;
		}
		if (!strcmp(argv[i], "-boundary_border"))
		{
			boundary = BoundaryCondition::Border;
		}
		if (!strcmp(argv[i], "-boundary_renormalize"))
		{
			boundary = BoundaryCondition::Renormalize;
		}
	}

	if ((srcName == NULL) || (dstName == NULL))
	{
		std::cout << "usage: " << argv[0] << std::endl;
		std::cout << "           -src <source_image> -dst <dst_image>" << std::endl;
		std::cout << "           [-device <device_num>] [-l <log_file>]" << std::endl;
		std::cout << "           [-start <s>] [-end <e>] (convert range of images)" << std::endl;
		std::cout << "           [-scale <source_scale>] (scale input image)" << std::endl;
		std::cout << "           [-gamma <source_gamma>] (apply gamma correction)" << std::endl;
		std::cout << "           [-signed] [-raw]        (signed output and raw input = no color conversion)" << std::endl;
		std::cout << "           [-blur <radius>]        (gaussian blur, negative values will un-blur)" << std::endl;
	}
	else
	{
		if (cuda_usage)
		{
			if (!Init(device)) cuda_usage = false;
		}

		bool png = false;
		std::locale loc;
		std::string sname = srcName; sname = sname.substr(sname.length() - 3);
		std::string dname = dstName; dname = dname.substr(dname.length() - 3);
		for (unsigned int i = 0; i < sname.length(); ++i) sname[i] = std::tolower(sname[i], loc);
		for (unsigned int i = 0; i < dname.length(); ++i) dname[i] = std::tolower(dname[i], loc);

		if ((end >= 0) && (start < 0)) start = 0;

		if ((start >= 0) && (end >= 0))
		{
			char* dName;
			char* sName;
			allocHostData<char>(dName, strlen(dstName) + 129, __FILE__, __LINE__);
			allocHostData<char>(sName, strlen(srcName) + 129, __FILE__, __LINE__);
			for (int i = start; i <= end; i++)
			{
				std::snprintf(dName, strlen(dstName) + 129, dstName, i);
				std::snprintf(sName, strlen(srcName) + 129, srcName, i);
				processImage(sName, dName, raw_image, gamma, signed_image, scale, blur, damp, dx, dy, boundary, cuda_usage, device);
			}
			freeHostData(dName);
			freeHostData(sName);
		}
		else
		{
			processImage(srcName, dstName, raw_image, gamma, signed_image, scale, blur, damp, dx, dy, boundary, cuda_usage, device);
		}
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	long long hours = std::chrono::duration_cast<std::chrono::hours>(end_time - start_time).count();
	long long minutes = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time).count();
	long long seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
	seconds -= minutes * 60;
	minutes -= hours * 60;
	logger << "Total run time: ";
	if (hours > 0) logger << hours << "h ";
	if ((hours > 0) || (minutes > 0)) logger << minutes << "m ";
	logger << seconds << "s (" << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms)" << std::endl;

	allocationLogger.destroy();

	return 0;
}
