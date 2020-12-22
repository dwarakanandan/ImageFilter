# Put all your source files here 
SET(png_SRCS
	jctrans.c
	jdtrans.c
	png.c
	pngerror.c
	pngget.c
	pngmem.c
	pngpread.c
	pngread.c
	pngrio.c
	pngrtran.c
	pngrutil.c
	pngset.c
	pngtest.c
	pngtrans.c
	pngwio.c
	pngwrite.c
	pngwtran.c
	pngwutil.c
)

# this command creates your executable from the files specified in MyProject_SRCS
ADD_LIBRARY(png ${png_SRCS})

# link all necessary libraries here (only the names are needed, the rest is done by CMake)
#TARGET_LINK_LIBRARIES(main_cuda ${CUDA_LIBRARIES})
