# Put all your source files here 
SET(z_SRCS
	adler32.c
	compress.c
	crc32.c
	deflate.c
	gzclose.c
	gzlib.c
	gzread.c
	gzwrite.c
	infback.c
	inffast.c
	inflate.c
	inftrees.c
	trees.c
	uncompr.c
	zutil.c
)

# this command creates your executable from the files specified in MyProject_SRCS
ADD_LIBRARY(z ${z_SRCS})

# link all necessary libraries here (only the names are needed, the rest is done by CMake)
#TARGET_LINK_LIBRARIES(main_cuda ${CUDA_LIBRARIES})
