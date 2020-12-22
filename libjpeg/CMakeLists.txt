# Put all your source files here 
SET(jpeg_SRCS
	jaricom.c
	jcapistd.c
	jccoefct.c
	jcdctmgr.c
	jcinit.c
	jcmarker.c
	jcomapi.c
	jcprepct.c
	jdapimin.c
	jdarith.c
	jdatasrc.c
	jdcolor.c
	jdhuff.c
	jdmainct.c
	jdmaster.c
	jdpostct.c
	jerror.c
	jfdctfst.c
	jidctflt.c
	jidctint.c
	jmemnobs.c
	jquant2.c
	jcapimin.c
	jcarith.c
	jccolor.c
	jchuff.c
	jcmainct.c
	jcmaster.c
	jcparam.c
	jcsample.c
	jdapistd.c
	jdatadst.c
	jdcoefct.c
	jddctmgr.c
	jdinput.c
	jdmarker.c
	jdmerge.c
	jdsample.c
	jfdctflt.c
	jfdctint.c
	jidctfst.c
	jmemmgr.c
	jquant1.c
	jutils.c
)

# this command creates your executable from the files specified in MyProject_SRCS
ADD_LIBRARY(jpeg ${jpeg_SRCS})

# link all necessary libraries here (only the names are needed, the rest is done by CMake)
#TARGET_LINK_LIBRARIES(main_cuda ${CUDA_LIBRARIES})
