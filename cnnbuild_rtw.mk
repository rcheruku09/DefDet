###########################################################################
## Makefile generated for MATLAB file/project |>PROJNAME<|. 
## 
## Makefile     : cnnbuild_rtw.mk
## 
## Final product: ../cnnbuild.a
## Product type : Static Library
## 
###########################################################################
#
# Copyright 2016-2018 The MathWorks, Inc.

###########################################################################
## MACROS
##########################################################################

PRODUCT_NAME              = ../cnnbuild.a
MAKEFILE                  = cnnbuild_rtw.mk
START_DIR                 = /mathworks/home/rcheruku/Downloads/Defective\ prod\ detect\ dem/codegen
ARCH                      = glnxa64 
MATLAB                    = /mathworks/devel/jobarchive/Bspkg18b/.zfs/snapshot/Bspkg18b.1000207.pass.ja1/current/build/matlab
MATLAB_ARCH_BIN           = $(MATLABROOT)/bin/$(ARCH)


###########################################################################
## TOOLCHAIN SPECIFICATIONS
###########################################################################

# Toolchain Name:          GNU gcc/g++ v4.4.x | gmake (64-bit Linux)
# Supported Version(s):    4.4.x

#------------------------
# BUILD TOOL COMMANDS
#------------------------

# C Compiler: GCC Compiler Driver
CC = gcc

# Linker: GCC Compiler Driver
LD = g++ 

# C++ Compiler: G++ Compiler Driver
CPP = g++

# C++ Linker: G++ Compiler Driver
CPP_LD = g++ 

# Archiver: GNU Archiver
AR = ar

# Execute: Execute
EXECUTE = $(PRODUCT)

# Builder: GMAKE Utility
MAKE_PATH = $(MATLAB)/bin/$(ARCH)
MAKE = $(MAKE_PATH)/gmake

#-------------------------
# Directives/Utilities
#-------------------------



#----------------------------------------
# "Faster Builds" Build Configuration
#----------------------------------------

ARFLAGS =  ruvs
CFLAGS =  -c $(C_STANDARD_OPTS) -fPIC -O3 -fno-loop-optimize -fno-aggressive-loop-optimizations
CPPFLAGS =  -c $(CPP_STANDARD_OPTS) -fPIC -O3 -fno-loop-optimize -fno-aggressive-loop-optimizations
CPP_LDFLAGS =  -Wl,-rpath,"$(MATLAB_ARCH_BIN)",-L"$(MATLAB_ARCH_BIN)"
CPP_SHAREDLIB_LDFLAGS =  -shared -Wl,-rpath,"$(MATLAB_ARCH_BIN)",-L"$(MATLAB_ARCH_BIN)" -Wl,--no-undefined
DOWNLOAD_FLAGS = 
EXECUTE_FLAGS = 
LDFLAGS =  -Wl,-rpath,"$(MATLAB_ARCH_BIN)",-L"$(MATLAB_ARCH_BIN)"
MEX_CPPFLAGS =  -R2018a -MATLAB_ARCH=$(ARCH) $(INCLUDES)   CXXOPTIMFLAGS="$(C_STANDARD_OPTS)  -O3 -fno-loop-optimize -fno-aggressive-loop-optimizations  $(DEFINES)"   -silent
MEX_CPPLDFLAGS =  LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS'
MEX_CFLAGS =  -R2018a -MATLAB_ARCH=$(ARCH) $(INCLUDES)   COPTIMFLAGS="$(C_STANDARD_OPTS)  -O3 -fno-loop-optimize -fno-aggressive-loop-optimizations  $(DEFINES)"   -silent
MEX_LDFLAGS =  LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS' LDFLAGS=='$$LDFLAGS'
MAKE_FLAGS =  -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE) -f $(MAKEFILE)
SHAREDLIB_LDFLAGS =  -shared -Wl,-rpath,"$(MATLAB_ARCH_BIN)",-L"$(MATLAB_ARCH_BIN)" -Wl,--no-undefined


#--------------------
# File extensions
#--------------------
#
# |>FILEEXTFLAGS<|
#

###########################################################################
## OUTPUT INFO
###########################################################################

PRODUCT = $(PRODUCT_NAME)

###########################################################################
## INCLUDE PATHS
###########################################################################

INCLUDES_BUILDINFO = -I"$(START_DIR)"

INCLUDES = $(INCLUDES_BUILDINFO)  -I"" -I"/include"

###########################################################################
## SOURCE FILES
###########################################################################

SRCS =  MWConvLayer.cpp cnn_api.cpp MWCNNLayerImpl.cpp MWConvLayerImpl.cpp MWTargetNetworkImpl.cpp cnn_exec.cpp

ALL_SRCS = $(SRCS)

LIB_BUILD = 1
###########################################################################
## OBJECTS
###########################################################################

OBJS =  MWConvLayer.o cnn_api.o MWCNNLayerImpl.o MWConvLayerImpl.o MWTargetNetworkImpl.o cnn_exec.o

ALL_OBJS = $(OBJS)

###########################################################################
## SYSTEM LIBRARIES
###########################################################################

SYSTEM_LIBS =  

TOOLCHAIN_LIBS =  /lib/libarm_compute.so  /lib/libarm_compute_core.so 

###########################################################################
## PHONY TARGETS
###########################################################################

.PHONY : all build buildobj clean info


all : build
	@echo "### Successfully generated all binary outputs."

build : buildobj $(PRODUCT)

buildobj : $(OBJS)

###########################################################################
## FINAL TARGET
###########################################################################

ifeq ($(LIB_BUILD),1)
ifeq ($(OS),Windows_NT)
$(PRODUCT) : $(OBJS)
	$(AR) /OUT:$(PRODUCT) $(OBJS) $(SYSTEM_LIBS) $(TOOLCHAIN_LIBS)
	@echo "### Created: $(PRODUCT)"
else
$(PRODUCT) : $(OBJS)
	$(AR) -rcs $(PRODUCT) $(OBJS) $(SYSTEM_LIBS) $(TOOLCHAIN_LIBS)
	@echo "### Created: $(PRODUCT)"
endif
else
ifeq ($(OS),Windows_NT)
$(PRODUCT) : $(OBJS)
	$(LD) $(LDFLAGS) /OUT:$(PRODUCT) $(OBJS) $(SYSTEM_LIBS) $(TOOLCHAIN_LIBS)
	@echo "### Created: $(PRODUCT)"    
else
$(PRODUCT) : $(OBJS)
	$(LD) $(LDFLAGS) -o $(PRODUCT) $(OBJS) $(SYSTEM_LIBS) $(TOOLCHAIN_LIBS)
	@echo "### Created: $(PRODUCT)"
endif
endif

###########################################################################
## INTERMEDIATE TARGETS
###########################################################################

#---------------------
# SOURCE-TO-OBJECT
#---------------------

%.o : %.cu
	$(CC) $(CFLAGS)  -mfpu=neon -march=armv7-a -std=gnu++11  $(INCLUDES) -o "$@" "$<"

%.o : %.c
	$(CC) $(CFLAGS)  -mfpu=neon -march=armv7-a -std=gnu++11  $(INCLUDES) -o "$@" "$<"

ifeq ($(OS),Windows_NT)
%.obj:%.cpp
	$(CC) $(CPPFLAGS) /EHsc $(INCLUDES) /c $< /Fo$@ 
else
%.o : %.cpp
	$(CPP) $(CPPFLAGS)  -mfpu=neon -march=armv7-a -std=gnu++11  $(INCLUDES) -o "$@" "$<"
endif

###########################################################################
## DEPENDENCIES
###########################################################################

$(ALL_OBJS) : $(MAKEFILE)


###########################################################################
## MISCELLANEOUS TARGETS
###########################################################################

info : 
	@echo "### PRODUCT = $(PRODUCT)"
	@echo "### PRODUCT_TYPE = $(PRODUCT_TYPE)"
	@echo "### BUILD_TYPE = $(BUILD_TYPE)"
	@echo "### INCLUDES = $(INCLUDES)"
	@echo "### DEFINES = $(DEFINES)"
	@echo "### ALL_SRCS = $(ALL_SRCS)"
	@echo "### ALL_OBJS = $(ALL_OBJS)"
	@echo "### LIBS = $(LIBS)"
	@echo "### MODELREF_LIBS = $(MODELREF_LIBS)"
	@echo "### SYSTEM_LIBS = $(SYSTEM_LIBS)"
	@echo "### TOOLCHAIN_LIBS = $(TOOLCHAIN_LIBS)"
	@echo "### CFLAGS = $(CFLAGS)"
	@echo "### LDFLAGS = $(LDFLAGS)"
	@echo "### SHAREDLIB_LDFLAGS = $(SHAREDLIB_LDFLAGS)"
	@echo "### CPPFLAGS = $(CPPFLAGS)"
	@echo "### CPP_LDFLAGS = $(CPP_LDFLAGS)"
	@echo "### CPP_SHAREDLIB_LDFLAGS = $(CPP_SHAREDLIB_LDFLAGS)"
	@echo "### ARFLAGS = $(ARFLAGS)"
	@echo "### DOWNLOAD_FLAGS = $(DOWNLOAD_FLAGS)"
	@echo "### EXECUTE_FLAGS = $(EXECUTE_FLAGS)"
	@echo "### MAKE_FLAGS = $(MAKE_FLAGS)"


clean : 
	$(ECHO) "### Deleting all derived files..."
	$(RM) $(PRODUCT)
	$(RM) $(ALL_OBJS)
	$(ECHO) "### Deleted all derived files."

