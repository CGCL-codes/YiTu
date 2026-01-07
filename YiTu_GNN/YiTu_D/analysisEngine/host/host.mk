VAR_TRUE=true

CP = cp -rf
XCLBIN := ./xclbin_$(TARGET)_$(APP)
DSA := $(call device2sandsa, $(DEVICE))

HOST_ARCH := x86

#Checks for g++
ifeq ($(HOST_ARCH), x86)
ifneq ($(shell expr $(shell g++ -dumpversion) \>= 5), 1)
ifndef XILINX_VIVADO
$(error [ERROR]: g++ version older. Please use 5.0 or above.)
else
CXX := $(XILINX_VIVADO)/tps/lnx64/gcc-6.2.0/bin/g++
$(warning [WARNING]: g++ version older. Using g++ provided by the tool : $(CXX))
endif
endif
else ifeq ($(HOST_ARCH), aarch64)
CXX := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++
else ifeq ($(HOST_ARCH), aarch32)
CXX := $(XILINX_VITIS)/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin/arm-linux-gnueabihf-g++
endif

XOCC := v++

GS_KERNEL_BASE_PATH    = ./acc_template
APPLY_KERNEL_PATH = ./acc_template

include $(ABS_COMMON_REPO)/utils/opencl.mk


HOST_SRCS  = $(wildcard host/*.cpp)
HOST_SRCS += $(wildcard host/graph_loader/*.cpp)
HOST_SRCS += $(wildcard host/acc_setup/*.cpp)
HOST_SRCS += $(wildcard host/host_config/*.cpp)
HOST_SRCS += $(wildcard host/preprocess/*.cpp)
HOST_SRCS += $(wildcard host/process/*.cpp)
HOST_SRCS += $(wildcard host/verification/*.cpp)
HOST_SRCS += $(wildcard $(APPCONFIG)/*.cpp)

CXXFLAGS += $(opencl_CXXFLAGS) -Wall
CXXFLAGS += -I/$(XILINX_SDX)/Vivado_HLS/include/ -O3 -g -fmessage-length=0 -std=c++14 -Wno-deprecated-declarations
CXXFLAGS += -I ./
CXXFLAGS += -I ./host
CXXFLAGS += -I ./host/graph_loader
CXXFLAGS += -I ./host/acc_setup
CXXFLAGS += -I ./host/host_config
CXXFLAGS += -I ./host/preprocess
CXXFLAGS += -I ./host/process
CXXFLAGS += -I ./host/verification
CXXFLAGS += -I ./acc_template/common


CXXFLAGS += -I $(APPCONFIG)
CXXFLAGS += -I ./acc_udfs

# Host linker flags
LDFLAGS := $(opencl_LDFLAGS)
LDFLAGS += -lrt -lstdc++  -lxilinxopencl


CLFLAGS := $(AUTOGEN_CFLAG)

ifeq ($(TARGET),$(filter $(TARGET), hw_emu))
CLFLAGS += -g -t $(TARGET) 
else
CLFLAGS += -t $(TARGET) 
endif

# Kernel compiler global settings
CLFLAGS += --platform $(DEVICE) --save-temps -O3
CLFLAGS += -I $(APPCONFIG)
CLFLAGS += -I ./
CLFLAGS += -I ./acc_template
CLFLAGS += -I ./acc_template/common
CLFLAGS += -I ./acc_udfs

#CLFLAGS += --xp prop:solution.kernel_compiler_margin=10%

# Kernel linker flags
#LDCLFLAGS += --kernel_frequency=250 #$(FREQ)
LDCLFLAGS += --xp prop:solution.kernel_compiler_margin=10% 

EXECUTABLE = host_graph_fpga_$(APP)

EMCONFIG_DIR = $(XCLBIN)/$(DSA)

BINARY_CONTAINERS += $(XCLBIN)/graph_fpga.$(TARGET).$(DSA).xclbin


#Include Libraries
include $(UTILS_PATH)/xcl/xcl.mk
CXXFLAGS +=  $(xcl_CXXFLAGS) 
LDFLAGS +=   $(xcl_CXXFLAGS) 
HOST_SRCS += $(xcl_SRCS) 

CXXFLAGS += -I$(UTILS_PATH)/common/includes/xcl2
HOST_SRCS += $(UTILS_PATH)/common/includes/xcl2/xcl2.cpp

#############################################################################
#                                                                           #
#                     Specific Build Configuration                          #
#                                                                           #
#############################################################################


ifeq ($(strip $(HAVE_APPLY)), $(strip $(VAR_TRUE)))
CXXFLAGS += -DHAVE_APPLY=1
else
CXXFLAGS += -DHAVE_APPLY=0
endif


ifeq ($(strip $(HAVE_EDGE_PROP)), $(strip $(VAR_TRUE)))
CXXFLAGS += -DHAVE_EDGE_PROP=1
CLFLAGS  += -DHAVE_EDGE_PROP=1
else
CXXFLAGS += -DHAVE_EDGE_PROP=0
CLFLAGS  += -DHAVE_EDGE_PROP=0
endif


ifeq ($(strip $(HAVE_UNSIGNED_PROP)), $(strip $(VAR_TRUE)))
CXXFLAGS += -DHAVE_UNSIGNED_PROP=1
CLFLAGS  += -DHAVE_UNSIGNED_PROP=1
else
CXXFLAGS += -DHAVE_UNSIGNED_PROP=0
CLFLAGS  += -DHAVE_UNSIGNED_PROP=0
endif


ifeq ($(strip $(HAVE_APPLY_OUTDEG)), $(strip $(VAR_TRUE)))
CXXFLAGS += -DHAVE_APPLY_OUTDEG=1
CLFLAGS  += -DHAVE_APPLY_OUTDEG=1
else
CXXFLAGS += -DHAVE_APPLY_OUTDEG=0
CLFLAGS  += -DHAVE_APPLY_OUTDEG=0
endif


ifeq ($(strip $(HAVE_VERTEX_PROP)), $(strip $(VAR_TRUE)))
CXXFLAGS += -DHAVE_VERTEX_PROP=1
CLFLAGS  += -DHAVE_VERTEX_PROP=1
else
CXXFLAGS += -DHAVE_VERTEX_PROP=0
CLFLAGS  += -DHAVE_VERTEX_PROP=0
endif



#############################################################################
#                                                                           #
#                     Specific Configuration                                #
#                                                                           #
#############################################################################

CLFLAGS  += -DPARTITION_SIZE=$(PARTITION_SIZE)
CXXFLAGS  += -DPARTITION_SIZE=$(PARTITION_SIZE)

CLFLAGS  += -DLITTLE_KERNEL_DST_BUFFER_SIZE=$(LITTLE_KERNEL_DST_BUFFER_SIZE)
CXXFLAGS  += -DLITTLE_KERNEL_DST_BUFFER_SIZE=$(LITTLE_KERNEL_DST_BUFFER_SIZE)

CLFLAGS  += -DBIG_KERNEL_DST_BUFFER_SIZE=$(BIG_KERNEL_DST_BUFFER_SIZE)
CXXFLAGS  += -DBIG_KERNEL_DST_BUFFER_SIZE=$(BIG_KERNEL_DST_BUFFER_SIZE)

CLFLAGS  += -DSRC_BUFFER_SIZE=$(SRC_BUFFER_SIZE)
CXXFLAGS  += -DSRC_BUFFER_SIZE=$(SRC_BUFFER_SIZE)

CLFLAGS  += -DLOG2_SRC_BUFFER_SIZE=$(LOG2_SRC_BUFFER_SIZE)
CXXFLAGS  += -DLOG2_SRC_BUFFER_SIZE=$(LOG2_SRC_BUFFER_SIZE)

CLFLAGS  += -DVERTEX_REORDER_ENABLE=$(VERTEX_REORDER_ENABLE)
CXXFLAGS  += -DVERTEX_REORDER_ENABLE=$(VERTEX_REORDER_ENABLE)

ifdef TARGET_PARTITION_SIZE
CLFLAGS  += -DTARGET_PARTITION_SIZE=$(TARGET_PARTITION_SIZE)
CXXFLAGS  += -DTARGET_PARTITION_SIZE=$(TARGET_PARTITION_SIZE)
endif

CXXFLAGS  += -DKERNEL_NUM=$(KERNEL_NUM)
CXXFLAGS  += -DBIG_KERNEL_NUM=$(BIG_KERNEL_NUM)
CXXFLAGS  += -DLITTLE_KERNEL_NUM=$(LITTLE_KERNEL_NUM)
CLFLAGS  += -DKERNEL_NUM=$(KERNEL_NUM)
CLFLAGS  += -DBIG_KERNEL_NUM=$(BIG_KERNEL_NUM)
CLFLAGS  += -DLITTLE_KERNEL_NUM=$(LITTLE_KERNEL_NUM)

ifeq ($(TARGET),$(filter $(TARGET), sw_emu hw_emu))
CLFLAGS  += -DSW_EMU
endif