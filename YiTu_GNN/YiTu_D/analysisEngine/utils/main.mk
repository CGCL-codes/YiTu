SHELL          := /bin/bash
TARGET         := $(TARGETS)
DEVICE         := $(DEVICES)

COMMON_REPO     = ./
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))
UTILS_PATH      = ./utils

.PHONY: all clean cleanall exe hwemuprepare emconfig

APPCONFIG = ./acc_udfs/$(APP)

include $(UTILS_PATH)/help.mk
include $(UTILS_PATH)/utils.mk

include global_para.mk

include $(APPCONFIG)/build.mk
include ./host/host.mk

include autogen/autogen.mk
include acc_template/acc.mk

include $(UTILS_PATH)/bitstream.mk
include $(UTILS_PATH)/clean.mk

all: $(BINARY_CONTAINERS) emconfig $(EXECUTABLE)
exe: $(EXECUTABLE)