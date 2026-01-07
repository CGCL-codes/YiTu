################################# additional compile configurations #######################
#CLFLAGS += --config ./acc_template/bandwidth.cfg
LDCLFLAGS += --config ./acc_template/connectivity.cfg
################################# add the gs kernel to bitstream ##########################

$(XCLBIN)/kernelApply.$(TARGET).$(DSA).xo: $(GS_KERNEL_BASE_PATH)/kernel_apply/kernel_apply.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) -c -k kernelApply -I'$(<D)' -o'$@' '$<'
BINARY_CONTAINER_OBJS += $(XCLBIN)/kernelApply.$(TARGET).$(DSA).xo


$(XCLBIN)/kernelHBMWrapper.$(TARGET).$(DSA).xo: $(GS_KERNEL_BASE_PATH)/kernel_hbm_wrapper/kernel_hbm_wrapper.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) -c -k kernelHBMWrapper -I'$(<D)' -o'$@' '$<'

BINARY_CONTAINER_OBJS += $(XCLBIN)/kernelHBMWrapper.$(TARGET).$(DSA).xo



ifneq ($(strip $(LITTLE_KERNEL_NUM)), 0)

$(XCLBIN)/littleKernelScatterGather.$(TARGET).$(DSA).xo: $(GS_KERNEL_BASE_PATH)/kernel_little_gs/kernel_scatter_gather.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) -c -k littleKernelScatterGather -I'$(<D)' -o'$@' '$<'

BINARY_CONTAINER_OBJS += $(XCLBIN)/littleKernelScatterGather.$(TARGET).$(DSA).xo


$(XCLBIN)/kernelLittleGSMerger.$(TARGET).$(DSA).xo: $(GS_KERNEL_BASE_PATH)/kernel_little_gs_merger/kernel_little_gs_merger.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) -c -k kernelLittleGSMerger -I'$(<D)' -o'$@' '$<'

BINARY_CONTAINER_OBJS += $(XCLBIN)/kernelLittleGSMerger.$(TARGET).$(DSA).xo

endif



ifneq ($(strip $(BIG_KERNEL_NUM)), 0)

$(XCLBIN)/bigKernelScatterGather.$(TARGET).$(DSA).xo: $(GS_KERNEL_BASE_PATH)/kernel_big_gs/kernel_scatter_gather.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) -c -k bigKernelScatterGather -I'$(<D)' -o'$@' '$<'

BINARY_CONTAINER_OBJS += $(XCLBIN)/bigKernelScatterGather.$(TARGET).$(DSA).xo


$(XCLBIN)/kernelBigGSMerger.$(TARGET).$(DSA).xo: $(GS_KERNEL_BASE_PATH)/kernel_big_gs_merger/kernel_big_gs_merger.cpp
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) -c -k kernelBigGSMerger -I'$(<D)' -o'$@' '$<'

BINARY_CONTAINER_OBJS += $(XCLBIN)/kernelBigGSMerger.$(TARGET).$(DSA).xo

endif



