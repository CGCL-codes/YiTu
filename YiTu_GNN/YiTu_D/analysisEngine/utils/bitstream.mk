# Building Hardware
$(XCLBIN)/graph_fpga.$(TARGET).$(DSA).xclbin: $(BINARY_CONTAINER_OBJS)
	$(XOCC) $(CLFLAGS) -l $(LDCLFLAGS)  $(BINARY_LINK_OBJS) -o'$@' $(+)

# Building Host
$(EXECUTABLE): $(HOST_SRCS) cleanexe
	mkdir -p $(XCLBIN)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) -o '$@' $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

.PHONY: hwemuprepare
hwemuprepare:
ifeq ($(TARGET),$(filter $(TARGET), hw_emu))
	@echo "prepare for hw_emu"
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	$(CP) $(UTILS_PATH)/xrt.ini .
	#source $(UTILS_PATH)/hw_emu.sh
else
	@echo "prepare for hw"
endif

check: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(XCLBIN)/graph_fpga.$(TARGET).$(DSA).xclbin rmat-12-4
else
	 ./$(EXECUTABLE) $(XCLBIN)/graph_fpga.$(TARGET).$(DSA).xclbin rmat-12-4
endif