cleanexe:
	-$(RMDIR) $(EXECUTABLE)
clean:
	-$(RMDIR) sdaccel_* TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) .Xil
	-$(RMDIR) *.zip
	-$(RMDIR) *.str
	-$(RMDIR) ./_x
	-$(RMDIR) ./membership.out
	-$(RMDIR) .run
	-$(RMDIR) makefile_gen
	-$(RMDIR) .ipcache

cleanall:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*,*hw*} 
	-$(RMDIR) sdaccel_* TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) .Xil
	-$(RMDIR) *.zip
	-$(RMDIR) *.str
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) ./_x
	-$(RMDIR) ./membership.out
	-$(RMDIR) xclbin*
	-$(RMDIR) .run
	-$(RMDIR) makefile_gen
	-$(RMDIR) .ipcache
	-$(RMDIR) *.csv
	-$(RMDIR) *.protoinst
