include global_para.mk

.PHONY: autogen

autogen:
	rm -f ./acc_template/kernel_hbm_wrapper/acc_apply.h
	rm -f ./acc_template/kernel_hbm_wrapper/kernel_merge_apply_write.cpp
	rm -f ./host/host_config/hbm_mapping.h
	rm -f ./acc_template/connectivity.cfg
	rm -f ./acc_template/kernel_little_gs_merger/kernel_little_gs_merger.cpp
	rm -f ./acc_template/kernel_big_gs_merger/kernel_big_gs_merger.cpp

	# for test
	# touch ./autogen/hbm_wrapper.h
	# touch ./autogen/kernel_hbm_wrapper_template.cpp
	# touch ./autogen/hbm_mapping.h
	# touch ./autogen/connectivity.cfg
	# touch ./autogen/kernel_little_gs_merger.cpp
	# touch ./autogen/kernel_big_gs_merger.cpp

	# for compile
	touch ./acc_template/kernel_hbm_wrapper/hbm_wrapper.h
	touch ./acc_template/kernel_hbm_wrapper/kernel_hbm_wrapper.cpp
	touch ./host/host_config/hbm_mapping.h
	touch ./acc_template/connectivity.cfg
	touch ./acc_template/kernel_little_gs_merger/kernel_little_gs_merger.cpp
	touch ./acc_template/kernel_big_gs_merger/kernel_big_gs_merger.cpp
	python3 ./autogen/autogen.py $(LITTLE_KERNEL_NUM) $(BIG_KERNEL_NUM) $(HAVE_APPLY_OUTDEG) $(HAVE_VERTEX_PROP)