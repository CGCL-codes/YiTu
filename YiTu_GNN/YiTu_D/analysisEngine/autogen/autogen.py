#!/usr/bin/python3

import os
from re import T
import sys

# configurable hbm wrapper bank id (for vertex properties)
wrapper_kernel_hbm_id  = [1,3,5,7,9,11,13,15,17,19,21,23,25,27]

# configurable little and big wrapper bank id (for edges)
little_and_big_kernels_hbm_id = [0,2,4,6,8,10,12,14,16,18,20,22,24,26]

# configurable little and big kernels slr id
little_and_big_kernels_slr_id = [0,1,2,0,1,2,0,1,2,0,1,2,1,2]

# configurable slr id
little_kernel_slr_id = []
big_kernel_slr_id    = []

little_kernel_hbm_id = []
big_kernel_hbm_id    = []

# string matching and substitution
REGRAPH_ALL_KERNEL_ID_TEXT            = "%REGRAPH_ALL_KERNEL_ID_TEXT%"

REGRAPH_LITTLE_KERNEL_GLOBAL_ID_TEXT  = "%REGRAPH_LITTLE_KERNEL_GLOBAL_ID_TEXT%"
REGRAPH_BIG_KERNEL_GLOBAL_ID_TEXT     = "%REGRAPH_BIG_KERNEL_GLOBAL_ID_TEXT%"

REGRAPH_LITTLE_KERNEL_ID_TEXT         = "%REGRAPH_LITTLE_KERNEL_ID_TEXT%"
REGRAPH_BIG_KERNEL_ID_TEXT            = "%REGRAPH_BIG_KERNEL_ID_TEXT%"

REGRAPH_LITTLE_KERNEL_ID_ARRAY        = "%REGRAPH_LITTLE_KERNEL_ID_ARRAY%"
REGRAPH_BIG_KERNEL_ID_ARRAY           = "%REGRAPH_BIG_KERNEL_ID_ARRAY%"

REGRAPH_LITTLE_KERNEL_NUM             = "%REGRAPH_LITTLE_KERNEL_NUM%"
REGRAPH_BIG_KERNEL_NUM                = "%REGRAPH_BIG_KERNEL_NUM%"

REGRAPH_APPLY_KERNEL_HBM_ID           = "%REGRAPH_APPLY_KERNEL_HBM_ID%"
REGRAPH_LITTLE_KERNEL_HBM_ID          = "%REGRAPH_LITTLE_KERNEL_HBM_ID%"
REGRAPH_BIG_KERNEL_HBM_ID             = "%REGRAPH_BIG_KERNEL_HBM_ID%"

REGRAPH_LITTLE_KERNEL_SLR_ID          = "%REGRAPH_LITTLE_KERNEL_SLR_ID%"
REGRAPH_BIG_KERNEL_SLR_ID             = "%REGRAPH_BIG_KERNEL_SLR_ID%"

REGRAPH_HAVE_APPLY_OUTDEG             = "%REGRAPH_HAVE_APPLY_OUTDEG%"
REGRAPH_HAVE_VERTEX_PROP              = "%REGRAPH_HAVE_VERTEX_PROP%"

REGRAPH_HAVE_LITTLE_KERNEL            = "%REGRAPH_HAVE_LITTLE_KERNEL%"
REGRAPH_HAVE_BIG_KERNEL               = "%REGRAPH_HAVE_BIG_KERNEL%"

#根据核数量划分SLR和HBM资源（顺序分配，确保负载均衡）
def assign_big_little(little_kernel_num, big_kernel_num):
    global little_and_big_kernels_slr_id
    global little_kernel_slr_id
    global big_kernel_slr_id

    global little_and_big_kernels_hbm_id
    global little_kernel_hbm_id
    global big_kernel_hbm_id

    little_kernel_slr_id = little_and_big_kernels_slr_id[:little_kernel_num]
    big_kernel_slr_id    = little_and_big_kernels_slr_id[little_kernel_num:]

    little_kernel_hbm_id = little_and_big_kernels_hbm_id[:little_kernel_num]
    big_kernel_hbm_id    = little_and_big_kernels_hbm_id[little_kernel_num:]
    # print(little_kernel_hbm_id)
    # print(big_kernel_hbm_id)


#小核生成：为每个小核实例生成独立代码段

#大核生成：为每个大核实例生成独立代码段

#条件编译：处理HAVE_APPLY_OUTDEG等编译标志
def gen_kernel_files(template_file, generated_file, little_kernel_num, big_kernel_num, have_apply, have_prop):

    global REGRAPH_LITTLE_KERNEL_ID_TEXT
    global REGRAPH_LITTLE_KERNEL_ID_ARRAY
    global REGRAPH_LITTLE_KERNEL_GLOBAL_ID_TEXT

    global REGRAPH_BIG_KERNEL_ID_TEXT
    global REGRAPH_BIG_KERNEL_ID_ARRAY
    global REGRAPH_BIG_KERNEL_GLOBAL_ID_TEXT

    global REGRAPH_ALL_KERNEL_ID_TEXT
    global REGRAPH_APPLY_KERNEL_HBM_ID

    global REGRAPH_LITTLE_KERNEL_NUM
    global REGRAPH_LITTLE_KERNEL_HBM_ID
    global REGRAPH_LITTLE_KERNEL_SLR_ID

    global REGRAPH_BIG_KERNEL_NUM
    global REGRAPH_BIG_KERNEL_HBM_ID
    global REGRAPH_BIG_KERNEL_SLR_ID

    global REGRAPH_HAVE_APPLY_OUTDEG
    global REGRAPH_HAVE_VERTEX_PROP

    global REGRAPH_HAVE_BIG_KERNEL
    global REGRAPH_HAVE_LITTLE_KERNEL

    global wrapper_kernel_hbm_id
    global little_kernel_hbm_id
    global big_kernel_hbm_id

    global little_kernel_slr_id
    global big_kernel_slr_id

    with open(template_file, "r", encoding="utf-8") as f1, open(generated_file, "w", encoding="utf-8") as f2:
        for line in f1:
            # little kernels
            if REGRAPH_LITTLE_KERNEL_ID_TEXT in line: # deal with little kernels
                for idx in range(little_kernel_num):
                    # little kernel gather scatter merge files
                    if REGRAPH_LITTLE_KERNEL_ID_TEXT in line: # REGRAPH_LITTLE_KERNEL_ID_TEXT
                        tmp_line = line.replace(REGRAPH_LITTLE_KERNEL_ID_TEXT, str(idx+1))
                    if REGRAPH_LITTLE_KERNEL_ID_ARRAY in tmp_line: # REGRAPH_LITTLE_KERNEL_ID_ARRAY
                        tmp_line = tmp_line.replace(REGRAPH_LITTLE_KERNEL_ID_ARRAY, str(idx))
                    if REGRAPH_LITTLE_KERNEL_GLOBAL_ID_TEXT in tmp_line: # REGRAPH_LITTLE_KERNEL_GLOBAL_ID_TEXT
                        tmp_line = tmp_line.replace(REGRAPH_LITTLE_KERNEL_GLOBAL_ID_TEXT, str(idx+1))
                    # connectivity config
                    if REGRAPH_LITTLE_KERNEL_HBM_ID in tmp_line: # REGRAPH_LITTLE_KERNEL_HBM_ID
                        tmp_line = tmp_line.replace(REGRAPH_LITTLE_KERNEL_HBM_ID, str(little_kernel_hbm_id[idx]))
                    if REGRAPH_LITTLE_KERNEL_SLR_ID in tmp_line: # REGRAPH_LITTLE_KERNEL_SLR_ID
                        tmp_line = tmp_line.replace(REGRAPH_LITTLE_KERNEL_SLR_ID, str(little_kernel_slr_id[idx]))
                    
                    # remove axi stream in the same SLR
                    if little_kernel_slr_id[idx] == 0 and "kernelLittleGSMerger_1" not in tmp_line and ":16" in tmp_line: 
                        tmp_line = tmp_line.replace(":16", " ")
                    if little_kernel_slr_id[idx] == 1 and "kernelLittleGSMerger_1" in tmp_line and ":16" in tmp_line: 
                        tmp_line = tmp_line.replace(":16", " ")
                    f2.write(tmp_line)

            elif REGRAPH_BIG_KERNEL_ID_TEXT in line: # deal with big kernels
                # big kernel gather scatter merge files
                for idx in range(big_kernel_num):
                    # big kernel gather scatter merge files
                    if REGRAPH_BIG_KERNEL_ID_TEXT in line: # REGRAPH_BIG_KERNEL_ID_TEXT
                        tmp_line = line.replace(REGRAPH_BIG_KERNEL_ID_TEXT, str(idx+1))
                    if REGRAPH_BIG_KERNEL_ID_ARRAY in tmp_line: # REGRAPH_BIG_KERNEL_ID_ARRAY
                        tmp_line = tmp_line.replace(REGRAPH_BIG_KERNEL_ID_ARRAY, str(idx))
                    if REGRAPH_BIG_KERNEL_GLOBAL_ID_TEXT in tmp_line: # REGRAPH_BIG_KERNEL_GLOBAL_ID_TEXT
                        tmp_line = tmp_line.replace(REGRAPH_BIG_KERNEL_GLOBAL_ID_TEXT, str(idx+1+little_kernel_num))
                    # connectivity config
                    if REGRAPH_BIG_KERNEL_HBM_ID in tmp_line: # REGRAPH_BIG_KERNEL_HBM_ID
                        tmp_line = tmp_line.replace(REGRAPH_BIG_KERNEL_HBM_ID, str(big_kernel_hbm_id[idx]))
                    if REGRAPH_BIG_KERNEL_SLR_ID in tmp_line: # REGRAPH_BIG_KERNEL_SLR_ID
                        tmp_line = tmp_line.replace(REGRAPH_BIG_KERNEL_SLR_ID, str(big_kernel_slr_id[idx]))
                    # remove axi stream in the same SLR
                    if big_kernel_slr_id[idx] == 0 and "kernelBigGSMerger_1" not in tmp_line and ":16" in tmp_line: 
                        tmp_line = tmp_line.replace(":16", " ")
                    if big_kernel_slr_id[idx] == 1 and "kernelBigGSMerger_1" in tmp_line and ":16" in tmp_line: 
                        tmp_line = tmp_line.replace(":16", " ")
                    f2.write(tmp_line)
            
            elif REGRAPH_ALL_KERNEL_ID_TEXT in line: # deal with apply kernels
                for idx in range(little_kernel_num+big_kernel_num):
                    tmp_line = line.replace(REGRAPH_ALL_KERNEL_ID_TEXT, str(idx+1))
                    if REGRAPH_APPLY_KERNEL_HBM_ID in tmp_line: 
                        tmp_line = tmp_line.replace(REGRAPH_APPLY_KERNEL_HBM_ID, str(wrapper_kernel_hbm_id[idx]))
                    f2.write(tmp_line)
            
            # # little kernel nums
            # elif REGRAPH_LITTLE_KERNEL_NUM in line:
            #     tmp_line = line.replace(REGRAPH_LITTLE_KERNEL_NUM, str(little_kernel_num))
            #     f2.write(tmp_line)
            
            # # big kernel nums
            # elif REGRAPH_BIG_KERNEL_NUM in line: #REGRAPH_BIG_KERNEL_NUM
            #     tmp_line = line.replace(REGRAPH_BIG_KERNEL_NUM, str(big_kernel_num))
            #     f2.write(tmp_line)

            # little kernel nums
            elif REGRAPH_LITTLE_KERNEL_ID_ARRAY in line:
                for idx in range(little_kernel_num):
                    tmp_line = line.replace(REGRAPH_LITTLE_KERNEL_ID_ARRAY, str(idx))
                    f2.write(tmp_line)
            
            # big kernel nums
            elif REGRAPH_BIG_KERNEL_ID_ARRAY in line: #REGRAPH_BIG_KERNEL_NUM
                for idx in range(big_kernel_num):
                    tmp_line = line.replace(REGRAPH_BIG_KERNEL_ID_ARRAY, str(idx))
                    f2.write(tmp_line)
            
            elif REGRAPH_HAVE_APPLY_OUTDEG in line: #REGRAPH_HAVE_APPLY_OUTDEG
                tmp_line = line.replace(REGRAPH_HAVE_APPLY_OUTDEG, "")
                if(have_apply):
                    f2.write(tmp_line)
                else:
                    f2.write("\n")

            elif REGRAPH_HAVE_VERTEX_PROP in line: #REGRAPH_HAVE_APPLY_OUTDEG
                tmp_line = line.replace(REGRAPH_HAVE_VERTEX_PROP, "")
                if(have_prop):
                    f2.write(tmp_line)
                else:
                    f2.write("\n")
            
            elif REGRAPH_HAVE_LITTLE_KERNEL in line:
                if(little_kernel_num):
                    tmp_line = line.replace(REGRAPH_HAVE_LITTLE_KERNEL, "")
                    if REGRAPH_LITTLE_KERNEL_NUM in line: #REGRAPH_LITTLE_KERNEL_NUM
                        tmp_line = tmp_line.replace(REGRAPH_LITTLE_KERNEL_NUM, str(little_kernel_num))
                    f2.write(tmp_line)

            elif REGRAPH_HAVE_BIG_KERNEL in line:
                if(big_kernel_num):
                    tmp_line = line.replace(REGRAPH_HAVE_BIG_KERNEL, "")
                    if REGRAPH_BIG_KERNEL_NUM in line: #REGRAPH_BIG_KERNEL_NUM
                        tmp_line = tmp_line.replace(REGRAPH_BIG_KERNEL_NUM, str(big_kernel_num))
                    f2.write(tmp_line)
                    
            else: # deal with normal lines
                f2.write(line)


#供主机程序使用的HBM通道映射表
def gen_hbm_config(hbm_config_file, little_kernel_num, big_kernel_num):

    global wrapper_kernel_hbm_id
    global little_kernel_hbm_id
    global big_kernel_hbm_id

    with open(hbm_config_file, "w", encoding="utf-8") as f1:
        f1.write("int little_kernel_hbm_mapping["+str(little_kernel_num+1)+"] = {")
        for idx in range(little_kernel_num):
            f1.write(str(little_kernel_hbm_id[idx])+", ")
        f1.write("0};\n")

        f1.write("int big_kernel_hbm_mapping["+str(big_kernel_num+1)+"] = {")
        for idx in range(big_kernel_num):
            f1.write(str(big_kernel_hbm_id[idx])+", ")
        f1.write("0};\n")

        f1.write("int apply_kernel_hbm_mapping["+str(little_kernel_num+big_kernel_num+1)+"] = {")
        for idx in range(little_kernel_num+big_kernel_num):
            f1.write(str(wrapper_kernel_hbm_id[idx])+", ")
        f1.write("0};\n")



#文件夹中剩余文件均为该脚本生成
def main():
    if len(sys.argv) < 2:
        raise Exception("Little kernel num missing!")

    if len(sys.argv) < 3:
        raise Exception("Big kernel num missing!")

    if len(sys.argv) < 4:
        raise Exception("HAVE_APPLY_OUTDEG missing!")

    if len(sys.argv) < 5:
        raise Exception("HAVE_VERTEX_PROP missing!")

    little_kernel_num = int(sys.argv[1])
    big_kernel_num    = int(sys.argv[2])
    have_apply        = sys.argv[3] == "true"
    have_prop         = sys.argv[4] == "true"
    assign_big_little(little_kernel_num, big_kernel_num)

    print("Template generation started!")

    # files: test
    # gen_kernel_files("./autogen/kernel_little_gs_merger_template.cpp", "./autogen/kernel_little_gs_merger.cpp", little_kernel_num, big_kernel_num, have_apply)
    # gen_kernel_files("./autogen/kernel_big_gs_merger_template.cpp", "./autogen/kernel_big_gs_merger.cpp", little_kernel_num, big_kernel_num, have_apply)
    # gen_kernel_files("./autogen/hbm_wrapper_template.h", "./autogen/hbm_wrapper.h", little_kernel_num, big_kernel_num, have_apply)
    # gen_kernel_files("./autogen/kernel_hbm_wrapper_template.cpp", "./autogen/kernel_hbm_wrapper.cpp", little_kernel_num, big_kernel_num, have_apply)
    # gen_kernel_files("./autogen/connectivity_template.cfg", "./autogen/connectivity.cfg", little_kernel_num, big_kernel_num, have_apply)
    # gen_hbm_config("./autogen/hbm_mapping.h", little_kernel_num, big_kernel_num)

    # files: compile
    gen_kernel_files("./autogen/kernel_little_gs_merger_template.cpp", "./acc_template/kernel_little_gs_merger/kernel_little_gs_merger.cpp", little_kernel_num, big_kernel_num, have_apply, have_prop)
    gen_kernel_files("./autogen/kernel_big_gs_merger_template.cpp", "./acc_template/kernel_big_gs_merger/kernel_big_gs_merger.cpp", little_kernel_num, big_kernel_num, have_apply, have_prop)
    gen_kernel_files("./autogen/hbm_wrapper_template.h", "./acc_template/kernel_hbm_wrapper/hbm_wrapper.h", little_kernel_num, big_kernel_num, have_apply, have_prop)
    gen_kernel_files("./autogen/kernel_hbm_wrapper_template.cpp", "./acc_template/kernel_hbm_wrapper/kernel_hbm_wrapper.cpp", little_kernel_num, big_kernel_num, have_apply, have_prop)
    gen_kernel_files("./autogen/connectivity_template.cfg", "./acc_template/connectivity.cfg", little_kernel_num, big_kernel_num, have_apply, have_prop)
    gen_hbm_config("./host/host_config/hbm_mapping.h", little_kernel_num, big_kernel_num)
    
    print("Template generation finished!")

if __name__ == "__main__":
    main()

