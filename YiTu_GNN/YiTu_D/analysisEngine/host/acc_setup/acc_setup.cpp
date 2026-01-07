#include "acc_setup.h"

/*
初始化FPGA加速器环境
1.设备发现与选择
2.二进制文件加载
3.OpenCL环境设置
4.内核创建
5.错误处理
*/
acc_descriptor_dt initAccelerator(std::string xcl_file) {
    cl_int err;
    acc_descriptor_dt acc;
    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();
    // read_binary_file() command will find the OpenCL binary file created using
    // V++ compiler load into OpenCL Binary and return pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(xcl_file);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, acc.context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, acc.q = cl::CommandQueue(acc.context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        acc.big_gs_queue.resize(acc.num_big_krnl);
        for (int k = 0; k < acc.num_big_krnl; k ++){
            cl::CommandQueue tmp_q;
            OCL_CHECK(err, tmp_q = cl::CommandQueue(acc.context, device, CL_QUEUE_PROFILING_ENABLE, &err));
            acc.big_gs_queue[k] = tmp_q;
        }

        acc.little_gs_queue.resize(acc.num_little_krnl);
        for (int k = 0; k < acc.num_little_krnl; k ++){
            cl::CommandQueue tmp_q;
            OCL_CHECK(err, tmp_q = cl::CommandQueue(acc.context, device, CL_QUEUE_PROFILING_ENABLE, &err));
            acc.little_gs_queue[k] = tmp_q;
        }
        
        OCL_CHECK(err, acc.apply_queue = cl::CommandQueue(acc.context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, acc.hbm_queue = cl::CommandQueue(acc.context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(acc.context, {device}, bins, nullptr, &err);

        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } 
        else 
        {
            std::cout << "Device[" << i << "]: program successful!\n";
            
            //create kernels....
            for (int i = 0; i < acc.num_big_krnl; i++) {
                std::string cu_id = std::to_string(i + 1);
                std::string krnl_name_full = acc.big_gs_kernel_name + ":{" + "bigKernelScatterGather_" + cu_id + "}";
                
                cl::Kernel tmp_gs_krnl;
                printf("Creating a big kernel [%s] for CU(%d)\n", krnl_name_full.c_str(), i + 1);
                OCL_CHECK(err, tmp_gs_krnl = cl::Kernel(program, krnl_name_full.c_str(), &err));
                acc.big_gs_krnls.push_back(tmp_gs_krnl);
            }

            for (int i = 0; i < acc.num_little_krnl; i++) {
                std::string cu_id = std::to_string(i + 1);
                std::string krnl_name_full = acc.little_gs_kernel_name + ":{" + "littleKernelScatterGather_" + cu_id + "}";
                
                cl::Kernel tmp_gs_krnl;
                printf("Creating a little kernel [%s] for CU(%d)\n", krnl_name_full.c_str(), i + 1);
                OCL_CHECK(err, tmp_gs_krnl = cl::Kernel(program, krnl_name_full.c_str(), &err));
                acc.little_gs_krnls.push_back(tmp_gs_krnl);
            }
        
            //create the apply kernel
            std::string apply_krnl_name_full = acc.apply_kernel_name + ":{" + "kernelApply_1}";
            printf("Creating the apply kernel [%s] \n", apply_krnl_name_full.c_str());
            OCL_CHECK(err, acc.apply_krnl = cl::Kernel(program, apply_krnl_name_full.c_str(), &err));

            //create the hbm kernel
            std::string hbm_krnl_name_full = acc.hbm_kernel_name + ":{" + "kernelHBMWrapper_1}";
            printf("Creating the hbm kernel [%s] \n", hbm_krnl_name_full.c_str());
            OCL_CHECK(err, acc.hbm_krnl = cl::Kernel(program, hbm_krnl_name_full.c_str(), &err));

            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    return acc;
}