#ifndef __ACC_SETUP_H__
#define __ACC_SETUP_H__

#include "xcl2.hpp"
#include "host_data_types.h"
#include "host_config.h"

acc_descriptor_dt initAccelerator(std::string xcl_file);

#endif
