#ifndef __SCHEDULE_H__
#define __SCHEDULE_H__

#include "xcl2.hpp"
#include "host_data_types.h"
#include "host_config.h"
#include "graph.h"

int schedulePartitions(partition_container_dt &partition_container);
int transferPartitions(partition_container_dt &partition_container, acc_descriptor_dt &acc);

#endif
