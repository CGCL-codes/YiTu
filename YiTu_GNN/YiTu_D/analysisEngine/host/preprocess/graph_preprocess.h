#ifndef __GRAPH_PREPROCESS_H__
#define __GRAPH_PREPROCESS_H__

#include "xcl2.hpp"
#include "host_data_types.h"
#include "host_config.h"
#include "graph.h"


void reorderGraph(CSR* csr);

int initializeProperty(CSR* csr);

partition_container_dt partitionGraph (CSR* csr);

void partitionDensePartitions(std::vector<std::vector<uint, aligned_allocator<uint> > > &part_edge_array_host,\
                                partition_container_dt &partition_container
                            );

#endif
