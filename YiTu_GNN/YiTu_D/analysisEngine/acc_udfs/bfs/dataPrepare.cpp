#include "host_config.h"
#include "host_data_types.h"
#include "fpga_application.h"
#include "graph.h"

#include <cstdlib>
#include <iostream>
#include <ctime>

// unsigned int dataPrepareGetArg(graphInfo *info)
// {
// 	return 0;
// }

int initializeProperty(CSR* csr){

	for (int u = 0; u < csr->vertexNum; u++)
	{
		csr->vProps.push_back(MAX_PROP);
	}

	int select_index  = ((double)std::rand()) / ((RAND_MAX + 1u) / csr->vertexNum);
	csr->vProps[select_index] = 0x80000001;

	return 1;
}