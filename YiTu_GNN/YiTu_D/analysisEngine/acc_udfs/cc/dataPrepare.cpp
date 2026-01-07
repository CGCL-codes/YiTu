#include "host_config.h"
#include "host_data_types.h"
#include "fpga_application.h"
#include "graph.h"

#include <cstdlib>
#include <iostream>
#include <ctime>

// unsigned int dataPrepareGetArg(graphInfo *info)
// {
//     return 0;
// }

int initializeProperty(CSR* csr){

    std::srand(std::time(nullptr));

    for (int u = 0; u < csr->vertexNum; u++)
    {
        csr->vProps.push_back(0);
    }

    for (int u = 0; u < 32; u++)
    {
        int select_index = u;//((double)std::rand())/((RAND_MAX + 1u)/csr->vertexNum);
        csr->vProps[select_index] = 1 << u;
    }

    return 1;
}