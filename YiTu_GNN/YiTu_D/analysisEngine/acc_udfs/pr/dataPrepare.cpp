#include "host_config.h"
#include "host_data_types.h"
#include "fpga_application.h"
#include "graph.h"
#include <cmath>

#define INT2FLOAT                   (pow(2,30))

int float2int(float a) {
    return (int)(a * INT2FLOAT);
}

float int2float(int a) {
    return ((float)a / INT2FLOAT);
}

int initializeProperty(CSR* csr){

    int num_vertex = csr->vertexNum;
    std::vector<uint, aligned_allocator<uint> > out_degree_host(NUM_VERTEX_ALIGNED);

    float init_score_float = 1.0f / csr->vertexNum;
    int init_score_int = float2int(init_score_float);

    for (int u = 0; u < csr->vertexNum; u++) {
        out_degree_host[u] = csr->rpao[u + 1] - csr->rpao[u];
        if (out_degree_host[u] > 0) {
            csr->vProps.push_back(init_score_int / out_degree_host[u]);
        }
        else {
            csr->vProps.push_back(0);
        }
    }

    return 1;
}