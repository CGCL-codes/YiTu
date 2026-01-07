#ifndef __GRAPH_LOADER_H__
#define __GRAPH_LOADER_H__

#define checkStatus(str) {                          \
    if (status != 0 || status != CL_SUCCESS) {      \
        DEBUG_PRINTF("Error code: %d\n", status);   \
        DEBUG_PRINTF("Error: %s\n", str);           \
        acceleratorDeinit();                        \
        exit(-1);                                   \
    }                                               \
}

Graph* createGraph(const std::string &gName, const std::string &mode);
CSR* createCsr(Graph* gptr);

#endif /* __GRAPH_LOADER_H__ */