#include "../../shared/globals.hpp"
#include "../../shared/graph.cuh"
#include "../../shared/argument_parsing.cuh"
extern "C" {
    void sssp_sig_sync(ArgumentParser arguments);
    void sssp_sig_async(ArgumentParser arguments);
}
