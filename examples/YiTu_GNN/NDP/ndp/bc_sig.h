#include "../../shared/globals.hpp"
#include "../../shared/graph.cuh"
#include "../../shared/argument_parsing.cuh"
extern "C" {
    void bc_sig_sync(ArgumentParser arguments);
    void bc_sig_async(ArgumentParser arguments);
}
