#include "../../shared/globals.hpp"
#include "../../shared/graph.cuh"
#include "../../shared/argument_parsing.cuh"
extern "C" {
    void sswp_sig_sync(ArgumentParser arguments);
    void sswp_sig_async(ArgumentParser arguments);
}
