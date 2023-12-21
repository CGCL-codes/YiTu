#include "../../shared/globals.hpp"
#include "../../shared/graph.cuh"
#include "../../shared/argument_parsing.cuh"
extern "C" {
    void cc_sig_sync(ArgumentParser arguments);
    void cc_sig_async(ArgumentParser arguments);
}
