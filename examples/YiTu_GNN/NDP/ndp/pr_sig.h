#include "../../shared/globals.hpp"
#include "../../shared/graph.cuh"
#include "../../shared/argument_parsing.cuh"
extern "C" {
    void pr_sig_sync(ArgumentParser arguments);
    void pr_sig_async(ArgumentParser arguments);
}
