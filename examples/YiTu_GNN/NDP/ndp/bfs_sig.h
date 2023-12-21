#include "../../shared/globals.hpp"
#include "../../shared/graph.cuh"
#include "../../shared/argument_parsing.cuh"
extern "C" {
    void bfs_sig_sync(ArgumentParser arguments);
    void bfs_sig_async(ArgumentParser arguments);
}
