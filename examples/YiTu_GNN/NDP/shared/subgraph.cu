
#include "subgraph.cuh"
#include "gpu_error_check.cuh"
#include "graph.cuh"
#include <cuda_profiler_api.h>


Subgraph::Subgraph(uint num_nodes, uint num_edges, bool hasEdgeWeight)
{
	cudaProfilerStart();
	cudaError_t error;
	cudaDeviceProp dev;
	int deviceID;
	cudaGetDevice(&deviceID);
	error = cudaGetDeviceProperties(&dev, deviceID);
	if(error != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	cudaProfilerStop();
	
	this->hasWeight = hasEdgeWeight;
	if (hasEdgeWeight)
		max_partition_size = 0.9 * (dev.totalGlobalMem - 8 * 4 * num_nodes) / (sizeof(OutEdge) + sizeof(uint));
	else
		max_partition_size = 0.9 * (dev.totalGlobalMem - 8 * 4 * num_nodes) / sizeof(OutEdge);
	//max_partition_size = 1000000000;
	
	if(max_partition_size > DIST_INFINITY)
		max_partition_size = DIST_INFINITY;
	
	//cout << "Max Partition Size: " << max_partition_size << endl;
	
	this->num_nodes = num_nodes;
	this->num_edges = num_edges;
	
	gpuErrorcheck(cudaMallocHost(&activeNodes, num_nodes * sizeof(uint)));
	gpuErrorcheck(cudaMallocHost(&activeNodesPointer, (num_nodes+1) * sizeof(uint)));
	gpuErrorcheck(cudaMallocHost(&activeEdgeList, num_edges * sizeof(OutEdge)));
	
	gpuErrorcheck(cudaMalloc(&d_activeNodes, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_activeNodesPointer, (num_nodes+1) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_activeEdgeList, (max_partition_size) * sizeof(OutEdge)));
	
	if (hasEdgeWeight)
	{
		gpuErrorcheck(cudaMallocHost(&activeWeightList, num_edges * sizeof(uint)));
		gpuErrorcheck(cudaMalloc(&d_activeWeightList, (max_partition_size) * sizeof(uint)));
	}
}

void Subgraph::FreeSubgraph()
{
	gpuErrorcheck(cudaFree(d_activeNodes));
	gpuErrorcheck(cudaFree(d_activeNodesPointer));
	gpuErrorcheck(cudaFree(d_activeEdgeList));
	gpuErrorcheck(cudaFreeHost(activeNodes));
	gpuErrorcheck(cudaFreeHost(activeNodesPointer));
	gpuErrorcheck(cudaFreeHost(activeEdgeList));
	if (hasWeight)
	{
		gpuErrorcheck(cudaFree(d_activeWeightList));
		gpuErrorcheck(cudaFreeHost(activeWeightList));
	}
}
// For initialization with one active node
//unsigned int numActiveNodes = 1;
//subgraph.activeNodes[0] = SOURCE_NODE;
//for(unsigned int i=graph.nodePointer[SOURCE_NODE], j=0; i<graph.nodePointer[SOURCE_NODE] + graph.outDegree[SOURCE_NODE]; i++, j++)
//	subgraph.activeEdgeList[j] = graph.edgeList[i];
//subgraph.activeNodesPointer[0] = 0;
//subgraph.activeNodesPointer[1] = graph.outDegree[SOURCE_NODE];
//gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodes, subgraph.activeNodes, numActiveNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
//gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodesPointer, subgraph.activeNodesPointer, (numActiveNodes+1) * sizeof(unsigned int), cudaMemcpyHostToDevice));


