
#include "globals.hpp"
#include "graph.cuh"
#include "subgraph.cuh"


__global__ void bfs_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *value,
							//bool *finished,
							bool *label1,
							bool *label2);

__global__ void cc_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2);

__global__ void sssp_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							uint* weightList,
							unsigned int *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2);
							
__global__ void sswp_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							uint* weightList,
							unsigned int *outDegree,
							unsigned int *dist,
							//bool *finished,
							bool *label1,
							bool *label2);
							
__global__ void pr_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							float *dist,
							float *delta,
							//bool *finished,
							float acc);						

__global__ void bc_kernel(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							unsigned int *sigma,
							float *bc,
							//bool *finished,
							bool *label1,
							bool *label2);

__global__ void bfs_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2);	
							
__global__ void sssp_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							uint* weightList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2);
							
__global__ void sswp_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							uint* weightList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2);
							
__global__ void cc_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2);		
							
__global__ void pr_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							float *dist,
							float *delta,
							bool *finished,
							float acc);	

__global__ void pr_async_distributed(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int* activeNodes,
							unsigned int* activeNodesPointer,
							OutEdge* edgeList,
							unsigned int* outDegree,
							float* dist,
							float* delta,
							bool* finished,
							float acc,
							bool* isInternalNode);

__global__ void bc_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							unsigned int *sigma,
							float *bc,
							bool *finished,
							bool *label1,
							bool *label2);

__global__ void bc_sigma_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int* activeNodes,
							unsigned int* activeNodesPointer,
							OutEdge* edgeList,
							unsigned int* outDegree,
							unsigned int* dist,
							unsigned int* sigma,
							int level);

__global__ void bc_ndp(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							unsigned int *sigma,
							float *bc,
							bool *label1,
							bool *label2,
							int level);

__global__ void bc(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							unsigned int *outDegree,
							unsigned int *dist,
							unsigned int *sigma,
							float *bc,
							int level);

__global__ void find_max(unsigned int numNodes,
							unsigned int *dist,
							unsigned int *max_level);

__global__ void clearLabel(unsigned int * activeNodes, bool *label, unsigned int size, unsigned int from);

__global__ void mixLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from);

__global__ void moveUpLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from);


