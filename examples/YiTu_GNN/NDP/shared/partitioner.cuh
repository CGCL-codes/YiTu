#ifndef PARTITIONER_CUH
#define PARTITIONER_CUH


#include "globals.hpp"
#include "subgraph.cuh"

class Partitioner
{
private:

public:
	uint numPartitions;
	vector<uint> fromNode;
	vector<uint> fromEdge;
	vector<uint> partitionNodeSize;
	vector<uint> partitionEdgeSize;
	Partitioner();
    void partition(Subgraph &subgraph, uint numActiveNodes);
    void reset();
};

#endif	//	PARTITIONER_CUH



