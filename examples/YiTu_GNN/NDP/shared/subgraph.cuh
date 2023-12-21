#ifndef SUBGRAPH_HPP
#define SUBGRAPH_HPP


#include "globals.hpp"


class Subgraph
{
private:

public:
	uint num_nodes;
	uint num_edges;
	uint numActiveNodes;
	
	uint *activeNodes;
	uint *activeNodesPointer;
	OutEdge *activeEdgeList;
	uint* activeWeightList;
	
	uint *d_activeNodes;
	uint *d_activeNodesPointer;
	OutEdge *d_activeEdgeList;
	uint* d_activeWeightList;
	
	ull max_partition_size;
	
	bool hasWeight;
	
	Subgraph(uint num_nodes, uint num_edges, bool hasEdgeWeight = false);
	void FreeSubgraph();
};

#endif	//	SUBGRAPH_HPP



