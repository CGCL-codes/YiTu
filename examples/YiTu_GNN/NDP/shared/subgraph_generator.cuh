#ifndef SUBGRAPH_GENERATOR_HPP
#define SUBGRAPH_GENERATOR_HPP


#include "globals.hpp"
#include "graph.cuh"
#include "subgraph.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thread>

template <class valueType>
class SubgraphGenerator
{
private:

public:
	unsigned int *activeNodesLabeling;
	unsigned int *activeNodesDegree;
	unsigned int *prefixLabeling;
	unsigned int *prefixSumDegrees;
	unsigned int *d_activeNodesLabeling;
	unsigned int *d_activeNodesDegree;
	unsigned int *d_prefixLabeling;
	unsigned int *d_prefixSumDegrees;
	SubgraphGenerator(GraphStructure& graph);
	void generate(GraphStructure& graph, GraphStates<valueType>& states, Subgraph& subgraph, float acc = -1);
	void FreeSubgraphGenerator();
};

#endif	//	SUBGRAPH_GENERATOR_HPP



