#ifndef GRAPH_CUH
#define GRAPH_CUH


#include "globals.hpp"

class GraphStructure
{
public:
	uint num_nodes;
	uint num_edges;
	uint* nodePointer;
	OutEdge* edgeList;
	uint* outDegree;
	uint* d_outDegree;
	string GetFileExtension(string fileName);
	void ReadGraph(string graphFilePath);
	void ReadGraphFromGraph(GraphStructure G);
	void FreeGraph();
};

template <class valueType>
class GraphStates
{
public:
	uint* edgeWeight;
	bool* label1;
	bool* label2;
	valueType* value;
	uint* sigma;
	float* delta;
	bool* d_label1;
	bool* d_label2;
	valueType* d_value;
	uint* d_sigma;
	float* d_delta;
	GraphStates();
	GraphStates(uint num_nodes, bool hasLabel, bool hasDelta, bool hasSigma = false, uint num_edgeWeight = 0);
	void ReadEdgeWeight(string weightFilePath, uint num_edgeWeight);
	void ReadEdgeWeightFromStates(GraphStates<valueType> S, uint num_edgeWeight);
	void FreeStates();
};

#endif	//	GRAPH_CUH



