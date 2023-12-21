#include "graph.cuh"
#include "gpu_error_check.cuh"

string GraphStructure::GetFileExtension(string fileName)
{
	if (fileName.find_last_of(".") != string::npos)
		return fileName.substr(fileName.find_last_of(".") + 1);
	return "";
}

void GraphStructure::ReadGraph(string graphFilePath)
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;

	string graphFormat = GetFileExtension(graphFilePath);

	if (graphFormat == "bcsr" || graphFormat == "bwcsr")
	{
		ifstream infile(graphFilePath, ios::in | ios::binary);

		infile.read((char*)&num_nodes, sizeof(uint));
		infile.read((char*)&num_edges, sizeof(uint));

		nodePointer = new uint[num_nodes + 1];
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(OutEdge)));

		infile.read((char*)nodePointer, sizeof(uint) * num_nodes);
		infile.read((char*)edgeList, sizeof(OutEdge) * num_edges);
		nodePointer[num_nodes] = num_edges;
	}
	else if (graphFormat == "el" || graphFormat == "wel")
	{
		ifstream infile;
		infile.open(graphFilePath);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;

		vector<Edge> edges;
		Edge newEdge;
		while (getline(infile, line))
		{
			ss.str("");
			ss.clear();
			ss << line;

			ss >> newEdge.source;
			ss >> newEdge.end;

			edges.push_back(newEdge);
			edgeCounter++;

			if (max < newEdge.source)
				max = newEdge.source;
			if (max < newEdge.end)
				max = newEdge.end;
		}
		infile.close();
		num_nodes = max + 1;
		num_edges = edgeCounter;
		nodePointer = new uint[num_nodes + 1];
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(OutEdge)));
		uint* degree = new uint[num_nodes];
		for (uint i = 0; i < num_nodes; i++)
			degree[i] = 0;
		for (uint i = 0; i < num_edges; i++)
			degree[edges[i].source]++;

		uint counter = 0;
		for (uint i = 0; i < num_nodes; i++)
		{
			nodePointer[i] = counter;
			counter = counter + degree[i];
		}
		nodePointer[num_nodes] = num_edges;
		uint* outDegreeCounter = new uint[num_nodes];
		uint location;
		for (uint i = 0; i < num_edges; i++)
		{
			location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
			edgeList[location].end = edges[i].end;
			//if(isWeighted)
			//	edgeList[location].w8 = edges[i].w8;
			outDegreeCounter[edges[i].source]++;
		}
		edges.clear();
		delete[] degree;
		delete[] outDegreeCounter;
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}

	outDegree = new unsigned int[num_nodes];

	for (uint i = 1; i < num_nodes; i++)
		outDegree[i - 1] = nodePointer[i] - nodePointer[i - 1];
	outDegree[num_nodes - 1] = num_edges - nodePointer[num_nodes - 1];

	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));

	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;
}

void GraphStructure::ReadGraphFromGraph(GraphStructure G)
{
	num_nodes = G.num_nodes;
	num_edges = G.num_edges;
	nodePointer = new uint[num_nodes + 1]{ 0 };
	gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(OutEdge)));
	for (uint i = 0; i < num_nodes; i++)
		nodePointer[i] = G.nodePointer[i];
	for (uint i = 0; i < num_edges; i++)
		edgeList[i] = G.edgeList[i];

	outDegree = new unsigned int[num_nodes];

	for (uint i = 1; i < num_nodes; i++)
		outDegree[i - 1] = nodePointer[i] - nodePointer[i - 1];
	outDegree[num_nodes - 1] = num_edges - nodePointer[num_nodes - 1];

	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));

	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;
}

void GraphStructure::FreeGraph()
{
	gpuErrorcheck(cudaFree(d_outDegree));
	gpuErrorcheck(cudaFreeHost(edgeList));
}

//--------------------------------------

template <class valueType>
GraphStates<valueType>::GraphStates() {}

template <class valueType>
GraphStates<valueType>::GraphStates(uint num_nodes, bool hasLabel, bool hasDelta, bool hasSigma, uint num_edgeWeight)
{
	edgeWeight = NULL;

	if (hasLabel)
	{
		label1 = new bool[num_nodes];
		label2 = new bool[num_nodes];
		gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
		gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));
	}
	else
	{
		label1 = NULL;
		label2 = NULL;
	}

	if (hasDelta)
	{
		delta = new float[num_nodes];
		gpuErrorcheck(cudaMalloc(&d_delta, num_nodes * sizeof(float)));
	}
	else
	{
		delta = NULL;
	}

	if (hasSigma)
	{
		sigma = new uint[num_nodes];
		gpuErrorcheck(cudaMalloc(&d_sigma, num_nodes * sizeof(uint)));
	}
	else
	{
		sigma = NULL;
	}

	value = new valueType[num_nodes];
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(valueType)));

}

template <class valueType>
void GraphStates<valueType>::ReadEdgeWeight(string weightFilePath, uint num_edgeWeight)
{
	ifstream infile(weightFilePath, ios::in | ios::binary);

	uint num_edges = 0;
	infile.read((char*)&num_edges, sizeof(uint));

	gpuErrorcheck(cudaMallocHost(&edgeWeight, (num_edges) * sizeof(uint)));

	infile.read((char*)edgeWeight, sizeof(uint) * num_edges);
	infile.close();
}

template <class valueType>
void GraphStates<valueType>::ReadEdgeWeightFromStates(GraphStates<valueType> S, uint num_edgeWeight)
{
	gpuErrorcheck(cudaMallocHost(&edgeWeight, (num_edgeWeight) * sizeof(uint)));
	for (uint i = 0; i < num_edgeWeight; ++i)
	{
		edgeWeight[i] = S.edgeWeight[i];
	}
}

template <class valueType>
void GraphStates<valueType>::FreeStates()
{
	if (edgeWeight != NULL)
	{
		gpuErrorcheck(cudaFreeHost(edgeWeight));
	}
	if (label1 != NULL)
	{
		gpuErrorcheck(cudaFree(d_label1));
	}
	if (label2 != NULL)
	{
		gpuErrorcheck(cudaFree(d_label2));
	}
	if (sigma != NULL) 
	{
		gpuErrorcheck(cudaFree(d_sigma));
	}
	if (delta != NULL)
	{
		gpuErrorcheck(cudaFree(d_delta));
	}
	gpuErrorcheck(cudaFree(d_value));
}

template class GraphStates<uint>;
template class GraphStates<float>;
