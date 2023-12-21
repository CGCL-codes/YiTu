#include "subgraph_generator.cuh"
#include "graph.cuh"
#include "subgraph.cuh"
#include "gpu_error_check.cuh"

const unsigned int NUM_THREADS = 64;

const unsigned int THRESHOLD_THREAD = 50000;

__global__ void prePrefix(unsigned int *activeNodesLabeling, unsigned int *activeNodesDegree, 
							unsigned int *outDegree, bool *label1, bool *label2, unsigned int numNodes)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numNodes){
		activeNodesLabeling[id] = label1[id] || label2[id]; // label1 is always zero in sync
		//activeNodesLabeling[id] = label[id];
		//activeNodesLabeling[id] = 1;
		activeNodesDegree[id] = 0;
		if(activeNodesLabeling[id] == 1)
			activeNodesDegree[id] = outDegree[id];	
	}	
}

__global__ void prePrefix(unsigned int *activeNodesLabeling, unsigned int *activeNodesDegree, 
							unsigned int *outDegree, float *delta, unsigned int numNodes, float acc)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numNodes){
		if(delta[id] > acc)
		{
			activeNodesLabeling[id] = 1;
		}
		else
		{
			activeNodesLabeling[id] = 0;
		}
		activeNodesDegree[id] = 0;
		if(activeNodesLabeling[id] == 1)
			activeNodesDegree[id] = outDegree[id];	
	}
}

__global__ void makeQueue(unsigned int *activeNodes, unsigned int *activeNodesLabeling,
							unsigned int *prefixLabeling, unsigned int numNodes)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numNodes && activeNodesLabeling[id] == 1){
		activeNodes[prefixLabeling[id]] = id;
	}
}

__global__ void makeActiveNodesPointer(unsigned int *activeNodesPointer, unsigned int *activeNodesLabeling, 
											unsigned int *prefixLabeling, unsigned int *prefixSumDegrees, 
											unsigned int numNodes)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < numNodes && activeNodesLabeling[id] == 1){
		activeNodesPointer[prefixLabeling[id]] = prefixSumDegrees[id];
	}
}

// pthread
void dynamic(unsigned int tId,
				unsigned int numThreads,	
				unsigned int numActiveNodes,
				unsigned int *activeNodes,
				unsigned int *outDegree, 
				unsigned int *activeNodesPointer,
				unsigned int *nodePointer, 
				OutEdge *activeEdgeList,
				OutEdge *edgeList,
				uint* activeWeightList = NULL,
				uint* weightList = NULL)
{

	unsigned int chunkSize = numActiveNodes / numThreads;
	if(numActiveNodes % numThreads != 0)
	{
		chunkSize++;
	}
	unsigned int left, right;
	left = tId * chunkSize;
	right = min(left+chunkSize, numActiveNodes);	
	
	unsigned int thisNode;
	unsigned int thisDegree;
	unsigned int fromHere;
	unsigned int fromThere;

	for(unsigned int i=left; i<right; i++)
	{
		thisNode = activeNodes[i];
		thisDegree = outDegree[thisNode];
		fromHere = activeNodesPointer[i];
		fromThere = nodePointer[thisNode];
		for(unsigned int j=0; j<thisDegree; j++)
		{
			activeEdgeList[fromHere+j] = edgeList[fromThere+j];
			if (weightList != NULL)
				activeWeightList[fromHere + j] = weightList[fromThere + j];
		}
	}
	
}

template <class valueType>
SubgraphGenerator<valueType>::SubgraphGenerator(GraphStructure& graph)
{
	gpuErrorcheck(cudaMallocHost(&activeNodesLabeling, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocHost(&activeNodesDegree, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocHost(&prefixLabeling, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMallocHost(&prefixSumDegrees, (graph.num_nodes+1) * sizeof(unsigned int)));
	
	gpuErrorcheck(cudaMalloc(&d_activeNodesLabeling, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_activeNodesDegree, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_prefixLabeling, graph.num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_prefixSumDegrees , (graph.num_nodes+1) * sizeof(unsigned int)));
}

template <class valueType>
void SubgraphGenerator<valueType>::FreeSubgraphGenerator()
{
	gpuErrorcheck(cudaFree(d_activeNodesLabeling));
	gpuErrorcheck(cudaFree(d_activeNodesDegree));
	gpuErrorcheck(cudaFree(d_prefixLabeling));
	gpuErrorcheck(cudaFree(d_prefixSumDegrees));
	gpuErrorcheck(cudaFreeHost(activeNodesLabeling));
	gpuErrorcheck(cudaFreeHost(activeNodesDegree));
	gpuErrorcheck(cudaFreeHost(prefixLabeling));
	gpuErrorcheck(cudaFreeHost(prefixSumDegrees));
}

template <class valueType>
void SubgraphGenerator<valueType>::generate(GraphStructure& graph, GraphStates<valueType>& states, Subgraph& subgraph, float acc)
{
	//std::chrono::time_point<std::chrono::system_clock> startDynG, finishDynG;
	//startDynG = std::chrono::system_clock::now();
	
	if (acc == -1)
		prePrefix << <graph.num_nodes / 512 + 1, 512 >> > (d_activeNodesLabeling, d_activeNodesDegree, graph.d_outDegree, states.d_label1, states.d_label2, graph.num_nodes);
	else
		prePrefix << <graph.num_nodes / 512 + 1, 512 >> > (d_activeNodesLabeling, d_activeNodesDegree, graph.d_outDegree, states.d_delta, graph.num_nodes, acc);
		
	thrust::device_ptr<unsigned int> ptr_labeling(d_activeNodesLabeling);
	thrust::device_ptr<unsigned int> ptr_labeling_prefixsum(d_prefixLabeling);
	
	subgraph.numActiveNodes = thrust::reduce(ptr_labeling, ptr_labeling + graph.num_nodes);
	//cout << "Number of Active Nodes = " << subgraph.numActiveNodes << endl;
				
	thrust::exclusive_scan(ptr_labeling, ptr_labeling + graph.num_nodes, ptr_labeling_prefixsum);
	
	makeQueue<<<graph.num_nodes/512+1, 512>>>(subgraph.d_activeNodes, d_activeNodesLabeling, d_prefixLabeling, graph.num_nodes);
	
	gpuErrorcheck(cudaMemcpy(subgraph.activeNodes, subgraph.d_activeNodes, subgraph.numActiveNodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	thrust::device_ptr<unsigned int> ptr_degrees(d_activeNodesDegree);
	thrust::device_ptr<unsigned int> ptr_degrees_prefixsum(d_prefixSumDegrees);
	
	thrust::exclusive_scan(ptr_degrees, ptr_degrees + graph.num_nodes, ptr_degrees_prefixsum);
	
	makeActiveNodesPointer<<<graph.num_nodes/512+1, 512>>>(subgraph.d_activeNodesPointer, d_activeNodesLabeling, d_prefixLabeling, d_prefixSumDegrees, graph.num_nodes);
	gpuErrorcheck(cudaMemcpy(subgraph.activeNodesPointer, subgraph.d_activeNodesPointer, subgraph.numActiveNodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	unsigned int numActiveEdges = 0;
	if(subgraph.numActiveNodes>0)
		numActiveEdges = subgraph.activeNodesPointer[subgraph.numActiveNodes-1] + graph.outDegree[subgraph.activeNodes[subgraph.numActiveNodes-1]];	
	
	unsigned int last = numActiveEdges;
	gpuErrorcheck(cudaMemcpy(subgraph.d_activeNodesPointer+subgraph.numActiveNodes, &last, sizeof(unsigned int), cudaMemcpyHostToDevice));
	
	gpuErrorcheck(cudaMemcpy(subgraph.activeNodesPointer, subgraph.d_activeNodesPointer, (subgraph.numActiveNodes+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	
	//finishDynG = std::chrono::system_clock::now();
	//std::chrono::duration<double> elapsed_seconds_dyng = finishDynG-startDynG;
	//std::time_t finish_time_dyng = std::chrono::system_clock::to_time_t(finishDynG);
	//std::cout << "Dynamic GPU Time = " << elapsed_seconds_dyng.count() << std::endl;
	
	//td::chrono::time_point<std::chrono::system_clock> startDynC, finishDynC;
	//startDynC = std::chrono::system_clock::now();
	
	unsigned int numThreads = NUM_THREADS;

	if(subgraph.numActiveNodes < THRESHOLD_THREAD)
		numThreads = 1;

	thread runThreads[numThreads];
	
	for(unsigned int t=0; t<numThreads; t++)
	{

		runThreads[t] = thread(dynamic,
								t,
								numThreads,
								subgraph.numActiveNodes,
								subgraph.activeNodes,
								graph.outDegree, 
								subgraph.activeNodesPointer,
								graph.nodePointer, 
								subgraph.activeEdgeList,
								graph.edgeList,
								subgraph.activeWeightList,
								states.edgeWeight);

	}
		
	for(unsigned int t=0; t<numThreads; t++)
		runThreads[t].join();
	
	//finishDynC = std::chrono::system_clock::now();
	//std::chrono::duration<double> elapsed_seconds_dync = finishDynC-startDynC;
	//std::time_t finish_time_dync = std::chrono::system_clock::to_time_t(finishDynC);
	//std::cout << "Dynamic CPU Time = " << elapsed_seconds_dync.count() << std::endl;
	
}

template class SubgraphGenerator<uint>;
template class SubgraphGenerator<float>;

