#include <cstdlib>

#include "../shared/globals.hpp"
#include "../shared/timer.hpp"
#include "../shared/argument_parsing.cuh"
#include "../shared/graph.cuh"
#include "../shared/subgraph.cuh"
#include "../shared/partitioner.cuh"
#include "../shared/subgraph_generator.cuh"
#include "../shared/gpu_error_check.cuh"
#include "../shared/gpu_kernels.cuh"
#include "../shared/subway_utilities.hpp"


__global__ void bc_w(unsigned int numNodes,
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
							int level)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(dist[id] != level)
			return;
			
		unsigned int sourceWeight = dist[id];

		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			
			finalDist = sourceWeight + 1;
			if(finalDist < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , level + 1);

				*finished = false;
			}
			if(dist[edgeList[i].end] == finalDist ) {
				atomicAdd(&sigma[edgeList[i].end] , sigma[id]);
			}
		}
	}
}


int main(int argc, char** argv)
{
	
	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);
	
	Timer timer;
	timer.Start();
	
	GraphStructure graph;
	graph.ReadGraph(arguments.input);
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
	
	GraphStates<uint> states(graph.num_nodes, true, true, true);
	
	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.value[i] = DIST_INFINITY;
		states.sigma[i] = 0;
		states.delta[i] = 0;
		states.label1[i] = true;
		states.label2[i] = false;
	}
	states.value[arguments.sourceNode] = 0;
	states.sigma[arguments.sourceNode] = 1;
	states.delta[arguments.sourceNode] = 0;


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_sigma, states.sigma, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_delta, states.delta, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	
	Subgraph subgraph(graph.num_nodes, graph.num_edges);
	SubgraphGenerator<uint> subgen(graph);
	
	subgen.generate(graph, states, subgraph);	
	for(unsigned int i=0; i<graph.num_nodes; i++)//仅将源顶点标记为活跃顶点
	{
		states.label1[i] = false;
	}
	states.label1[arguments.sourceNode] = true;
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	Partitioner partitioner;
	
	timer.Start();
	
	uint gItr = 0;
	unsigned int level = 0;
	unsigned int* d_level;
	bool finished;
	bool *d_finished;
	bool all_finished;
	gpuErrorcheck(cudaMalloc(&d_level, sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	partitioner.partition(subgraph, subgraph.numActiveNodes);

	do
	{
		gItr++;
		all_finished = true;
		uint itr = 0;//分区迭代数
		//partitioner.partition(subgraph, subgraph.numActiveNodes);
		//cout << "partition number: " << partitioner.numPartitions << endl;
		// a super iteration
		for (int i = 0; i < partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			//cudaDeviceSynchronize();

			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels << <partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (subgraph.d_activeNodes, states.d_label1, states.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			//uint itr = 0;
			do
			{
				itr++;
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
				bfs_async << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (partitioner.partitionNodeSize[i],
					partitioner.fromNode[i],
					partitioner.fromEdge[i],
					subgraph.d_activeNodes,
					subgraph.d_activeNodesPointer,
					subgraph.d_activeEdgeList,
					graph.d_outDegree,
					states.d_value,
					d_finished,
					(itr % 2 == 1) ? states.d_label1 : states.d_label2,
					(itr % 2 == 1) ? states.d_label2 : states.d_label1);
				cudaDeviceSynchronize();
				gpuErrorcheck(cudaPeekAtLastError());

				gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
				if (!finished) all_finished = false;
			} while (!(finished));
			//cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;
		}
		//subgen.generate(graph, states, subgraph);
	} while (!(all_finished));

	cudaDeviceSynchronize();
	gpuErrorcheck(cudaMemcpy(d_level, &level, sizeof(unsigned int), cudaMemcpyHostToDevice));
	find_max << < graph.num_nodes / 512 + 1, 512 >> > (graph.num_nodes,
		states.d_value,
		d_level);
	cudaDeviceSynchronize();
	gpuErrorcheck(cudaPeekAtLastError());
	gpuErrorcheck(cudaMemcpy(&level, d_level, sizeof(bool), cudaMemcpyDeviceToHost));
	level++;
	cout << level << endl;

	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.label1[i] = true;
		states.label2[i] = false;
	}
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	//subgen.generate(graph, states, subgraph);

	int start = 0;
	while (start < level) {
		//partitioner.partition(subgraph, subgraph.numActiveNodes);
		//cout<< level << "  " <<subgraph1.numActiveNodes<<" .   "<<partitioner1.numPartitions <<endl;
		for (int i = 0; i < partitioner.numPartitions; i++) {
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			//cudaDeviceSynchronize();
			//moveUpLabels<<<partitioner1.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph1.d_activeNodes, states.d_label1, states.d_label2, partitioner1.partitionNodeSize[i], partitioner1.fromNode[i]);

			bc_sigma_async << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (partitioner.partitionNodeSize[i],
				partitioner.fromNode[i],
				partitioner.fromEdge[i],
				subgraph.d_activeNodes,
				subgraph.d_activeNodesPointer,
				subgraph.d_activeEdgeList,
				graph.d_outDegree,
				states.d_value,
				states.d_sigma,
				start);

			gpuErrorcheck(cudaDeviceSynchronize());
			gpuErrorcheck(cudaPeekAtLastError());

		}
		start++;
	}

	//subgen.generate(graph, states, subgraph);

	while (level >= 1) {
		level--;

		//partitioner.partition(subgraph, subgraph.numActiveNodes);
		//cout<< level << "  " <<subgraph1.numActiveNodes<<" .   "<<partitioner1.numPartitions <<endl;
		for (int i = 0; i < partitioner.numPartitions; i++) {
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			//cudaDeviceSynchronize();
			//moveUpLabels<<<partitioner1.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph1.d_activeNodes, states.d_label1, states.d_label2, partitioner1.partitionNodeSize[i], partitioner1.fromNode[i]);

			bc_ndp << < partitioner.partitionNodeSize[i] / 512 + 1, 512 >> > (partitioner.partitionNodeSize[i],
				partitioner.fromNode[i],
				partitioner.fromEdge[i],
				subgraph.d_activeNodes,
				subgraph.d_activeNodesPointer,
				subgraph.d_activeEdgeList,
				graph.d_outDegree,
				states.d_value,
				states.d_sigma,
				states.d_delta,
				states.d_label1,
				states.d_label2,
				level);

			gpuErrorcheck(cudaDeviceSynchronize());
			gpuErrorcheck(cudaPeekAtLastError());

		}
		//subgen.generate(graph, states, subgraph);
	}

	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime / 1000 << " (s).\n";

	gpuErrorcheck(cudaMemcpy(states.value, states.d_value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	gpuErrorcheck(cudaMemcpy(states.sigma, states.d_sigma, graph.num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	gpuErrorcheck(cudaMemcpy(states.delta, states.d_delta, graph.num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
	utilities::PrintResults(states.delta, min(100, graph.num_nodes));


	if (arguments.hasOutput)
		utilities::SaveResults(arguments.output, states.delta, graph.num_nodes);
}

