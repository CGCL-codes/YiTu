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


int main(int argc, char **argv)
{
	cudaFree(0);
	ArgumentParser arguments(argc, argv, true, false);

	Timer timer;
	timer.Start();

	GraphStructure graph;
	graph.ReadGraph(arguments.input);

	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime / 1000 << " (s).\n";

	GraphStates<uint> states(graph.num_nodes, true, false, false);

	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.value[i] = DIST_INFINITY;
		states.label1[i] = true;
		states.label2[i] = false;
	}
	states.value[arguments.sourceNode] = 0;
	//graph.label[arguments.sourceNode] = true;

	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_label2, states.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	Subgraph subgraph(graph.num_nodes, graph.num_edges);

	SubgraphGenerator<uint> subgen(graph);

	subgen.generate(graph, states, subgraph);

	for (unsigned int i = 0; i < graph.num_nodes; i++)
	{
		states.label1[i] = false;
	}
	states.label1[arguments.sourceNode] = true;
	gpuErrorcheck(cudaMemcpy(states.d_label1, states.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));

	Partitioner partitioner;

	timer.Start();

	unsigned int gItr = 0;

	bool finished;
	bool *d_finished;
	bool all_finished;
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	
	partitioner.partition(subgraph, subgraph.numActiveNodes);
	
		// a super iteration
	uint level = 0;
	do
	{
		all_finished = true;
		uint itr = 0;//分区迭代数
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			//cudaDeviceSynchronize();

			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			mixLabels<<<partitioner.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph.d_activeNodes, states.d_label1, states.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			do
			{
				itr++;
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
			
				bfs_async<<<partitioner.partitionNodeSize[i] / 512 + 1, 512>>>(partitioner.partitionNodeSize[i],
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
				gpuErrorcheck( cudaPeekAtLastError() );
				
				gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
				if(!finished) all_finished = false;
			} while(!finished);
			
		//subgen.generate(graph, subgraph);//根据活跃顶点重新生成子图
		}
		//cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << level << endl;
		level++;
	}while(!(all_finished));
	//cout << level << endl;
	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime / 1000 << " (s).\n";
	cout << "partition number:" << partitioner.numPartitions << endl;
	gpuErrorcheck(cudaMemcpy(states.value, states.d_value, graph.num_nodes * sizeof(uint), cudaMemcpyDeviceToHost));

	utilities::PrintResults(states.value, min(100, graph.num_nodes));

	if (arguments.hasOutput)
		utilities::SaveResults(arguments.output, states.value, graph.num_nodes);
}
