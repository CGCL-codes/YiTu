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
#include "../shared/test.cuh"
#include "../shared/test.cu"


int main(int argc, char** argv)
{
	
	cudaFree(0);

	ArgumentParser arguments(argc, argv, true, false);
	// cpu
	if (arguments.deviceID == -1)
	{
		std::string cmd = "./apps/PageRank " + arguments.input;
		FILE *pp = popen(cmd.data(), "r"); // build pipe
		if (!pp)
			return 1;
		// collect cmd execute result
		char tmp[1024];
		while (fgets(tmp, sizeof(tmp) * 1024, pp) != NULL)
			std::cout << tmp << std::endl; // can join each line as string
		pclose(pp);
		return 1;
	}
	Timer timer;
	timer.Start();
	
	GraphStructure graph;
	graph.ReadGraph(arguments.input);
	
	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
	
	//for(unsigned int i=0; i<100; i++)
	//	cout << graph.edgeList[i].end << " " << graph.edgeList[i].w8;
	GraphStates<float> states(graph.num_nodes, false, true, false);
	
	float initPR = 0.15;
	float acc = 0.01;
	
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		states.delta[i] = initPR;
		states.value[i] = 0;
	}
	//graph.value[arguments.sourceNode] = 0;
	//graph.label[arguments.sourceNode] = true;


	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_value, states.value, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(states.d_delta, states.delta, graph.num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	
	Subgraph subgraph(graph.num_nodes, graph.num_edges);
	
	SubgraphGenerator<float> subgen(graph);
	
	subgen.generate(graph, states, subgraph, acc);	

	Partitioner partitioner;
	
	timer.Start();
	
	uint gItr = 0;
	
	bool finished;
	bool *d_finished;
	bool all_finished;
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
		
	do
	{
		all_finished = true;
		gItr++;
		
		partitioner.partition(subgraph, subgraph.numActiveNodes);
		// a super iteration
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();

			//moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			//mixLabels<<<partitioner.partitionNodeSize[i]/512 + 1 , 512>>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);
			
			uint itr = 0;
			do
			{
				itr++;
				finished = true;
				gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

				pr_async<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
													partitioner.fromNode[i],
													partitioner.fromEdge[i],
													subgraph.d_activeNodes,
													subgraph.d_activeNodesPointer,
													subgraph.d_activeEdgeList,
													graph.d_outDegree,
													states.d_value,
													states.d_delta,
													d_finished,
													acc);		
																						

				cudaDeviceSynchronize();
				gpuErrorcheck( cudaPeekAtLastError() );	
				
				gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
				if(!finished) all_finished = false;
			}while(!(finished));
			
			//cout << itr << ((itr>1) ? " Inner Iterations" : " Inner Iteration") << " in Global Iteration " << gItr << ", Partition " << i  << endl;			
		}
		
		//subgen.generate(graph, states, subgraph, acc);
	}	while(!(all_finished));
	
	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";
	
	gpuErrorcheck(cudaMemcpy(states.value, states.d_value, graph.num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
	
	utilities::PrintResults(states.value, min(30, graph.num_nodes));
	
		
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, states.value, graph.num_nodes);
}

