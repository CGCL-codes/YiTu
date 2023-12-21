
#include "gpu_kernels.cuh"
#include "globals.hpp"
#include "gpu_error_check.cuh"
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
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int sourceWeight = value[id];

		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			finalDist = sourceWeight + 1;
			if(finalDist < value[edgeList[i].end])
			{
				atomicMin(&value[edgeList[i].end] , finalDist);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

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
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		//unsigned int finalDist;
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			if(sourceWeight < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , sourceWeight);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}


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
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int sourceWeight = dist[id];

		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			finalDist = sourceWeight + weightList[i];
			if(finalDist < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , finalDist);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

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
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			finalDist = min(sourceWeight, weightList[i]);
			if(finalDist > dist[edgeList[i].end])
			{
				atomicMax(&dist[edgeList[i].end] , finalDist);

				//*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

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
							float acc)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		unsigned int degree = outDegree[id];
		float thisDelta = delta[id];

		if(thisDelta > acc)
		{
			dist[id] += thisDelta;
			
			if(degree != 0)
			{
				//*finished = false;
				
				float sourcePR = ((float) thisDelta / degree) * 0.85;

				unsigned int thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;
				unsigned int thisto = thisfrom + degree;
				
				for(unsigned int i=thisfrom; i<thisto; i++)
				{
					atomicAdd(&delta[edgeList[i].end], sourcePR);
				}				
			}
			
			atomicAdd(&delta[id], -thisDelta);
		}
		
	}
}
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
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId < numNodes) 
	{
		unsigned int id = activeNodes[from + tId];

		if(label1[id] != true)
			return;
        label1[id] = false;

		unsigned int sourceWeight = dist[id];
		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		unsigned int finalDist;
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{
			finalDist = sourceWeight + 1;
			if(finalDist < dist[edgeList[i].end])
			{

				atomicMin(&dist[edgeList[i].end] , finalDist);
				
				//*finished = false;

				label2[edgeList[i].end] = true;
			}
			if(dist[edgeList[i].end] == finalDist ) {
				atomicAdd(&sigma[edgeList[i].end] , sigma[id]);
			}

		}
		
	}
}



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
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
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
				atomicMin(&dist[edgeList[i].end] , finalDist);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

__global__ void sssp_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int *activeNodes,
							unsigned int *activeNodesPointer,
							OutEdge *edgeList,
							uint *weightList,
							unsigned int *outDegree,
							unsigned int *dist,
							bool *finished,
							bool *label1,
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		unsigned int finalDist;
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			finalDist = sourceWeight + weightList[i];
			if(finalDist < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , finalDist);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}

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
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;
		
		unsigned int sourceWeight = dist[id];

		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		
		
		unsigned int finalDist;
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			finalDist = min(sourceWeight, weightList[i]);
			if(finalDist > dist[edgeList[i].end])
			{
				atomicMax(&dist[edgeList[i].end] , finalDist);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}


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
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		
		if(label1[id] == false)
			return;
			
		label1[id] = false;

		unsigned int sourceWeight = dist[id];

		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		
		//printf("******* %i\n", thisFrom);
		
		//unsigned int finalDist;
		
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{	
			//finalDist = sourceWeight + edgeList[i].w8;
			if(sourceWeight < dist[edgeList[i].end])
			{
				atomicMin(&dist[edgeList[i].end] , sourceWeight);

				*finished = false;
				
				//label1[edgeList[i].end] = true;

				label2[edgeList[i].end] = true;
			}
		}
	}
}


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
							float acc)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if(tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		unsigned int degree = outDegree[id];
		float thisDelta = delta[id];

		if(thisDelta > acc)
		{
			dist[id] += thisDelta;
			
			if(degree != 0)
			{
				*finished = false;
				
				float sourcePR = ((float) thisDelta / degree) * 0.85;

				unsigned int thisfrom = activeNodesPointer[from+tId]-numPartitionedEdges;
				unsigned int thisto = thisfrom + degree;
				
				for(unsigned int i=thisfrom; i<thisto; i++)
				{
					atomicAdd(&delta[edgeList[i].end], sourcePR);
				}				
			}
			
			atomicAdd(&delta[id], -thisDelta);
		}
		
	}
}

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
										bool* isInternalNode)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];
		unsigned int degree = outDegree[id];
		float thisDelta = delta[id];

		if (isInternalNode[id] && thisDelta > acc)
		{
			dist[id] += thisDelta;

			if (degree != 0)
			{
				*finished = false;

				float sourcePR = ((float)thisDelta / degree) * 0.85;

				unsigned int thisfrom = activeNodesPointer[from + tId] - numPartitionedEdges;
				unsigned int thisto = thisfrom + degree;

				for (unsigned int i = thisfrom; i < thisto; i++)
				{
					atomicAdd(&delta[edgeList[i].end], sourcePR);
				}
			}

			atomicAdd(&delta[id], -thisDelta);
		}

	}
}

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
							bool *label2)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId < numNodes) 
	{
		unsigned int id = activeNodes[from + tId];

		if(label1[id] != true)
			return;
        label1[id] = false;

		unsigned int sourceWeight = dist[id];
		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		unsigned int finalDist;
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{
			finalDist = sourceWeight + 1;
			if(finalDist < dist[edgeList[i].end])
			{

				atomicMin(&dist[edgeList[i].end] , finalDist);
				
				*finished = false;

				label2[edgeList[i].end] = true;
			}
			if(dist[edgeList[i].end] == finalDist ) {
				atomicAdd(&sigma[edgeList[i].end] , sigma[id]);
			}

		}
		
	}
}


__global__ void find_max(unsigned int numNodes,
							unsigned int *dist,
							unsigned int *max_level)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId < numNodes) 
	{
		if(*max_level < dist[tId] && dist[tId] != DIST_INFINITY)
			atomicMax(max_level , dist[tId]);
	}
}

__global__ void bc_sigma_async(unsigned int numNodes,
							unsigned int from,
							unsigned int numPartitionedEdges,
							unsigned int* activeNodes,
							unsigned int* activeNodesPointer,
							OutEdge* edgeList,
							unsigned int* outDegree,
							unsigned int* dist,
							unsigned int* sigma,
							int level)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < numNodes)
	{
		unsigned int id = activeNodes[from + tId];

		if (dist[id] != level)
			return;
		unsigned int sourceWeight = dist[id];
		unsigned int thisFrom = activeNodesPointer[from + tId] - numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		unsigned int finalDist = sourceWeight + 1;
		for (unsigned int i = thisFrom; i < thisTo; i++)
		{
			if (dist[edgeList[i].end] == finalDist) {
				//label2[edgeList[i].end] = true;
				atomicAdd(&sigma[edgeList[i].end], sigma[id]);
			}
		}

	}
}

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
							int level)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId < numNodes) 
	{
		unsigned int id = activeNodes[from + tId];
		if(dist[id] != level){
			return;
		}
		label1[id] = false;
		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{
			if(dist[edgeList[i].end] == level + 1) {
				bc[id] = bc[id] + ((float)sigma[id]/sigma[edgeList[i].end]) * (bc[edgeList[i].end] + 1);
			}
		}
		
	}
}


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
							int level)
{
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId < numNodes) 
	{
		unsigned int id = activeNodes[from + tId];

		if(dist[id] != level)
			return;

		unsigned int thisFrom = activeNodesPointer[from+tId]-numPartitionedEdges;
		unsigned int degree = outDegree[id];
		unsigned int thisTo = thisFrom + degree;
		for(unsigned int i=thisFrom; i<thisTo; i++)
		{
			if(dist[edgeList[i].end] == level + 1) {
				bc[id] = bc[id] + ((float)sigma[id]/sigma[edgeList[i].end]) * (bc[edgeList[i].end] + 1);
			}
		}
		
	}
}
__global__ void clearLabel(unsigned int * activeNodes, bool *label, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
	{
		label[activeNodes[id+from]] = false;
	}
}

__global__ void mixLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size){
		int nID = activeNodes[id+from];
		label1[nID] = label1[nID] || label2[nID];
		label2[nID] = false;	
	}
}

__global__ void moveUpLabels(unsigned int * activeNodes, bool *label1, bool *label2, unsigned int size, unsigned int from)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int nID;
	if(id < size){
		nID = activeNodes[id+from];
		label1[nID] = label2[nID];
		label2[nID] = false;	
	}
}

