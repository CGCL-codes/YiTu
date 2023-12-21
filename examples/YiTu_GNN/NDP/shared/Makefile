CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60


all: timer.o argument_parsing.o graph.o subgraph.o partitioner.o subgraph_generator.o gpu_kernels.o subway_utilities.o test.o


timer.o: timer.cpp
	$(CC) -c timer.cpp -o timer.o $(CFLAGS)
	
argument_parsing.o: argument_parsing.cu
	$(NC) -c argument_parsing.cu -o argument_parsing.o $(CFLAGS) $(NFLAGS)

graph.o: graph.cu
	$(NC) -c graph.cu -o graph.o $(CFLAGS) $(NFLAGS)

subgraph.o: subgraph.cu
	$(NC) -c subgraph.cu -o subgraph.o $(CFLAGS) $(NFLAGS)

partitioner.o: partitioner.cu
	$(NC) -c partitioner.cu -o partitioner.o $(CFLAGS) $(NFLAGS)

subgraph_generator.o: subgraph_generator.cu
	$(NC) -c subgraph_generator.cu -o subgraph_generator.o $(CFLAGS) $(NFLAGS)
	
gpu_kernels.o: gpu_kernels.cu
	$(NC) -c gpu_kernels.cu -o gpu_kernels.o $(CFLAGS) $(NFLAGS)

subway_utilities.o: subway_utilities.cpp
	$(CC) -c subway_utilities.cpp -o subway_utilities.o $(CFLAGS)

test.o: test.cu
	$(NC) -c test.cu -o test.o $(CFLAGS) $(NFLAGS)

clean:
	rm *.o
