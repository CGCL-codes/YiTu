CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60

SHARED=../shared
APPS=../nondp

DEP=$(SHARED)/timer.o $(SHARED)/argument_parsing.o $(SHARED)/graph.o $(SHARED)/subgraph.o $(SHARED)/partitioner.o $(SHARED)/subgraph_generator.o $(SHARED)/gpu_kernels.o $(SHARED)/subway_utilities.o $(SHARED)/test.o  

all: make1 bfs-w cc-w bc-w pr-w sssp-w

make1:
	make -C $(SHARED)

bfs-w: $(APPS)/bfs-w.cu $(DEP)
	$(NC) -o bfs-w.o -c bfs-w.cu $(CFLAGS) $(NFLAGS)
	$(NC) $(APPS)/bfs-w.o $(DEP) -o bfs-w $(CFLAGS) $(NFLAGS)

bc-w: $(APPS)/bc-w.cu $(DEP)
	$(NC) -o bc-w.o -c bc-w.cu $(CFLAGS) $(NFLAGS)
	$(NC) $(APPS)/bc-w.o $(DEP) -o bc-w $(CFLAGS) $(NFLAGS)

cc-w: $(APPS)/cc-w.cu $(DEP)
	$(NC) -o cc-w.o -c cc-w.cu $(CFLAGS) $(NFLAGS)	
	$(NC) $(APPS)/cc-w.o $(DEP) -o cc-w $(CFLAGS) $(NFLAGS)

pr-w: $(APPS)/pr-w.cu $(DEP)
	$(NC) -o pr-w.o -c pr-w.cu $(CFLAGS) $(NFLAGS)
	$(NC) $(APPS)/pr-w.o $(DEP) -o pr-w $(CFLAGS) $(NFLAGS)
	
sssp-w: $(APPS)/sssp-w.cu $(DEP)
	$(NC) -o sssp-w.o -c sssp-w.cu $(CFLAGS) $(NFLAGS)
	$(NC) $(APPS)/sssp-w.o $(DEP) -o sssp-w $(CFLAGS) $(NFLAGS)	

.PHONY : clean		

clean:
	make -C $(SHARED) clean
	rm bfs-w cc-w bc-w pr-w sssp-w


