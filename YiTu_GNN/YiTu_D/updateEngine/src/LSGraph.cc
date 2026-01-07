#define ENABLE_LOCK 1
#define WEIGHTED 0
#define VERIFY 0
#define DEL 1

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <queue>
#include "sys/time.h"

#include "LSGraph.h"
#include "parallel.h"
#include "util.h"
#include "parse_command_line.h"
#include "rmat_util.h"
#include "bench_utils.h"

#include "BFS.h"
#include "Pagerank.h"
#include "Components.h"
#include "BC.h"
#include "TC.h"
#include "io_util.h"

// #include "sim_api.h"

using namespace graphstore;

#define LOGGING_TICK (1ULL << 24)
#define BATCH_SIZE (1ULL << 9)

#include	<stdlib.h>

std::string test_name[] = {
    "BFS",
    "PR",
    "CC",
    "BC",
    "SSSP_BF",
};

template <class Graph>
double execute(Graph& G, commandLine& P, std::string testname, int i) {
  if (testname == "BFS") {
    return test_bfs(G, P,i );
  } else if (testname == "PR") {
    return test_pr(G, P);
  } else if (testname == "CC") {
    return test_cc(G, P);
  } else if (testname == "BC") {
    return test_bc(G, P);
  } else if (testname == "TC") {
    return test_tc(G, P);
  } else {
    std::cout << "Unknown test: " << testname << ". Quitting." << std::endl;
    exit(0);
  }
}

template <class G>
double test_pr(G& GA, commandLine& P) {
	struct timeval start, end;
	struct timezone tzp;
  long maxiters = P.getOptionLongValue("-maxiters",10);
  std::cout << "Running PR" << std::endl;

  // with edge map
  gettimeofday(&start, &tzp);
  auto pr_edge_map = PR_S<double>(GA, maxiters); 
  gettimeofday(&end, &tzp);
  free(pr_edge_map);
  PRINT("PR finished");
  return cal_time_elapsed(&start, &end);
}

template <class G>
double test_cc(G& GA, commandLine& P) {
	struct timeval start, end;
	struct timezone tzp;
  std::cout << "Running CC" << std::endl;
  // with edge map
  gettimeofday(&start, &tzp);
  auto cc_result = CC(GA);
  gettimeofday(&end, &tzp);
  free(cc_result);
  PRINT("CC finished");
  return cal_time_elapsed(&start, &end);
}

template <class G>
double test_bc(G& GA, commandLine& P) {
	struct timeval start, end;
	struct timezone tzp;

  long src = P.getOptionLongValue("-src",-1);
  if (src == -1) {
    std::cout << "Please specify a source vertex to run the BC from" << std::endl;
    exit(0);
  }
  std::cout << "Running BC from source = " << src << std::endl;

  // with edge map
  gettimeofday(&start, &tzp);
  auto bc_result = BC(GA, src);
  gettimeofday(&end, &tzp);
  free(bc_result);
  PRINT("BC finished");
  return cal_time_elapsed(&start, &end);
}

template <class G>
double test_tc(G& GA, commandLine& P) {
	struct timeval start, end;
	struct timezone tzp;

  std::cout << "Running TC" << std::endl;

  // with edge map
  gettimeofday(&start, &tzp);
  std::vector<std::vector<uint32_t>>mp(GA.get_num_vertices());
  parallel_for(int i = 0; i < GA.get_num_vertices(); ++i){
    std::vector<uint32_t>nei;
    GA.print(i, nei);
    mp[i] = nei;
  }
  auto count = TC(GA, mp);
  gettimeofday(&end, &tzp);
  printf("TC finished, counted %ld\n", count);
  return cal_time_elapsed(&start, &end);
}


// return time elapsed
template <class G>
double test_bfs(G& GA, commandLine& P, int trial) {
	struct timeval start, end;
	struct timezone tzp;

  long src = P.getOptionLongValue("-src",-1);
  if (src == -1) {
    std::cout << "Please specify a source vertex to run the BFS from" << std::endl;
    exit(0);
  }
  std::cout << "Running BFS from source = " << src << std::endl;
  // with edge map
  gettimeofday(&start, &tzp);
  auto bfs_edge_map = BFS_with_edge_map(GA, src);
  gettimeofday(&end, &tzp);
  free(bfs_edge_map);
  return cal_time_elapsed(&start, &end);
}

/*
P:参考main函数的调用方式：commandLine P(argc, argv, "./graph_bm [-t testname -r rounds -f file");
getGraph：传入数组，函数会把每个id的顶点的邻居写入该数组
*/
void get_neighbors(commandLine& P, std::vector<std::vector<uint32_t>> &getGraph){
  auto test_id = P.getOptionValue("-t", "BFS");
  size_t rounds = P.getOptionLongValue("-bs", 5);
  //size_t rounds = P.getOptionLongValue("-bs", 10000);
  
	std::string src, dest;
	// read edges as source and destination
	//int cnt = 0;
	struct timeval start, end;
	struct timezone tzp;

  // initialize graph
	uint32_t num_nodes;
  uint64_t num_edges;
  auto filename = P.getOptionValue("-f", "none");
	//pair_uint *edges = get_edges_from_file(filename.c_str(), 1,true, &num_edges, &num_nodes);
	pair_uint *edges = get_edges_from_file_adj_sym(filename.c_str(), &num_edges, &num_nodes);

	LSGraph graph(num_nodes);
    std::vector<uint32_t> new_srcs(num_edges);
    std::vector<uint32_t> new_dests(num_edges);
             for (uint64_t i = 0; i < num_edges; i++) {
               new_srcs[i] = edges[i].x;
               new_dests[i] = edges[i].y;
       }
       // auto perm = get_random_permutation(num_edges);
  std::cout<<"checkpoint4"<<std::endl;
  std::vector<uint32_t> perm;
	PRINT("Inserting edges");	
	gettimeofday(&start, &tzp);
	graph.bulk_load(new_srcs.data(), new_dests.data(), num_edges);
	gettimeofday(&end, &tzp);

  //get neighbors of the graph
  struct timeval testStart, testEnd;
	struct timezone tmp;
  std::vector<uint32_t> neig;
  gettimeofday(&testStart, &tmp);
  for(uint32_t i = 0; i < num_nodes; i ++){
    neig.clear();
    // 先添加 in-place neighbors
    uint32_t deg = graph.degree(i);
    uint32_t inplace_deg = std::min(deg, (uint32_t)NUM_IN_PLACE_NEIGHBORS);
    for (uint32_t j = 0; j < inplace_deg; j++) {
        neig.push_back(graph.vertices[i].neighbors[j]);
    }
    // 再添加 aux neighbors
    if (graph.has_aux(i)) {
        graph.print_hitree(i, neig);
    }
    getGraph.push_back(neig);
  }
  gettimeofday(&testEnd, &tmp);
  print_time_elapsed("Getting neighbors: ", &testStart, &testEnd);
  
  //save the data of edges
  std::string testFilename = "test.mtx";
  std::ofstream outFile(testFilename);
  if (!outFile.is_open()) {
    throw std::runtime_error("无法打开文件: " + testFilename);
  }
  for (size_t i = 0; i < getGraph.size(); i++) {
    // 输出边的信息
    for (size_t j = 0; j < getGraph[i].size(); j++) {
      outFile << i << " " << getGraph[i][j] << "\n";
    }
  }
  outFile.close();

  parallel_for(uint32_t i = 0; i < num_edges; i++) {
    if (!graph.is_edge(edges[i].x, edges[i].y) && edges[i].x != edges[i].y) {
      printf("edge (%u, %u) not found, should be\n", edges[i].x, edges[i].y);
      printf("\tdegree %u = %u\n", edges[i].x, graph.degree(edges[i].x));
    }
  }

	free(edges);
	       new_srcs.clear();
       new_dests.clear();
	print_time_elapsed("Inserting edges: ", &start, &end);
	PRINT("Throughput: " <<
				graph.get_num_edges()/(float)cal_time_elapsed(&start, &end));
#if 0
  std::vector<std::string> test_ids;
  // if testname is TC, include it, otherwise exclude it
  if (P.getOptionLongValue("-TC",0) != 0) {
    test_ids = {"BFS","PR","CC","BC", "TC"};
  } else {
    test_ids = {"BFS", "PR", "CC", "BC"};
  }
  for (auto test_id : test_ids) {
    double total_time = 0.0;
    for (size_t i=0; i<rounds; i++) {
      double tm = execute(graph, P, test_id, i);

      // std::cout << "RESULT"  << fixed << setprecision(6)
      std::cout << "\ttest=" << test_id
       << "\ttime=" << tm
       << "\titeration=" << i << std::endl;
      total_time += tm;
    }
    // std::cout << "RESULT (AVG)" << fixed << setprecision(6)
    std::cout << "AVG"
      << "\ttest=" << test_id
      << "\ttime=" << (total_time / rounds)
      << "\tgraph=" << filename << std::endl;
  }
  return;
#endif
  /** insert bm below here **/
  // std::vector<uint32_t> update_sizes = {10, 100, 1000 ,10000,100000,1000000,10000000};
  uint32_t batch_size = static_cast<uint32_t>(rounds);
  std::vector<uint32_t> update_sizes = {batch_size};
  auto r = random_aspen();
  auto update_times = std::vector<double>();
  size_t n_trials = 1;
  for (size_t us=0; us<update_sizes.size(); us++) {
    double avg_insert = 0;
    double avg_delete = 0;
    std::cout << "Running bs: " << update_sizes[us] << std::endl;

    if (update_sizes[us] <= 10000000) {
      n_trials = 20;
    }
    else {
      n_trials = 5;
    }
    // n_trials = 1;
    size_t updates_to_run = update_sizes[us];
		// auto perm = get_random_permutation(updates_to_run);
    for (size_t ts=0; ts<n_trials; ts++) {
      uint32_t num_nodes = graph.get_num_vertices();

      std::vector<uint32_t> new_srcs(updates_to_run);
      std::vector<uint32_t> new_dests(updates_to_run);
      double a = 0.5;
      double b = 0.1;
      double c = 0.1;
      size_t nn = 1 << (log2_up(num_nodes) - 1);
      // auto rmat = rMat<uint32_t>(nn, r.ith_rand(100+ts), a, b, c);
      auto rmat = rMat<uint32_t>(nn, r.ith_rand(ts), a, b, c);

      auto my_updates = parlay::sequence<std::tuple<uint32_t, uint32_t>>(updates_to_run);
      parallel_for(auto i = 0; i < updates_to_run; i++) {
        my_updates[i] = rmat(i);
      }
      gettimeofday(&start, &tzp);
      graph.add_edge_batch_sort(my_updates, updates_to_run, nn);
      gettimeofday(&end, &tzp);
      avg_insert += cal_time_elapsed(&start, &end);
     // printf("%lld\n", graph.get_index());
     
#if DEL
      gettimeofday(&start, &tzp);
      parallel_for(uint32_t i = 0; i < updates_to_run; i++) {
        graph.remove_edge(get<0>(my_updates[i]), get<1>(my_updates[i]));
      }
      gettimeofday(&end, &tzp);
      avg_delete +=  cal_time_elapsed(&start, &end);
#endif
    }
    double time_i = (double) avg_insert / n_trials;
    double insert_throughput = updates_to_run / time_i;
    printf("batch_size = %zu, average insert: %f, throughput %e\n", updates_to_run, time_i, insert_throughput);
    
#if DEL
    double time_d = (double) avg_delete / n_trials;
    double delete_throughput = updates_to_run / time_d;
    printf("batch_size = %zu, average delete: %f, throughput %e\n", updates_to_run, time_d, delete_throughput);
#endif    
  }
}

void run_algorithm(commandLine& P) {
  auto test_id = P.getOptionValue("-t", "BFS");
  size_t rounds = P.getOptionLongValue("-bs", 5);
  //size_t rounds = P.getOptionLongValue("-bs", 10000);
  
	std::string src, dest;
	// read edges as source and destination
	//int cnt = 0;
	struct timeval start, end;
	struct timezone tzp;

  // initialize graph
	uint32_t num_nodes;
  uint64_t num_edges;
  auto filename = P.getOptionValue("-f", "none");
	//pair_uint *edges = get_edges_from_file(filename.c_str(), 1,true, &num_edges, &num_nodes);

	pair_uint *edges = get_edges_from_file_adj_sym(filename.c_str(), &num_edges, &num_nodes);
  
	LSGraph graph(num_nodes);
    std::vector<uint32_t> new_srcs(num_edges);
    std::vector<uint32_t> new_dests(num_edges);
             for (uint64_t i = 0; i < num_edges; i++) {
               new_srcs[i] = edges[i].x;
               new_dests[i] = edges[i].y;
       }
       // auto perm = get_random_permutation(num_edges);
  std::cout<<"checkpoint4"<<std::endl;
  std::vector<uint32_t> perm;
	PRINT("Inserting edges");	
	gettimeofday(&start, &tzp);
	graph.bulk_load(new_srcs.data(), new_dests.data(), num_edges);
	gettimeofday(&end, &tzp);
  parallel_for(uint64_t i = 0; i < num_edges; i++) {
    if (!graph.is_edge(edges[i].x, edges[i].y) && edges[i].x != edges[i].y) {
      printf("edge (%u, %u) not found, should be\n", edges[i].x, edges[i].y);
      printf("\tdegree %u = %u\n", edges[i].x, graph.degree(edges[i].x));
    }
  }

	free(edges);
	       new_srcs.clear();
       new_dests.clear();
	print_time_elapsed("Inserting edges: ", &start, &end);
	PRINT("Throughput: " <<
				graph.get_num_edges()/(float)cal_time_elapsed(&start, &end));
#if 0
  std::vector<std::string> test_ids;
  // if testname is TC, include it, otherwise exclude it
  if (P.getOptionLongValue("-TC",0) != 0) {
    test_ids = {"BFS","PR","CC","BC", "TC"};
  } else {
    test_ids = {"BFS", "PR", "CC", "BC"};
  }
  for (auto test_id : test_ids) {
    double total_time = 0.0;
    for (size_t i=0; i<rounds; i++) {
      double tm = execute(graph, P, test_id, i);

      // std::cout << "RESULT"  << fixed << setprecision(6)
      std::cout << "\ttest=" << test_id
       << "\ttime=" << tm
       << "\titeration=" << i << std::endl;
      total_time += tm;
    }
    // std::cout << "RESULT (AVG)" << fixed << setprecision(6)
    std::cout << "AVG"
      << "\ttest=" << test_id
      << "\ttime=" << (total_time / rounds)
      << "\tgraph=" << filename << std::endl;
  }
  return;
#endif
  /** insert bm below here **/
  // std::vector<uint32_t> update_sizes = {10, 100, 1000 ,10000,100000,1000000,10000000};
  uint32_t batch_size = static_cast<uint32_t>(num_edges*0.001);
  std::vector<uint32_t> update_sizes = {batch_size};
  auto r = random_aspen();
  auto update_times = std::vector<double>();
  size_t n_trials = 1;
  for (size_t us=0; us<update_sizes.size(); us++) {
    double avg_insert = 0;
    double avg_delete = 0;
    std::cout << "Running bs: " << update_sizes[us] << std::endl;

    if (update_sizes[us] <= 10000000) {
      n_trials = 20;
    }
    else {
      n_trials = 5;
    }
    // n_trials = 1;
    size_t updates_to_run = update_sizes[us];
		// auto perm = get_random_permutation(updates_to_run);
    for (size_t ts=0; ts<n_trials; ts++) {
      uint32_t num_nodes = graph.get_num_vertices();

      std::vector<uint32_t> new_srcs(updates_to_run);
      std::vector<uint32_t> new_dests(updates_to_run);
      double a = 0.5;
      double b = 0.1;
      double c = 0.1;
      size_t nn = 1 << (log2_up(num_nodes) - 1);
      // auto rmat = rMat<uint32_t>(nn, r.ith_rand(100+ts), a, b, c);
      auto rmat = rMat<uint32_t>(nn, r.ith_rand(ts), a, b, c);

      auto my_updates = parlay::sequence<std::tuple<uint32_t, uint32_t>>(updates_to_run);
      parallel_for(auto i = 0; i < updates_to_run; i++) {
        my_updates[i] = rmat(i);
      }
      gettimeofday(&start, &tzp);
      graph.add_edge_batch_sort(my_updates, updates_to_run, nn);
      gettimeofday(&end, &tzp);
      avg_insert += cal_time_elapsed(&start, &end);
     // printf("%lld\n", graph.get_index());
     
#if DEL
      gettimeofday(&start, &tzp);
      parallel_for(uint32_t i = 0; i < updates_to_run; i++) {
        graph.remove_edge(get<0>(my_updates[i]), get<1>(my_updates[i]));
      }
      gettimeofday(&end, &tzp);
      avg_delete +=  cal_time_elapsed(&start, &end);
#endif
    }
    double time_i = (double) avg_insert / n_trials;
    double insert_throughput = updates_to_run / time_i;
    printf("batch_size = %zu, average insert: %f, throughput %e\n", updates_to_run, time_i, insert_throughput);
    
#if DEL
    double time_d = (double) avg_delete / n_trials;
    double delete_throughput = updates_to_run / time_d;
    printf("batch_size = %zu, average delete: %f, throughput %e\n", updates_to_run, time_d, delete_throughput);
#endif    
  }
}
 

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  main
 *  Description:  
 * ============================================================================
 */
int main(int argc, char** argv) {
  srand(time(NULL));
  printf("Num workers: %ld\n", getWorkers());
  commandLine P(argc, argv, "./graph_bm [-t testname -r rounds -f file");
  //run_algorithm(P);
  
  std::vector<std::vector<uint32_t>> getGraph;
  get_neighbors(P, getGraph);
  
}
/*
	int
main ( int argc, char *argv[] )
{
	srand(time(NULL));
#if 0
    uint32_t num_nodesPMA = 100;
	 PMA pma = PMA(num_nodesPMA);
	 for (int i = 0; i < 50; i++) {
		 for (int j = 0; j < 100; j++) {
        uint32_t y = rand()%num_nodesPMA;
        //printf("adding edge (%u, %u\n", i, y);
			 pma.add_edge(i,y,1);
       //pma.print_array();
       if (rand()%5 == 0) {
        //printf("removing edge (%u, %u\n", i, y);
        pma.remove_edge(i,y);
        //pma.print_array();
       }
		 }
	 }
   //return 0;


	 for (int i = 0; i < 10; i++) {
		 std::cout << "Neighbors for node: " << i << " ";
		 for (auto it = pma.begin(i); it != pma.end(i); ++it) {
			 std::cout << (*it).dest << " ";
		 }
		 std::cout << "\n";
	 }

	 //pma.print_array(0);

#else
	
	if (argc < 3) {
		fprintf(stderr, "Please specify the name of the input file and node id\n");
		exit(1);
	}

	std::string src, dest;
	// read edges as source and destination
	//int cnt = 0;
	struct timeval start, end;
	struct timezone tzp;

	uint32_t num_nodes;
  	uint64_t num_edges;
	pair_uint *edges = get_edges_from_file(argv[1], 1,true, &num_edges, &num_nodes);

	Graph graph(num_nodes);
	//std::random_shuffle(edgelist.begin(), edgelist.end());

	PRINT("Inserting edges");	
	gettimeofday(&start, &tzp);

	for (uint64_t i = 0; i < num_edges; i++) {
		graph.add_edge(edges[i].x, edges[i].y);
	}
	gettimeofday(&end, &tzp);
	free(edges);

	PRINT("Construction finished. Nodes: " << graph.get_num_vertices() <<
				" Edges: " << graph.get_num_edges());
	print_time_elapsed("Inserting edges: ", &start, &end);
	PRINT("Throughput: " <<
				graph.get_num_edges()/(float)cal_time_elapsed(&start, &end));

  PRINT("Starting BFS");
  gettimeofday(&start, &tzp);
  auto bfs_result = BFS_with_edge_map(graph, atoi(argv[2]));
  //pvector<int32_t> bfs_result = parallel_bfs(graph, atoi(argv[2]), num_edges);
  //std::vector<int32_t> bfs_result = bfs(graph, atoi(argv[2]));
  gettimeofday(&end, &tzp);
  PRINT("BFS finished. parent of 0: " << bfs_result[0]);
  print_time_elapsed("BFS traversal: ", &start, &end);
#if 0
	PRINT("Starting BFS");
	auto it = graph.find();
	uint64_t total = 0;
	gettimeofday(&start, &tzp);
	while (!it.done()) {
		total++;
		++it;
	}
	gettimeofday(&end, &tzp);
	PRINT("BFS finished. Nodes: " << total);
	print_time_elapsed("BFS traversal: ", &start, &end);
#endif

#endif
	return EXIT_SUCCESS;
}*/				/* ----------  end of function main  ---------- */
