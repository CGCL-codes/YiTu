/*
 * ============================================================================
 *
 *       Filename:  graphalex.h
 *
 *
 * ============================================================================
 */

#ifndef _GRAPHALEX_H_
#define _GRAPHALEX_H_

#include <stdlib.h>
#include <algorithm>

#include <set>
#include <unordered_set>
#include <vector>
#include <iostream>

#include <parlay/io.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/sequence.h>

// #include <pthread.h>
// #include <thread>
// #include <unistd.h>
//
// #include <sys/syscall.h>  
// #define gettid() syscall(__NR_gettid)

#include <string.h>

#include "graph.h"
#include "../src/HITree/HITree.h"
#include "BitArray.h"
#include "util.h"
//#include "cpp-btree/btree_set.h"

namespace graphstore {



#define PREFETCH 1

class LSGraph {
    public:
      typedef uint32_t vertex;
      typedef uint32_t weight;
      typedef std::pair<vertex, vertex> edge;
      typedef hitreespace::HITree<vertex, weight> fl_container;
      // for backward compatibility. Will remove these typedefs in some time.
      typedef std::set<vertex> vertex_set;
      typedef std::set<edge> edge_set;
      typedef vertex_set::iterator vertex_set_iterator;
      LSGraph(uint32_t size); // create a graph with the given size (#num nodes)
      // LSGraph(uint32_t sizes, uint32_t size); // create a graph with the given size (#num nodes)
      LSGraph(std::string prefix); // read graph from disk
      ~LSGraph();

      void serialize(std::string prefix); // write graph to disk
			int remove_edge(const vertex s, const vertex d);
			void bulk_load(vertex *srcs,vertex *dests, uint64_t edge_count);
			void add_edge_batch_1for(vertex *srcs,vertex *dests, uint32_t edge_count);
			void add_edge_batch_sort(parlay::sequence<std::tuple<uint32_t, uint32_t>> &updates, uint32_t edge_count, size_t nn);
      // check for the existence of the edge
      uint32_t is_edge(const vertex s, const vertex d);

      void visit_edge(const vertex s);
      uint64_t get_index();
      void visit_edge_test(const vertex s, std::vector<uint32_t> &array);
      
      uint64_t collect_conflict(const vertex s);
      
      void print(const vertex s, std::vector<uint32_t>& nei);
      // get out neighbors of vertex s
      // NeighborIterator neighbors(const vertex v) const;
      // get out degree of vertex v
      uint32_t degree(const vertex v) const;
      uint64_t get_size(void);
  
      template <class F, typename VS>
      void map_sparse(F &f, VS &output_vs, uint32_t self_index, bool output);
      //template <class F, typename VS>
      //void map_dense(F &f, VS &vs, uint32_t self_index, bool output);
      template <class F, typename VS>
      void map_dense_vs_all(F &f, VS &vs, VS &output_vs, uint32_t self_index, bool output);
      template <class F, typename VS>
      void map_dense_vs_not_all(F &f, VS &vs, VS &output_vs, uint32_t self_index, bool output);

      uint32_t get_num_edges(void);
      uint32_t get_num_vertices(void) const;

      uint32_t count_common(const vertex s, const vertex d) const;
      void print_vertex_block(const vertex v) const;

    private:
    public:

      inline bool has_aux(const vertex s) const {
        return vertices[s].aux_neighbors != nullptr;
      }

      void print_hitree(const vertex s, std::vector<uint32_t>& nei) const {
		    fl_container *container = (fl_container*)(vertices[s].aux_neighbors);
        container->print_hitree(nei);  
      }
      bool check_in_place_dup(const vertex s, const vertex d) const;
     
  uint64_t get_imas(){
    uint64_t sum[4] = {0,0,0,0};
    uint64_t sums = 0;
    uint64_t sum2[4] = {0,0,0,0};
    uint64_t sum2s = 0;
    for (uint32_t i = 0; i < num_vertices; ++i) {
      // printf("%d\n", num_vertices);
      auto degree = vertices[i].degree & 0x7fffffff;
      // printf("loop %d\n", i);
      if(degree > NUM_IN_PLACE_NEIGHBORS){
        auto container = (fl_container*)(vertices[i].aux_neighbors);
        if (container == nullptr) printf("nullptr\n");
        if (degree < 100) {
          sum[0] += container->get_ima();
          sum2[0] += container->get_size();
        } else if (degree < 512) {
          sum[1] += container->get_ima();
          sum2[1] += container->get_size();
        } else if (degree < 1024) {
          sum[2] += container->get_ima();
          sum2[2] += container->get_size();
        } else {
          sum[3] += container->get_ima();
          sum2[3] += container->get_size();
        }
      }
    }
    for (int i = 0; i < 4; ++i) {
      printf("%lld\n", sum[i]);
      sums += sum[i];
    }
    printf("%lld\n", sums);
    for (int i = 0; i < 4; ++i) {
      printf("%lld\n", sum2[i]);
      sum2s += sum2[i];
    }
    printf("%lld\n", sum2s);
    return sums;
  }

      // Need to make this block cache line (64 Bytes) aligned. 
      typedef struct __attribute__ ((__packed__)) {
        uint32_t degree; // 4 Bytes
        vertex neighbors[NUM_IN_PLACE_NEIGHBORS]; // 4*13=52 Bytes
        void *aux_neighbors{nullptr};  // 8 Bytes
        } vertex_block;

      vertex_block *vertices;
      uint32_t num_vertices{0};
      uint64_t inplace_size{0};
  };

  LSGraph::LSGraph(uint32_t size) : num_vertices(size) {
    inplace_size = size * sizeof(vertex_block);
    vertices = (vertex_block*)calloc(size, sizeof(vertex_block));
  }

  LSGraph::~LSGraph() {
    free(vertices);
  }

  bool LSGraph::check_in_place_dup(const vertex s, const vertex d) const {
    for (uint32_t i = 0; i < NUM_IN_PLACE_NEIGHBORS; i++) {
      if (vertices[s].neighbors[i] == d)
        return true;
    }

    return false;
  }

  void sort_updates(parlay::sequence<std::tuple<uint32_t, uint32_t> > &edges, size_t m, size_t nn = std::numeric_limits<size_t>::max()) {
      // parlay::integer_sort_inplace(edges);
    using edge = std::tuple<uint32_t, uint32_t>;
    // auto E_orig = parlay::make_range(edges, edges + m);
    // auto E_orig = edges;
    size_t vtx_bits = parlay::log2_up(nn);
    // if (nn == std::numeric_limits<size_t>::max()) {
    //   auto max_edge_id = parlay::delayed_seq<size_t>(m, [&] (size_t i) { return std::max(std::get<0>(E_orig[i]),
    //         std::get<1>(E_orig[i])); });
    //   vtx_bits = parlay::log2_up(parlay::reduce(max_edge_id, parlay::maxm<size_t>()));
    //   nn = 1 << vtx_bits;
    // }

    // std::cout << vtx_bits << std::endl;

    auto edge_to_long = [vtx_bits] (edge e) -> size_t {
      return (static_cast<size_t>(std::get<0>(e)) << vtx_bits) | static_cast<size_t>(std::get<1>(e));
    };
    // auto edge_ids_log = parlay::delayed_seq<size_t>(m, [&] (size_t i) { return parlay::log2_up(edge_to_long(E_orig[i])); });
    // auto edge_ids_log = parlay::delayed_seq<size_t>(m, [&] (size_t i) { return parlay::log2_up(edge_to_long(edges[i])); });

    size_t bits = 2*vtx_bits;

    // Only apply integer sort if it will be work-efficient
    if (nn <= (m * parlay::log2_up(m))) {
      // cout << "running integer sort: " << nn << " and mm = " << (m * parlay::log2_up(m)) << " bits = " << bits << endl;
      // parlay::integer_sort_inplace(E_orig, edge_to_long, bits);
      // parlay::internal::integer_sort_inplace(parlay::make_slice(edges), edge_to_long, bits);
      //parlay::stable_integer_sort_inplace(edges, edge_to_long, bits); 新版parlay通过std::bit_width等方式自动估计键的最大位数，从而决定排序范围
      parlay::stable_integer_sort_inplace(edges, edge_to_long);
      // parlay::integer_sort_inplace(edges);
    } else {
      // cout << "running sample sort" << endl;
      // parlay::sample_sort_inplace(E_orig, std::less<edge>());
      parlay::stable_sort_inplace(edges, std::less<edge>());
    }
  }

  void LSGraph::add_edge_batch_sort(parlay::sequence<std::tuple<uint32_t, uint32_t>> &updates, uint32_t edge_count, size_t nn) {
    // struct timeval start, end;
    // struct timezone tzp;
    parlay::internal::timer t("Time");

    uint32_t threads = getWorkers();
    uint32_t* oparts = new uint32_t[edge_count+1];
    uint32_t* parts = new uint32_t[edge_count+1];
    uint32_t* pnum = new uint32_t[threads];
    uint32_t* psum = new uint32_t[threads];
    memset(pnum, 0, threads * sizeof(uint32_t));
    memset(psum, 0, threads * sizeof(uint32_t));

    // 0. sort update
    sort_updates(updates, edge_count, nn);

		// 1. generate partitions array
    uint32_t part = edge_count / threads;
    // printf("part:  %d %d\n", part, sizeof (pnum));
    parallel_for_1 (uint32_t tid = 0; tid < threads; ++tid) {
    // for (uint32_t tid = 0; tid < threads; ++tid) {
      uint32_t begin = part * tid;
      uint32_t end = begin + part;
      if (tid == (threads-1)) {
        end = edge_count;
      }

      // printf("%d %d\n", begin, end);

      uint32_t base = begin;
      vertex cur_src = get<0>(updates[begin]);
      oparts[base] = begin;
      pnum[tid]++;
      for (uint32_t i = begin+1; i < end; ++i) {
        if (cur_src != get<0>(updates[i])) {
          oparts[base + pnum[tid]] = i;
          pnum[tid]++;
          cur_src = get<0>(updates[i]);
        } 
      }

      if (tid == (threads-1)) {
        oparts[base + pnum[tid]] = edge_count;
        pnum[tid]++;
      }
      // oparts[base + pnum[tid]] = end;
      // pnum[tid]++;
    }

    // int sum = 0;
    // for (int i = 0; i < threads; ++i) {
    //   printf("%d\n", pnum[i]);
    //   sum+=pnum[i];
    // }
    // printf("part over, sum = %d\n", sum);
    //
    // int x = 0;
    for (uint32_t tid = 1; tid < threads; ++tid) {
      if (get<0>(updates[part * tid]) == get<0>(updates[(part * tid) - 1])) {
        // printf("edge same\n");
        pnum[tid-1]--;
        oparts[part*tid] = oparts[part*(tid-1) + pnum[tid-1]];
        // x++;
      }
      if (tid == 1) {
        psum[tid] = pnum[tid-1];
      } else {  
        psum[tid] = psum[tid-1] + pnum[tid-1];
      }
    }
    // printf("edge over %d\n", x);
    
    parallel_for_1 (uint32_t i = 0; i < threads; ++i) {
    // for (uint32_t i = 0; i < threads; ++i) {
      // printf("memcpy: %d, %d, %d\n", psum[i], i*part, pnum[i]);
      // printf("%d\n", oparts[i*part]);
      memcpy(parts + psum[i], oparts + (i * part), pnum[i] * sizeof(uint32_t));
    }
  
    // printf("memcpy over\n");
    // for (int i = 0; i < threads; ++i) {
    //   printf("%d\n", pnum[i]);
    // }
    // for (int i = 0; i < threads; ++i) {
    //   printf("%d\n", psum[i]);
    // }
    // set parts num
    part = psum[threads-1] + pnum[threads-1];

    delete [] oparts;
    delete [] pnum;
    delete [] psum;

    // 2. insert
		// parallel_for (uint32_t i = 0; i < parts.size()-1; i++) {
		parallel_for_1 (uint32_t i = 0; i < part-1; i++) {
      vertex s = get<0>(updates[parts[i]]);

      std::pair<uint32_t, uint32_t> range = {parts[i], parts[i+1]};
      uint32_t size = range.second - range.first;

#if ENABLE_LOCK
      lock(&vertices[s].degree);
#endif
      uint32_t degree = this->degree(s);
      //pre-peprocess & remove duplicate
      uint32_t* edge_insert_array = new uint32_t[size + degree];
      int new_size = 0;
      if (degree <= NUM_IN_PLACE_NEIGHBORS){ //in-cacheline
        int m = 0, n = 0;
        while(m < degree && n < size){
          if(vertices[s].neighbors[m] < get<1>(updates[range.first+n])){
            if(new_size == 0 || new_size > 0 && edge_insert_array[new_size-1] != vertices[s].neighbors[m]){
              edge_insert_array[new_size++] = vertices[s].neighbors[m];
            }
            m++;
          }
          else if(vertices[s].neighbors[m] > get<1>(updates[range.first+n])){
            if(new_size == 0 || new_size > 0 && edge_insert_array[new_size-1] != get<1>(updates[range.first+n])){
              edge_insert_array[new_size++] = get<1>(updates[range.first+n]);
            }
            n++;
          }
          else{
            if(new_size == 0 || edge_insert_array[new_size-1] != vertices[s].neighbors[m]){
              edge_insert_array[new_size++] = vertices[s].neighbors[m];
            }
            m++;
            n++;
          }
        }

        while(m < degree){
          if(new_size == 0 || new_size > 0 && edge_insert_array[new_size-1] != vertices[s].neighbors[m]){
            edge_insert_array[new_size++] = vertices[s].neighbors[m];
          }
          m++;
        }

        while(n < size){
          if(new_size == 0 || new_size > 0 && edge_insert_array[new_size-1] != get<1>(updates[range.first+n])){
            edge_insert_array[new_size++] = get<1>(updates[range.first+n]);
          }
          n++;
        }
      }else{
       int m = 0, n = 0;
        while(m < NUM_IN_PLACE_NEIGHBORS && n < size){
          if(vertices[s].neighbors[m] < get<1>(updates[range.first+n])){
            if(new_size == 0 || new_size > 0 && edge_insert_array[new_size-1] != vertices[s].neighbors[m]){
              edge_insert_array[new_size++] = vertices[s].neighbors[m];
            }
            m++;
          }
          else if(vertices[s].neighbors[m] > get<1>(updates[range.first+n])){
            if(new_size == 0 || new_size > 0 && edge_insert_array[new_size-1] != get<1>(updates[range.first+n])){
              edge_insert_array[new_size++] = get<1>(updates[range.first+n]);
            }
            n++;
          }
          else{
            if(new_size == 0 || edge_insert_array[new_size-1] != vertices[s].neighbors[m]){
              edge_insert_array[new_size++] = vertices[s].neighbors[m];
            }
            m++;
            n++;
          }
        }

        while(m < NUM_IN_PLACE_NEIGHBORS){
          if(new_size == 0 || new_size > 0 && edge_insert_array[new_size-1] != vertices[s].neighbors[m]){
            edge_insert_array[new_size++] = vertices[s].neighbors[m];
          }
          m++;
        }

        while(n < size){
          if(new_size == 0 || new_size > 0 && edge_insert_array[new_size-1] != get<1>(updates[range.first+n])){
            edge_insert_array[new_size++] = get<1>(updates[range.first+n]);
          }
          n++;
        }
      //  for (auto j = 0; j < size; j++){
      //   if(j == 0 || edge_insert_array[new_size-1] != get<1>(updates[range.first+j])){
      //     edge_insert_array[new_size++] = get<1>(updates[range.first+j]);
      //   }
      //  } 
      }
      auto final_degree = new_size;

      if (degree == 0) {
        // cacheline
        if (new_size <= NUM_IN_PLACE_NEIGHBORS) {
           memcpy(vertices[s].neighbors, edge_insert_array, new_size*sizeof(vertex));
        } else { // HITree
          fl_container *container = new fl_container();
          vertices[s].aux_neighbors = container;
          // std::pair<uint32_t, uint32_t>* init_data = new std::pair<uint32_t, uint32_t>[size];
          // for (int j = 0; j < size; ++j) {
          //   init_data[j].first = edge_insert_array[j];
          //   init_data[j].second = edge_insert_array[j];
          // }
          container->bulk_load(edge_insert_array, new_size);
          // delete [] init_data;
        }
        vertices[s].degree += new_size;
      } else if (final_degree <= NUM_IN_PLACE_NEIGHBORS) {
        memcpy(vertices[s].neighbors, edge_insert_array, final_degree * sizeof(vertex));
        vertices[s].degree = final_degree | LOCK_MASK;
      } else {
        if (degree <= NUM_IN_PLACE_NEIGHBORS) {
          // load_bulk, old+update, merge
          fl_container *container = new fl_container();
          vertices[s].aux_neighbors = container;
          // std::pair<uint32_t, uint32_t>* init_data = new std::pair<uint32_t, uint32_t>[degree+size];
          // for (int j = 0; j < final_degree; j++){
          //   init_data[j].first = edge_insert_array[j];
          //   init_data[j].second = edge_insert_array[j];
          // }
          //cache-line
          memcpy(vertices[s].neighbors, edge_insert_array, NUM_IN_PLACE_NEIGHBORS * sizeof(vertex));
          //HITree
          container->bulk_load(edge_insert_array + NUM_IN_PLACE_NEIGHBORS, final_degree-NUM_IN_PLACE_NEIGHBORS);
          // delete [] init_data;
          vertices[s].degree = final_degree | LOCK_MASK;
        } else {
          fl_container *container = (fl_container*)(vertices[s].aux_neighbors);
          memcpy(vertices[s].neighbors, edge_insert_array, NUM_IN_PLACE_NEIGHBORS * sizeof(vertex)); 
          // if (container == nullptr) {
          //   std::cout << "error container is null" << std::endl;
          // }
          for (uint32_t j = NUM_IN_PLACE_NEIGHBORS; j < new_size; j++) {
            // if (s != dests[j] && dests[j] != 191487) {
            // if (s != edge_insert_array[j] && s == 32731) {
            if (s != edge_insert_array[j] && container->find(edge_insert_array[j]) == false) {
              // std::cout << "degree " << (vertices[s].degree & 0x7fffffff) << "; insert: " << s << " " << edge_insert_array[j] << std::endl;
              // container->print_hitree();
              container->insert(edge_insert_array[j]);
              vertices[s].degree++;
            }
          }
        }
      }
      delete [] edge_insert_array;
#if ENABLE_LOCK
		  unlock(&vertices[s].degree);
#endif
		}
    delete [] parts;
  }

  void LSGraph::add_edge_batch_1for(vertex *srcs,vertex *dests, uint32_t edge_count) {
    struct timeval start, end;
    struct timezone tzp;
	  gettimeofday(&start, &tzp);

		// 1. generate partitions array
    uint32_t threads = getWorkers();
    uint32_t* oparts = new uint32_t[edge_count+1];
    uint32_t* parts = new uint32_t[edge_count+1];
    uint32_t* pnum = new uint32_t[threads];
    uint32_t* psum = new uint32_t[threads];
    memset(pnum, 0, threads * sizeof(uint32_t));
    memset(psum, 0, threads * sizeof(uint32_t));

    uint32_t part = edge_count / threads;
    // printf("part:  %d %d\n", part, sizeof (pnum));
    parallel_for_1 (uint32_t tid = 0; tid < threads; ++tid) {
    // for (uint32_t tid = 0; tid < threads; ++tid) {
      uint32_t begin = part * tid;
      uint32_t end = begin + part;
      if (tid == (threads-1)) {
        end = edge_count;
      }

      // printf("%d %d\n", begin, end);

      uint32_t base = begin;
      vertex cur_src = srcs[begin];
      oparts[base] = begin;
      pnum[tid]++;
      for (uint32_t i = begin+1; i < end; ++i) {
        if (cur_src != srcs[i]) {
          oparts[base + pnum[tid]] = i;
          pnum[tid]++;
          cur_src = srcs[i];
        } 
      }

      if (tid == (threads-1)) {
        oparts[base + pnum[tid]] = edge_count;
        pnum[tid]++;
      }
      // oparts[base + pnum[tid]] = end;
      // pnum[tid]++;
    }

	  gettimeofday(&end, &tzp);
  	print_time_elapsed("Partation edge: ", &start, &end);
	  gettimeofday(&start, &tzp);
    // int sum = 0;
    // for (int i = 0; i < threads; ++i) {
    //   printf("%d\n", pnum[i]);
    //   sum+=pnum[i];
    // }
    // printf("part over, sum = %d\n", sum);
    //
    // int x = 0;
    for (uint32_t tid = 1; tid < threads; ++tid) {
      if (srcs[part * tid] == srcs[(part * tid) - 1]) {
        // printf("edge same\n");
        pnum[tid-1]--;
        oparts[part*tid] = oparts[part*(tid-1) + pnum[tid-1]];
        // x++;
      }
      if (tid == 1) {
        psum[tid] = pnum[tid-1];
      } else {  
        psum[tid] = psum[tid-1] + pnum[tid-1];
      }
    }
    // printf("edge over %d\n", x);
    
    parallel_for_1 (uint32_t i = 0; i < threads; ++i) {
    // for (uint32_t i = 0; i < threads; ++i) {
      // printf("memcpy: %d, %d, %d\n", psum[i], i*part, pnum[i]);
      // printf("%d\n", oparts[i*part]);
      memcpy(parts + psum[i], oparts + (i * part), pnum[i] * sizeof(uint32_t));
    }
  
    // printf("memcpy over\n");
    // for (int i = 0; i < threads; ++i) {
    //   printf("%d\n", pnum[i]);
    // }
    // for (int i = 0; i < threads; ++i) {
    //   printf("%d\n", psum[i]);
    // }
    // set parts num
    part = psum[threads-1] + pnum[threads-1];

    delete [] oparts;
    delete [] pnum;
    delete [] psum;
	  gettimeofday(&end, &tzp);
    // printf("Part num = %d\n", part);
  	print_time_elapsed("Partation edge: ", &start, &end);

    // 2. insert
	  gettimeofday(&start, &tzp);
		// parallel_for (uint32_t i = 0; i < parts.size()-1; i++) {
		parallel_for_1 (uint32_t i = 0; i < part-1; i++) {
      vertex s = srcs[parts[i]];
      std::pair<uint32_t, uint32_t> range = {parts[i], parts[i+1]};
      uint32_t size = range.second - range.first;

      // if(s == 1832927){
      //   std::cout<<"before insert"<<std::endl;
      //   print_vertex_block(s);
      //   std::cout<<"insert vertex: "<<s<<std::endl;
      //   std::cout<<"size: "<<size<<std::endl;
      //   for(auto i = range.first; i < range.second; i++){
      //     std::cout<<dests[i]<<",";
      //   }
      //   std::cout << std::endl;
      // }

#if ENABLE_LOCK
      lock(&vertices[s].degree);
#endif
      uint32_t degree = this->degree(s);
      // printf("%d degree: %d\n", s, degree);
      
      if (degree == 0) {
        // cacheline
        if (size <= NUM_IN_PLACE_NEIGHBORS) {
			    memcpy(vertices[s].neighbors, &dests[range.first], size*sizeof(vertex));
        } else { // hitree
          fl_container *container = new fl_container();
          vertices[s].aux_neighbors = container;
          // std::pair<uint32_t, uint32_t>* init_data = new std::pair<uint32_t, uint32_t>[size];
          //
          // for (int j = 0; j < size; ++j) {
          //   init_data[j].first = dests[range.first+j];
          //   init_data[j].second = dests[range.first+j];
          // }

          container->bulk_load(&dests[range.first], size);

          // delete [] init_data;
        }
        vertices[s].degree += size;
      } else if (degree + size <= NUM_IN_PLACE_NEIGHBORS) {
        int newp = degree + size - 1;
        int oldp = degree-1;
        int upp = size - 1;
        for (; newp >= 0; --newp) {
          if (!(upp >= 0)) break;
          if (!(oldp >= 0)) {
			      memcpy(vertices[s].neighbors, &dests[range.first], (upp+1)*sizeof(vertex));
            //vertices[s].degree += (upp+1);
            break;
          }
          if (vertices[s].neighbors[oldp] > dests[range.first + upp]) {
            vertices[s].neighbors[newp] = vertices[s].neighbors[oldp];
            oldp--;
          } else if (vertices[s].neighbors[oldp] < dests[range.first + upp]) {
            vertices[s].neighbors[newp] = dests[range.first + upp];
            upp--;
          } else {
            vertices[s].neighbors[newp] = vertices[s].neighbors[oldp];
            oldp--;
            upp--;
            for (int j = newp; j <= degree + size - 1; j++) {
              vertices[s].neighbors[j-1] = vertices[s].neighbors[j];  
            }
            newp--;
            vertices[s].degree--;
          }
        }
        vertices[s].degree+=size;
      } else {
        if (degree <= NUM_IN_PLACE_NEIGHBORS) {
          // load_bulk, old+update, merge
          fl_container *container = new fl_container();
          vertices[s].aux_neighbors = container;
          uint32_t* init_data = new uint32_t[degree+size];

          int l = 0, j = 0;
          int k = 0;
          while (true) {
            if (l == degree) {
              for (; j < size; ++j) {
                init_data[k] = dests[range.first+j];
                k++;
              }
              break;
            }

            if (j == size) {
              for (; l < degree; ++l) {
                init_data[k] = vertices[s].neighbors[l];
                k++;
              }
              break;
            }

            if (vertices[s].neighbors[l] > dests[range.first+j]) {
              init_data[k] = dests[range.first+j];
              k++, j++;
            } else if (vertices[s].neighbors[l] < dests[range.first+j]) {
              init_data[k] = vertices[s].neighbors[l];
              k++, l++;
            } else {
              init_data[k] = dests[range.first+j];
              k++, j++, l++;
            }
          }

          container->bulk_load(init_data, k);

          delete [] init_data;

          vertices[s].degree = k | LOCK_MASK;
        } else {
          fl_container *container = (fl_container*)(vertices[s].aux_neighbors);
          // if (container == nullptr) {
          //   std::cout << "error container is null" << std::endl;
          // }
          for (uint32_t j = range.first; j < range.second; j++) {
            // if (s != dests[j] && dests[j] != 191487) {
            if (s != dests[j]) {
              // std::cout << "degree " << (vertices[s].degree & 0x7fffffff) << "; insert: " << s << " " << dests[j] << std::endl;
              // if (s == 60415) container->print_hitree();
              container->insert(dests[j]);
              vertices[s].degree++;
            }
          }
        }
      }
#if ENABLE_LOCK
		  unlock(&vertices[s].degree);
#endif
		}
    delete [] parts;
	  gettimeofday(&end, &tzp);
  	print_time_elapsed("Insert edge: ", &start, &end);
  }


	void LSGraph::bulk_load(vertex *srcs,vertex *dests, uint64_t edge_count) {
    struct timeval start, end;
    struct timezone tzp;
	  gettimeofday(&start, &tzp);

		// 1. generate partitions array
		std::vector<uint64_t> parts;
		vertex cur_src = srcs[0];
		parts.emplace_back(0);
    // int part_num = 0;
		for (uint64_t i = 1; i < edge_count; i++) {
			if (cur_src != srcs[i]) {
				parts.emplace_back(i);
				cur_src = srcs[i];
        // part_num++;
			}
		}
		parts.emplace_back(edge_count);

	  gettimeofday(&end, &tzp);
    // printf("Part num = %d\n", part_num);
  	print_time_elapsed("Partation edge: ", &start, &end);

    // 2. insert array and hitree
	  gettimeofday(&start, &tzp);
    parallel_for (uint32_t i = 0; i < parts.size()-1; ++i) {
      auto s = srcs[parts[i]];
      std::pair<uint64_t, uint64_t> range = {parts[i], parts[i+1]};
      uint64_t size = range.second - range.first;

#if ENABLE_LOCK
      lock(&vertices[s].degree);
#endif
      
      if (size <= NUM_IN_PLACE_NEIGHBORS) {
			  memcpy(vertices[s].neighbors, &dests[range.first], size*sizeof(vertex));
      } else {
			  memcpy(vertices[s].neighbors, &dests[range.first], NUM_IN_PLACE_NEIGHBORS*sizeof(vertex));
        fl_container *container = new fl_container();
        vertices[s].aux_neighbors = container;
        // std::pair<uint32_t, uint32_t>* init_data = new std::pair<uint32_t, uint32_t>[size];
        //
        // for (int j = 0; j < size; ++j) {
        //   init_data[j].first = dests[range.first+j];
        //   init_data[j].second = dests[range.first+j];
        // }

        container->bulk_load(&dests[range.first+NUM_IN_PLACE_NEIGHBORS], size - NUM_IN_PLACE_NEIGHBORS);
        // container->bulk_load(&dests[range.first], size);
        
        // delete [] init_data;
      }

      vertices[s].degree += size;

      // if (s == 10){
      //   print_hitree(10);
      // } 
      // if (s == 60)
      //   printf("s = %d\n, degree = %d\n", s, vertices[s].degree);

#if ENABLE_LOCK
      unlock(&vertices[s].degree);
#endif
      // std::cout<<"process id: "<<getpid()<<std::endl;
      // std::cout<<"kernel id: "<<gettid()<<std::endl;
      // std::cout<<"std thread id: "<< std::this_thread::get_id()<<std::endl;
      // std::cout<<"pthread id:"<< pthread_self()<<std::endl;
    }

	  gettimeofday(&end, &tzp);
  	print_time_elapsed("bulk load time: ", &start, &end);
  }

  uint64_t LSGraph::get_index() {

    uint64_t sum = 0;
    for (int s = 0; s < get_num_vertices(); ++s) {
      auto degree = vertices[s].degree;
      if(degree <= NUM_IN_PLACE_NEIGHBORS){
      }else if(degree > NUM_IN_PLACE_NEIGHBORS){
       sum +=  ((fl_container*)(vertices[s].aux_neighbors))->get_index_num();
       }
    }
    return sum;
  }

  void LSGraph::visit_edge(const vertex s){ 
    auto degree = vertices[s].degree;
    if(degree <= NUM_IN_PLACE_NEIGHBORS){
      for (uint32_t i = 0; i < degree; i++) {
        uint32_t d = vertices[s].neighbors[i];
        std::cout << d << " ";
      }
    }else if(degree > NUM_IN_PLACE_NEIGHBORS){
      ((fl_container*)(vertices[s].aux_neighbors))->visit_edges();
     }
     //std::cout << std::endl;
  }

  void LSGraph::visit_edge_test(const vertex s, std::vector<uint32_t> &array){
    auto degree = vertices[s].degree;
    if(degree <= NUM_IN_PLACE_NEIGHBORS){
      for (uint32_t i = 0; i < degree; i++) {
        uint32_t d = vertices[s].neighbors[i];
        array.push_back(d);
      }
    }else if(degree > NUM_IN_PLACE_NEIGHBORS){
      ((fl_container*)(vertices[s].aux_neighbors))->visit_edges_test(array);
     }
  }
  
  uint64_t LSGraph::collect_conflict(const vertex s){
    auto degree = vertices[s].degree;
    if(degree > NUM_IN_PLACE_NEIGHBORS){
      auto container = (fl_container*)(vertices[s].aux_neighbors);
      return container->num_confilicts();
    }else{
      return 0;
    }
  }

  uint32_t LSGraph::is_edge(const vertex s, const vertex d) {
    for (uint32_t i = 0; i < vertices[s].degree && i <= NUM_IN_PLACE_NEIGHBORS;
         i++) {
      if (vertices[s].neighbors[i] == d)
        return 1;
    }
    if (vertices[s].degree > NUM_IN_PLACE_NEIGHBORS) {
      return ((fl_container*)(vertices[s].aux_neighbors))->find(d);
    }
    return 0;
  }

  void LSGraph::print(const vertex s, std::vector<uint32_t>& nei) {
    //std::cout<<"degree:"<<vertices[s].degree<<std::endl;
    for (uint32_t i = 0; i < vertices[s].degree && i < NUM_IN_PLACE_NEIGHBORS;
         i++) {
      nei.push_back(vertices[s].neighbors[i]);
    }
    if (vertices[s].degree > NUM_IN_PLACE_NEIGHBORS) {
      // ((fl_container*)(vertices[s].aux_neighbors))->print_stats();
      ((fl_container*)(vertices[s].aux_neighbors))->print_hitree(nei);
    }
  }

	int LSGraph::remove_edge(const vertex s, const vertex d) {
    if(s == d) return -1;
		uint32_t idx;
		int ret{0};
#if ENABLE_LOCK
		lock(&vertices[s].degree);
#endif
		uint32_t degree = this->degree(s);
    // if(s == 46760 && d == 2930360) {
    //   std::cout<<"degree:"<<degree<<std::endl;
    // }
		if (degree == 0) {
			ret = -1;
			goto unlock;
		}
		else {
#if PREFETCH
			if (has_aux(s)) {
				__builtin_prefetch(vertices[s].aux_neighbors, 1);  
			}
#endif
			if (degree <= NUM_IN_PLACE_NEIGHBORS) { // only in place
				for (idx = 0; idx < degree; idx++) {
					if (vertices[s].neighbors[idx] < d)
						continue;
					else if (vertices[s].neighbors[idx] == d) {
						memcpy(&vertices[s].neighbors[idx], &vertices[s].neighbors[idx+1],
									 (degree-idx-1)*sizeof(vertex));
						goto decr;
					}
					else
						break;
				}
				if (idx == degree) {
					ret = -1;
					goto unlock;
				}
			} else if (d > vertices[s].neighbors[NUM_IN_PLACE_NEIGHBORS-1]) {
				// only in hitree
          fl_container *container = (fl_container*)(vertices[s].aux_neighbors);
          if(!container->remove(d)){
            ret = -1;
            goto unlock;
          }
			} else { // both in place and hitree
				// first remove from in place
				for (idx = 0; idx < NUM_IN_PLACE_NEIGHBORS; idx++) {
					if (vertices[s].neighbors[idx] < d)
						continue;
					else if (vertices[s].neighbors[idx] == d) {
						memcpy(&vertices[s].neighbors[idx], &vertices[s].neighbors[idx+1],
									 (NUM_IN_PLACE_NEIGHBORS-idx-1)*sizeof(vertex));
						break;
					}
					else
						break;
				}
				// remove the smallest from hitree
				vertex bump{0};
        fl_container *container = (fl_container*)(vertices[s].aux_neighbors);
        bump = container->get_first_key();
        if (!container->remove(bump)) {
            //delete failed : key not found.
						ret = -1;
						goto unlock;
				}
				// insert in in place
				vertices[s].neighbors[NUM_IN_PLACE_NEIGHBORS-1] = bump;	
			}
		}

decr:
		vertices[s].degree--;
unlock:
#if ENABLE_LOCK
		unlock(&vertices[s].degree);
#endif

		return ret;
	}

  // LSGraph::NeighborIterator LSGraph::neighbors(const vertex v) const {
  //   return NeighborIterator(this, v);
  // }

  inline uint32_t LSGraph::degree(const vertex v) const {
#if ENABLE_LOCK
    return vertices[v].degree & UNLOCK_MASK;
#else
    return vertices[v].degree;
#endif
  }

  // uint64_t LSGraph::get_size(void) {
  //   uint64_t total_size{0};
  //   total_size += inplace_size;
  //   for (uint32_t idx = 0; idx < num_vertices; idx++) {
  //     if (vertices[idx].aux_neighbors != nullptr)
  //       total_size +=
  //         ((tl_container*)(vertices[idx].aux_neighbors))->get_size(); 
  //   }
  //   return total_size;
  // }

  uint32_t LSGraph::get_num_edges(void) {
		uint64_t num_edges{0};
		for (uint32_t i = 0; i < num_vertices; i++)
			num_edges += vertices[i].degree;
		return num_edges;
  }

  uint32_t LSGraph::get_num_vertices(void) const {
    return num_vertices;
  }

  void LSGraph::print_vertex_block(const vertex v) const {
    // std::cout << "Vertex: " << v << "\n";
    // std::cout << "Degree: " << vertices[v].degree << "\n";
    std::cout << "In place neighbors: \n";
    for (uint32_t i = 0; i < vertices[v].degree && i < NUM_IN_PLACE_NEIGHBORS;
         i++) {
      std::cout << vertices[v].neighbors[i] << ", ";
    }
    std::cout << '\b';
    std::cout << '\b' << '\n';
  }

  template<class F, typename VS> struct HITREE_map{
    VS &vs;
    F f;
    bool output;
    LSGraph::vertex self_index;
    LSGraph::fl_container* hitree;

    HITREE_map(VS &vs, F &f, bool output, LSGraph::vertex self_index, LSGraph::fl_container* hitree) : vs(vs), f(f), output(output), self_index(self_index), hitree(hitree) {}

    void update(LSGraph::vertex v){
      if (f.cond(v) == 1 && f.updateAtomic(self_index, v) == 1) {
				if (output) {
					vs.insert_sparse(v);
				}
			}
    }
  };

  template<class F, typename VS> struct HITREE_map_dense{
    VS &vs;
    F &f;
    bool output;
    LSGraph::vertex self_index;
    LSGraph::fl_container* hitree;

    HITREE_map_dense(VS &vs, F &f, bool output, LSGraph::vertex self_index, LSGraph::fl_container* hitree) : vs(vs), f(f), output(output), self_index(self_index), hitree(hitree) {}

    void update(LSGraph::vertex v) const {
      if (f.update(v, self_index) == 1)
      {
        // printf("all %d %d\n", v, self_index);
        if (output) {
					vs.insert_dense(self_index);
				}
      }
    }
  };

 template<class F, typename VS> struct HITREE_map_dense_no_all{
    VS &vs;
    VS &output_vs;
    F &f;
    bool output;
    LSGraph::vertex self_index;
    LSGraph::fl_container* hitree;

    HITREE_map_dense_no_all(VS &vs, VS &output_vs, F &f, bool output, LSGraph::vertex self_index, LSGraph::fl_container* hitree) : vs(vs), output_vs(output_vs), f(f), output(output), self_index(self_index), hitree(hitree) {}

    void update(LSGraph::vertex v) const{
      if (vs.has_dense_no_all(v) && f.update(v, self_index) == 1)
      {
        // printf("not all %d %d\n", v, self_index);
        if (output) {
					output_vs.insert_dense(self_index);
				}
      }
    }
  };


  template <class F, typename VS>
    void LSGraph::map_sparse(F &f, VS &output_vs, uint32_t self_index, bool output) {
      uint32_t degree = vertices[self_index].degree;
      // std::cout << "vertex: " << self_index << " degree: " << degree << '\n';
      uint32_t local_idx = 0;
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[self_index].neighbors[local_idx];
          if (f.cond(v) == 1 && f.updateAtomic(self_index, v) == 1) {
            if (output) {
              output_vs.insert_sparse(v);
            }
          }
          ++local_idx;  
        }
      } else {
#if PREFETCH
        if (has_aux(self_index)) {
          __builtin_prefetch(vertices[self_index].aux_neighbors);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[self_index].neighbors[local_idx];
          if (f.cond(v) == 1 && f.updateAtomic(self_index, v) == 1) {
            if (output) {
              output_vs.insert_sparse(v);
            }
          }
          ++local_idx;
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (has_aux(self_index)) {
              __builtin_prefetch(((fl_container*)vertices[self_index].aux_neighbors)->get_root()); 
            }
          }
#endif
        }
        {
          fl_container *hitree = (fl_container *)vertices[self_index].aux_neighbors;
          struct HITREE_map<F, VS> update_fn(output_vs, f, output, self_index, hitree);
          hitree->map(update_fn);
        }
      }
    }

  template <class F, typename VS>
    void LSGraph::map_dense_vs_all(F &f, VS &vs, VS &output_vs, uint32_t self_index, bool output) {
      uint32_t degree = vertices[self_index].degree;
      uint32_t local_idx = 0;
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[self_index].neighbors[local_idx];
          if (f.update(v, self_index) == 1) {
            if (output) {
              output_vs.insert_dense(self_index);
            }
          }
          if (f.cond(self_index) == 0) {
            return;
          }
          ++local_idx;  
        }
      } else {
#if PREFETCH
        if (has_aux(self_index)) {
          __builtin_prefetch(vertices[self_index].aux_neighbors);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[self_index].neighbors[local_idx];
          if (f.update(v, self_index) == 1) {
            if (output) {
              output_vs.insert_dense(self_index);
            }
          }
          if (f.cond(self_index) == 0) {
            return;
          }
          ++local_idx;
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (has_aux(self_index)) {
              __builtin_prefetch(((fl_container*)vertices[self_index].aux_neighbors)->get_root()); 
            }
          }
#endif
        }
        {
          fl_container *hitree = (fl_container *)vertices[self_index].aux_neighbors;
          struct HITREE_map_dense<F, VS> update_fn(output_vs, f, output, self_index, hitree);
          hitree->map_dense(update_fn);
        }
      }
    }

  template <class F, typename VS>
    void LSGraph::map_dense_vs_not_all(F &f, VS &vs, VS &output_vs, uint32_t self_index, bool output) {
      uint32_t degree = vertices[self_index].degree;
      uint32_t local_idx = 0;
      if (degree <= NUM_IN_PLACE_NEIGHBORS) {
        while (local_idx < degree) {
          auto v = vertices[self_index].neighbors[local_idx];
          if (vs.has_dense_no_all(v) && f.update(v, self_index) == 1) {
            if (output) {
              output_vs.insert_dense(self_index);
            }
          }
          if (f.cond(self_index) == 0) {
            return;
          }
          ++local_idx;  
        }
      } else {
#if PREFETCH
        if (has_aux(self_index)) {
          __builtin_prefetch(vertices[self_index].aux_neighbors);  
        }
#endif
        while (local_idx < NUM_IN_PLACE_NEIGHBORS) {
          auto v = vertices[self_index].neighbors[local_idx];
          if (vs.has_dense_no_all(v) && f.update(v, self_index) == 1) {
            if (output) {
              output_vs.insert_dense(self_index);
            }
          }
          if (f.cond(self_index) == 0) {
            return;
          }
          ++local_idx;
#if PREFETCH
          if (local_idx == NUM_IN_PLACE_NEIGHBORS/2) {
        		if (has_aux(self_index)) {
              __builtin_prefetch(((fl_container*)vertices[self_index].aux_neighbors)->get_root()); 
            }
          }
#endif
        }
        {
          fl_container *hitree = (fl_container *)vertices[self_index].aux_neighbors;
          struct HITREE_map_dense_no_all<F, VS> update_fn(vs, output_vs, f, output, self_index, hitree);
          hitree->map_dense(update_fn);
        }
      }
    }
}


#endif // _GRAPHALEX_H_
