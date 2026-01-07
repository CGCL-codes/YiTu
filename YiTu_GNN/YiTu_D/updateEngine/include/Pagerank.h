// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of 
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// #include "ligra.h"
#include "math.h"
#pragma once
#include "Map.cpp"
#define newA(__E,__n) (__E*) malloc((__n)*sizeof(__E))

// template <class vertex>
template<typename T, class Graph>
struct PR_F {
  T* p_curr, *p_next;
  // vertex* V;
  // PR_F(double* _p_curr, double* _p_next, vertex* _V) : 
  Graph& G;
  PR_F(T* _p_curr, T* _p_next, Graph& _G) : 
    p_curr(_p_curr), p_next(_p_next), G(_G) {}
  inline bool update(uint32_t s, uint32_t d){ //update function applies PageRank equation
    // p_next[d] += p_curr[s]/V[s].getOutDegree();
    p_next[d] += p_curr[s];

    return 1;
  }
  inline bool updateAtomic ([[maybe_unused]] uint32_t s, [[maybe_unused]] uint32_t d) { //atomic Update
    
    printf("should never be called for now since its always dense\n");
    while(1) { }
    /*
    uint64_t x = 0;
    uint64_t *y = (uint64_t *) x;
    *y = 0;
    */
    // exit(-1);
    //__sync_fetch_and_add(&p_next[d],p_curr[s]/G.lines[s].get_n());
    return 1;
  }
  inline bool cond ([[maybe_unused]] uint32_t  d) { return 1; }}; // from ligra readme: for cond which always ret true, ret cond_true// return cond_true(d); }};

template<typename T, class Graph>
struct PR_Vertex {
  T *p_curr;
  Graph& G;
  PR_Vertex(T* _p_curr, Graph& _G) : p_curr(_p_curr), G(_G) {}
  inline bool operator () (uint32_t i) {
    p_curr[i] = p_curr[i] /G.degree(i);// G.num_neighbors(i);; // damping*p_next[i] + addedConstant;
    return 1;
  }
};

/*
//vertex map function to update its p value according to PageRank equation
struct PR_Vertex_F {
  double damping;
  double addedConstant;
  double* p_curr;
  double* p_next;
  PR_Vertex_F(double* _p_curr, double* _p_next, double _damping, intE n) :
    p_curr(_p_curr), p_next(_p_next), 
    damping(_damping), addedConstant((1-_damping)*(1/(double)n)){}
  inline bool operator () (uintE i) {
    p_next[i] = p_next[i]; // damping*p_next[i] + addedConstant;
    return 1;
  }
};
*/

//resets p
template<typename T>
struct PR_Vertex_Reset {
  T* p;
  PR_Vertex_Reset(T* _p) :
    p(_p) {}
  inline bool operator () (uint32_t i) {
    p[i] = 0.0;
    return 1;
  }
};
/*
template <class OT, class intT>
OT plusReduce(OT* A, intT n) {
  return reduce<OT>((intT)0,n,addF<OT>(),getA<OT,intT>(A));
}
*/
template<typename T, class Graph>
T* PR_S(Graph& G, long maxIters) {
  size_t n = G.get_num_vertices();
  //const double damping = 0.85, epsilon = 0.0000001;

  //timer ss; ss.start();
  //auto vtxs = G.fetch_all_vertices(); 
  //ss.stop(); ss.reportTotal("snapshot time");

  /*
  uint64_t test_edges = 0;
  ofstream ofile0;
  ofile0.open("degs_s.txt");
  for(uint32_t i = 0; i < n; i++) {
    // printf("degree of %u = %u\n", i, vtxs[i].degree());
    // ofile0 << vtxs[i].degree() << "\n";
    test_edges += vtxs[i].degree();
  }

  assert(test_edges == G.num_edges());
  ofile0.close();
  */

  T one_over_n = 1/(double)n;
  T* p_curr = (T*) memalign(32, n*sizeof(T));
  T* p_next = (T*) memalign(32, n*sizeof(T));
  parallel_for(size_t i = 0; i < n; i++) {
    p_curr[i] = one_over_n;
  }
  VertexSubset Frontier = VertexSubset(0, n, true);
  // VertexSubset Frontier = VertexSubset((uint32_t)0, n);
  
  long iter = 0;
  printf("max iters %lu\n", maxIters);
  while(iter++ < maxIters) {
    //printf("\t running iteration %lu\n", iter);
    // using flat snapshot
    vertexMap(Frontier,PR_Vertex(p_curr, G), false);
    vertexMap(Frontier,PR_Vertex_Reset(p_next), false);
    edgeMap(G, Frontier,PR_F(p_curr,p_next,G), false);

    //vertexMap(Frontier,PR_Vertex_F(p_curr,p_next,damping,n));
    //compute L1-norm between p_curr and p_next
    /*
    {parallel_for(long i=0;i<n;i++) {
      p_curr[i] = fabs(p_curr[i]-p_next[i]);
      }}
    */
    //parallel_for(0, n, [&] (size_t i) { p_curr[i] = fabs(p_curr[i]-p_next[i]); } );

    // required for early exit
    // auto temp = pbbs::sequence<double>(p_curr, n); 
    // double L1_norm = pbbs::reduce(temp, pbbs::addm<size_t>());
    // p_curr = temp.to_array();
    // if(L1_norm < epsilon) break;

    // reset p_curr
    std::swap(p_curr,p_next);
  }

#if VERIFY
	std::ofstream ofile;
  ofile.open("p_curr_s.out");
  for(uint32_t i = 0; i < n; i++) {
    ofile << p_curr[i] << "\n";
  }
  ofile.close();

	std::ofstream ofile2;
  ofile2.open("p_next_s.out");
  for(uint32_t i = 0; i < n; i++) {
    ofile2 << p_next[i] << "\n";
  }
  ofile2.close();
#endif
  free(p_next);

  // printf("p curr %p, p next %p\n", p_curr, p_next);
  return p_curr;
}
