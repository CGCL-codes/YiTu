#pragma once
#include "Map.cpp"
#include <vector>
#include "parallel_util.h"

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
#include <vector>

typedef double fType;
typedef uint32_t uintE;

struct BC_F {
  fType* NumPaths;
  bool* Visited;
  BC_F(fType* _NumPaths, bool* _Visited) : 
    NumPaths(_NumPaths), Visited(_Visited) {}
  inline bool update(uintE s, uintE d){ //Update function for forward phase
    fType oldV = NumPaths[d];
    NumPaths[d] += NumPaths[s];
    return oldV == 0.0;
  }
  inline bool updateAtomic (uintE s, uintE d) { //atomic Update, basically an add
    volatile fType oldV, newV; 
    do { 
      oldV = NumPaths[d]; newV = oldV + NumPaths[s];
    } while(!CAS(&NumPaths[d],oldV,newV));
    return oldV == 0.0;
  }
  inline bool cond (uintE d) { return Visited[d] == 0; } //check if visited
};

struct BC_Back_F {
  fType* Dependencies;
  bool* Visited;
  BC_Back_F(fType* _Dependencies, bool* _Visited) : 
    Dependencies(_Dependencies), Visited(_Visited) {}
  inline bool update(uintE s, uintE d){ //Update function for backwards phase
    fType oldV = Dependencies[d];
    Dependencies[d] += Dependencies[s];
    return oldV == 0.0;
  }
  inline bool updateAtomic (uintE s, uintE d) { //atomic Update
    volatile fType oldV, newV;
    do {
      oldV = Dependencies[d];
      newV = oldV + Dependencies[s];
    } while(!CAS(&Dependencies[d],oldV,newV));
    return oldV == 0.0;
  }
  inline bool cond (uintE d) { return Visited[d] == 0; } //check if visited
};

//vertex map function to mark visited vertexSubset
struct BC_Vertex_F {
  bool* Visited;
  BC_Vertex_F(bool* _Visited) : Visited(_Visited) {}
  inline bool operator() (uintE i) {
    Visited[i] = 1;
    return 1;
  }
};

//vertex map function (used on backwards phase) to mark visited vertexSubset
//and add to Dependencies score
struct BC_Back_Vertex_F {
  bool* Visited;
  fType* Dependencies, *inverseNumPaths;
  BC_Back_Vertex_F(bool* _Visited, fType* _Dependencies, fType* _inverseNumPaths) : 
    Visited(_Visited), Dependencies(_Dependencies), inverseNumPaths(_inverseNumPaths) {}
  inline bool operator() (uintE i) {
    Visited[i] = 1;
    Dependencies[i] += inverseNumPaths[i];
    return 1; }
};

template <class Graph>
auto BC(Graph& G, const uintE& start, [[maybe_unused]] bool use_dense_forward=false) {
  size_t n = G.get_num_vertices();
  fType* NumPaths = newA(fType,n);
  {parallel_for(uint64_t i=0;i<n;i++) NumPaths[i] = 0.0;}
  bool* Visited = newA(bool,n);
  {parallel_for(uint64_t i=0;i<n;i++) Visited[i] = 0;}
  Visited[start] = 1; NumPaths[start] = 1.0;
  VertexSubset Frontier = VertexSubset(start, n); //creates initial frontier

  std::vector<VertexSubset> Levels;
  Levels.push_back(Frontier);
  long round = 0;
  while (Frontier.not_empty()) {
    //printf("%u, %u\n", round, Frontier.get_n());
    round++;
    VertexSubset output = edgeMap(G, Frontier, BC_F(NumPaths,Visited));
    Frontier = output;
    Levels.push_back(Frontier);
    vertexMap(Frontier, BC_Vertex_F(Visited), false); // mark visited
  }

  fType* Dependencies = newA(fType,n);
  {parallel_for(uint64_t i=0;i<n;i++) Dependencies[i] = 0.0;}
  
  parallel_for(uint64_t i = 0; i < n; i++) {
    NumPaths[i] = 1/NumPaths[i];
  }
  Levels[round].del();

  parallel_for(uint64_t i = 0; i < n; i++) {
    Visited[i] = 0;
  }

  vertexMap(Levels[round-1], BC_Back_Vertex_F(Visited,Dependencies,NumPaths), false);

  for(long r=round-2;r>=0;r--) {
    edgeMap(G, Levels[r+1], BC_Back_F(Dependencies,Visited), false);
    Levels[r+1].del();
    vertexMap(Levels[r], BC_Back_Vertex_F(Visited,Dependencies,NumPaths), false);
  }

  parallel_for(uint32_t i = 0; i < n; i++) {
    Dependencies[i] = (Dependencies[i]-NumPaths[i])/NumPaths[i];
  }
  Levels[0].del();
  free(NumPaths);
  free(Visited);

#if VERIFY
  // write out to file
  std::ofstream myfile;
  myfile.open ("bc.out");
  for (int i = 0; i < n; i++) {
    myfile << Dependencies[i] << "\n";
  }
#endif

  return Dependencies;
}

