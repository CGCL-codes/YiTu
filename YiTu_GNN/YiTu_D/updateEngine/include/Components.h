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
#include "Map.cpp"
#include "parallel_util.h"

#define newA(__E,__n) (__E*) malloc((__n)*sizeof(__E))

struct CC_Shortcut {
   uint32_t* IDs, *prevIDs;
  CC_Shortcut( uint32_t* _IDs,  uint32_t* _prevIDs) :
    IDs(_IDs), prevIDs(_prevIDs) {}
  inline bool operator () ( uint32_t i) {
     uint32_t l = IDs[IDs[i]];
    if(IDs[i] != l) IDs[i] = l;
    if(prevIDs[i] != IDs[i]) {
      prevIDs[i] = IDs[i];
      return 1; }
    else return 0;
  }
};
struct CC_Vertex_F {
  uint32_t* IDs, *prevIDs;
  CC_Vertex_F(uint32_t* _IDs, uint32_t* _prevIDs) :
    IDs(_IDs), prevIDs(_prevIDs) {}
  inline bool operator () (uint32_t i) {
    prevIDs[i] = IDs[i];
    return 1; }};


struct CC_F {
   uint32_t* IDs, *prevIDs;
  CC_F( uint32_t* _IDs,  uint32_t* _prevIDs) : 
    IDs(_IDs), prevIDs(_prevIDs) {}
  inline bool update( uint32_t s,  uint32_t d){ //Update function writes min ID
     uint32_t origID = IDs[d];
    if(IDs[s] < origID) {
      IDs[d] = IDs[s];
      if(origID == prevIDs[d]) return 1;
    } return 0; }
  inline bool updateAtomic ( uint32_t s,  uint32_t d) { //atomic Update
     uint32_t origID = IDs[d];
    return (writeMin(&IDs[d],IDs[s]) && origID == prevIDs[d]);
  }
  inline bool cond ([[maybe_unused]] uint32_t d) { return true; } //does nothing
};

/*
uint32_t *CC_shortcut(Graph &G) {
  long n = G.get_num_vertices();
  uint32_t* IDs = newA( uint32_t,n), *prevIDs = newA( uint32_t,n);
  //initialize unique IDs
  parallel_for(long i=0;i<n;i++) {
    IDs[i] = i;
    prevIDs[i] = i;
  }

  VertexSubset *Active = new VertexSubset(0, n, true); //initial frontier contains all vertices

  while(Active->not_empty()){ //iterate until IDS converge
    edgeMap(G, *Active, CC_F(IDs,prevIDs), false);
    delete Active;
    Active = new VertexSubset(0,n,true, true);
    vertexMap(*Active,CC_Shortcut(IDs,prevIDs));
    Active->move_next_to_current();
  }
  delete Active;
  free(prevIDs);
  return IDs;
}
*/
template <class Graph>
uint32_t *CC(Graph &G) {
  long n = G.get_num_vertices();
  uint32_t* IDs = newA( uint32_t,n), *prevIDs = newA( uint32_t,n);
  //initialize unique IDs
  parallel_for(long i=0;i<n;i++) {
    IDs[i] = i;
  }

  VertexSubset Active = VertexSubset(0, n, true); //initial frontier contains all vertices

  while(Active.not_empty()){ //iterate until IDS converge
    // printf("frontier size  %lu\n", Active.get_n());
    vertexMap(Active,CC_Vertex_F(IDs,prevIDs), false);
    VertexSubset next = edgeMap(G, Active, CC_F(IDs,prevIDs), true); //INT_MAX);
    Active.del();
    Active = next;
  }
  Active.del();
  free(prevIDs);
  
#if VERIFY
  std::set<uint32_t> components_set;
  for (uint32_t i = 0; i < n; i++) {
    components_set.insert(IDs[i]);
  }
  printf("number of components is %lu\n", components_set.size());
#endif
  
  return IDs;
}
