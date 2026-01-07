#pragma once
#define newA(__E,__n) (__E*) malloc((__n)*sizeof(__E))
#include "parallel.h"
#include "util.h"
#include <iostream>
#include <map>

#define MAXVAL 254

uint32_t rand_in_range(uint32_t max) { return rand() % max; }

// A structure that keeps a sequence of strings all allocated from
// the same block of memory
struct words {
  char* Chars;  // array storing all strings
  long n; // total number of characters
  char** Strings; // pointers to strings (all should be null terminated)
  long m; // number of substrings
  words() {}
  words(char* C, long nn, char** S, long mm)
    : Chars(C), n(nn), Strings(S), m(mm) {}
  void del() {free(Chars); free(Strings);}
};

inline bool isSpace(char c) {
  switch (c)  {
  case '\r': 
  case '\t': 
  case '\n': 
  case 0:
  case ' ' : return true;
  default : return false;
  }
}
// parallel code for converting a string to words
words stringToWords(char *Str, uint64_t n) {
  parallel_for (uint64_t i=0; i < n; i++) 
    if (isSpace(Str[i])) Str[i] = 0; 

  // mark start of words
  bool *FL = newA(bool,n);
  FL[0] = Str[0];
  parallel_for (uint64_t i=1; i < n; i++) FL[i] = Str[i] && !Str[i-1];

  uint32_t worker_count = getWorkers();
  std::vector<uint64_t> sub_counts(worker_count, 0);
  uint64_t section_count = (n/worker_count)+1;
  parallel_for_1(uint64_t i = 0; i < worker_count; i++) {
    uint64_t start = i * section_count;
    uint64_t end = std::min((i+1)*section_count, n);
    uint64_t local_count = 0;
    for (uint64_t j = start; j < end; j++) {
      if (FL[j]) {
        local_count+=1;
      }
    }
    sub_counts[i] = local_count;
  }
  // count and prefix sum
  for (uint32_t i = 1 ; i < worker_count; i++) {
     sub_counts[i] += sub_counts[i-1];
  }
  uint64_t m = sub_counts[worker_count-1];
  /*
  uint64_t *offsets = newA(uint64_t, m);
  uint64_t j = 0;
  for (uint64_t i = 0; i < m; i++) {
    while (FL[j] != 1) {
      j++;
    }
    offsets[i] = j;
    j++;
  }
  */
  uint64_t *offsets = newA(uint64_t, m);
  parallel_for_1(uint64_t i = 0; i < worker_count; i++) {
    uint64_t start = i * section_count;
    uint64_t end = std::min((i+1)*section_count, n);
    uint64_t offset;
    if (i == 0) offset = 0;
    else offset = sub_counts[i-1];
    for (uint64_t j = start; j < end; j++) {
      if (FL[j] == 1) {
        offsets[offset++] = j;
      }
    }
  }

  // pointer to each start of word
  char **SA = newA(char*, m);
  parallel_for (uint64_t j=0; j < m; j++) SA[j] = Str+offsets[j];

  free(offsets); free(FL);
  return words(Str,n,SA,m);
}
char* readStringFromFile(const char *fileName, long* length) {
  std::ifstream file (fileName, std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cout << "Unable to open file: " << fileName << std::endl;
    abort();
  }
  long end = file.tellg();
  file.seekg (0, std::ios::beg);
  long n = end - file.tellg();
  char* bytes = newA(char,n+1);
  file.read (bytes,n);
  file.close();
  *length = n;
  return bytes;
}

pair_uint *get_edges_from_file_adj_sym(std::string filename,
                              uint64_t *edge_count, uint32_t *node_count, [[maybe_unused]] bool print = true) {
  long length;
  char *S = readStringFromFile(filename.c_str(), &length);
  words W = stringToWords(S, length);
  if (strcmp(W.Strings[0], "AdjacencyGraph") != 0) {
    std::cout << "Bad input file: missing header: AdjacencyGraph" << std::endl;
    exit(-1);
  }
  std::cout<<"checkpoint1"<<std::endl;
  uint64_t len = W.m -1;
  uint64_t * In = newA(uint64_t, len);
  {parallel_for(uint64_t i=0; i < len; i++) In[i] = atol(W.Strings[i + 1]);}
  W.del();
  uint64_t n = In[0];
  uint64_t m = In[1];

  if (len != n + m + 2) {
    std::cout << "Bad input file: length = "<<len<< " n+m+2 = " << n+m+2 << std::endl;
    exit(-1);
  }
  std::cout<<"checkpoint2"<<std::endl;
  uint64_t* offsets = In+2;
  uint64_t* edges = In+2+n;
  pair_uint *edges_array = (pair_uint *)malloc(m * sizeof(pair_uint));
  parallel_for (uint32_t i=0; i < n; i++) {
    uint64_t o = offsets[i];
    uint64_t l = ((i == n-1) ? m : offsets[i+1])-offsets[i];
    for (uint64_t j = o; j < o+l; j++) {
      edges_array[j] = {i, static_cast<uint32_t>(edges[j])};
    }
  }
  std::cout<<"checkpoint3"<<std::endl;
  *edge_count = m;
  *node_count = n;
  return edges_array;
}


trip_uint *get_wgh_edges_from_file_adj_sym(std::string filename,
                              uint64_t *edge_count, uint32_t *node_count, [[maybe_unused]] bool print = true) {
  long length;
  char *S = readStringFromFile(filename.c_str(), &length);
  words W = stringToWords(S, length);
  if (strcmp(W.Strings[0], "WeightedAdjacencyGraph") != 0) {
    std::cout << "Bad input file: missing header: WeightedAdjacencyGraph" << std::endl;
    exit(-1);
  }
  uint64_t len = W.m -1;
  uint64_t * In = newA(uint64_t, len);
  {parallel_for(uint64_t i=0; i < len; i++) In[i] = atol(W.Strings[i + 1]);}
  W.del();
  uint64_t n = In[0];
  uint64_t m = In[1];

  if (len != n + 2*m + 2) {
    std::cout << "Bad input file: length = "<<len<< " n+2*m+2 = " << n+2*m+2 << std::endl;
    exit(-1);
  }
  uint64_t* offsets = In+2;
  uint64_t* edges = In+2+n;
  uint64_t* weights = In+2+n+m;
  trip_uint *edges_array = (trip_uint *)malloc(m * sizeof(trip_uint));
  parallel_for (uint32_t i=0; i < n; i++) {
    uint64_t o = offsets[i];
    uint64_t l = ((i == n-1) ? m : offsets[i+1])-offsets[i];
    for (uint64_t j = o; j < o+l; j++) {
      edges_array[j] = {i, static_cast<uint32_t>(edges[j]), static_cast<uint32_t>(weights[j])};
    }
  }
  *edge_count = m;
  *node_count = n;
  return edges_array;
}

trip_uint * get_wgh_edges_from_file(const char *filename, int zero_indexed, bool add_both_directions, uint64_t *edge_count, uint32_t *node_count) {
  printf("counting lines in file %s\n", filename);
  FILE *fp;
  fp = fopen(filename, "r");
  setvbuf ( fp , NULL , _IOFBF , 1<<30 );
  size_t buf_size = 64;
  char *line = (char *) malloc(buf_size);
  uint64_t line_count = 0;
  if (fp) {
    while (getline(&line, &buf_size, fp) != -1) {
      if (line[0] == '#') continue;
      line_count++;
    }
    // return 0;
  } else {
    printf("file was not opened\n");
    exit(EXIT_FAILURE);
  }
  if (add_both_directions) {
    line_count *= 2;
  }
  *edge_count = line_count;
  rewind(fp);
  trip_uint *edges = (trip_uint *) malloc(line_count * sizeof(trip_uint));
  uint32_t num_nodes = 0;
  uint64_t index = 0;
  while (getline(&line, &buf_size, fp) != -1) {
    if (line[0] == '#') continue;
    /*
    uint32_t elem_1 = edges[i].x;
    uint32_t elem_2 = edges[i].y;
    uint32_t elem_3 = edges[i].z;
    */
    uint32_t elem_1 = 0;
    uint32_t elem_2 = 0;
    uint32_t elem_3 = 0;

    sscanf(line, "%u   %u   %u", &elem_1, &elem_2, &elem_3);
    num_nodes = std::max(num_nodes, elem_1);
    num_nodes = std::max(num_nodes, elem_2);
    uint32_t src = elem_1 + zero_indexed;
    uint32_t dest = elem_2 + zero_indexed;
    num_nodes = std::max(num_nodes, src);
    num_nodes = std::max(num_nodes, dest);
    src--;
    dest--;
    edges[index++] = {src,dest, elem_3};
    if (add_both_directions) {
      edges[index++] = {dest, src, elem_3};
    }
  }
  printf("weighted num edges %lu\n", index);
  fclose(fp);
    // return 0;
  free(line);
  *node_count = num_nodes;
  return edges;
}


/*
        Gets edges from file and puts the in a malloced buffer
#include "parse_command_line.h"
        reads the file twice to get count of edges so we don't pay a factor of 2 on space
*/
pair_uint * get_edges_from_file(const char *filename, int zero_indexed, bool add_both_directions, uint64_t *edge_count, uint32_t *node_count) {
  printf("counting lines in file %s\n", filename);
  FILE *fp;
  fp = fopen(filename, "r");
  setvbuf ( fp , NULL , _IOFBF , 1<<20 );
  size_t buf_size = 64;
  char *line = (char *) malloc(buf_size);
  uint64_t line_count = 0;
  if (fp) {
    while (getline(&line, &buf_size, fp) != -1) {
      if (line[0] == '#') continue;
      line_count++;
    }
    // return 0;
  } else {
    printf("file was not opened\n");
    exit(EXIT_FAILURE);
  }
  if (add_both_directions) {
    line_count *= 2;
  }
  *edge_count = line_count;
  printf("getting edges from file %s\n", filename);
  rewind(fp);
  pair_uint *edges = (pair_uint *) malloc(line_count * sizeof(pair_uint));
  uint32_t num_nodes = 0;
  uint64_t index = 0;
  while (getline(&line, &buf_size, fp) != -1) {
    if (line[0] == '#') continue;
    uint32_t elem_1;
    uint32_t elem_2;
    sscanf(line, "%u   %u", &elem_1, &elem_2);
    num_nodes = std::max(num_nodes, elem_1);
    num_nodes = std::max(num_nodes, elem_2);
    uint32_t src = elem_1 + zero_indexed;
    uint32_t dest = elem_2 + zero_indexed;
    num_nodes = std::max(num_nodes, src);
    num_nodes = std::max(num_nodes, dest);
    src--;
    dest--;
    edges[index++] = {src,dest};
    if (add_both_directions) {
      edges[index++] = {dest, src};
    }
  }
  fclose(fp);
    // return 0;
  free(line);
  *node_count = num_nodes;
  return edges;
}

typedef std::pair<uint32_t, uint32_t> upair;
std::map<upair, uint32_t> get_unique_edges_from_file(const char *filename) {
  FILE *fp;
  fp = fopen(filename, "r");
  setvbuf ( fp , NULL , _IOFBF , 1<<20 );
  size_t buf_size = 64;
  char *line = (char *) malloc(buf_size);
  // uint64_t line_count = 0;
  if (!fp) {
    printf("file was not opened\n");
    exit(EXIT_FAILURE);
  }
  std::map<upair, uint32_t> edges;
  printf("getting edges from file %s\n", filename);
  while (getline(&line, &buf_size, fp) != -1) {
    if (line[0] == '#') continue;
    uint32_t src;
    uint32_t dest;
    sscanf(line, "%u   %u", &src, &dest);
    // add only one direction
    uint32_t val = rand_in_range(MAXVAL) + 1;
    if (edges.count(upair(src, dest)) == 0 && 
      edges.count(upair(dest, src)) == 0 ) {
      edges.insert(std::pair<upair, uint32_t>(upair(src, dest), val));
      // edges.insert(std::pair<std::pair<uint32_t, uint32_t>(dest, src), val>);
    }
  }
  fclose(fp);
    // return 0;
  free(line);
  return edges;
}
