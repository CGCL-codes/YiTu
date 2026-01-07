#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "host_config.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <map>
#include <algorithm>

#define HERE do {std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;} while(0)

class Vertex {
    public:
        int idx;
        int inDeg;
        int outDeg;

        std::vector<int> inVid;
        std::vector<int> outVid;

        explicit Vertex(int _idx) {
            idx = _idx;
        }

        ~Vertex(){
            // Nothing is done here.
        }

};

class Graph{
    public:
        int vertexNum;
        int edgeNum;
        std::vector<Vertex*> vertices; 

        Graph(const std::string &fName);
        ~Graph(){
			for(int i = 0; i < vertexNum; i++){
				delete vertices[i];
			}
		};
        void getRandomStartIndices(std::vector<int> &startIndices);
        void getStat();

    private:
        bool isUgraph;
        int getMaxIdx(const std::vector<std::vector<int>> &data);
        int getMinIdx(const std::vector<std::vector<int>> &data);
        void loadFile(
				const std::string& fName,
                std::vector<std::vector<int>> &data
				);

};

class CSR{
    public:
		const int vertexNum;
		const int edgeNum;
        std::vector<int> rpao; 
        std::vector<int> ciao;
        std::vector<int> rpai;
        std::vector<int> ciai;
		std::vector<prop_t> eProps;
		std::vector<prop_t> vProps;

        // The CSR is constructed based on the simple graph
        explicit CSR(const Graph &g);
        int save2File(const std::string & fName);
		~CSR();
};

class CSR_BLOCK{
	public:
		const int cordx;
		const int cordy;
		int vertexNum;
		int edgeNum;
		int srcStart;
		int srcEnd;
		int sinkStart;
		int sinkEnd;
		std::vector<int> rpa;
		std::vector<int> cia;
		std::vector<prop_t> eProps;
		explicit CSR_BLOCK(const int _cordx, const int _cordy, CSR* csr);
		~CSR_BLOCK(){};
};

#endif
