#include<pybind11/pybind11.h>
#include "../shared/globals.hpp"

namespace py = pybind11;
bool IsWeightedFormat(string format)
{
	if((format == "bwcsr")	||
		(format == "wcsr")	||
		(format == "wel"))
			return true;
	return false;
}

string GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

int convert(py::str path)
{
	string input = string(path);
	if(GetFileExtension(input) == "el")
	{
		ifstream infile;
		infile.open(input);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		
		vector<Edge> edges;
		Edge newEdge;
		while(getline( infile, line ))
		{
			if(line.substr(0, 1) != "#"){
			ss.str("");
			ss.clear();
			ss << line;
			
			ss >> newEdge.source;
			ss >> newEdge.end;
			
			edges.push_back(newEdge);
			edgeCounter++;
			
			if(max < newEdge.source)
				max = newEdge.source;
			if(max < newEdge.end)
				max = newEdge.end;
			}			
		}			
		infile.close();
		
		uint num_nodes = max + 1;
		uint num_edges = edgeCounter;
		uint *nodePointer = new uint[num_nodes+1];
		OutEdge *edgeList = new OutEdge[num_edges];
		uint *degree = new uint[num_nodes];
		for(uint i=0; i<num_nodes; i++)
			degree[i] = 0;
		for(uint i=0; i<num_edges; i++)
			degree[edges[i].source]++;
		
		uint counter=0;
		for(uint i=0; i<num_nodes; i++)
		{
			nodePointer[i] = counter;
			counter = counter + degree[i];
		}
		uint *outDegreeCounter  = new uint[num_nodes];
		uint location;  
		for(uint i=0; i<num_edges; i++)
		{
			uint location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
			edgeList[location].end = edges[i].end;
			outDegreeCounter[edges[i].source]++;  
		}
		edges.clear();
		delete[] degree;
		delete[] outDegreeCounter;
		
		std::ofstream outfile(input.substr(0, input.length()-2)+"bcsr", std::ofstream::binary);
		
		outfile.write((char*)&num_nodes, sizeof(unsigned int));
		outfile.write((char*)&num_edges, sizeof(unsigned int));
		outfile.write ((char*)nodePointer, sizeof(unsigned int)*num_nodes);
		outfile.write ((char*)edgeList, sizeof(OutEdge)*num_edges);
		
		outfile.close();
	}
	else if(GetFileExtension(input) == "wel")
	{
		ifstream infile;
		infile.open(input);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		
		vector<EdgeWeighted> edges;
		EdgeWeighted newEdge;
		while(getline( infile, line ))
		{
			ss.str("");
			ss.clear();
			ss << line;
			
			ss >> newEdge.source;
			ss >> newEdge.end;
			ss >> newEdge.w8;
			
			edges.push_back(newEdge);
			edgeCounter++;
			
			if(max < newEdge.source)
				max = newEdge.source;
			if(max < newEdge.end)
				max = newEdge.end;				
		}			
		infile.close();
		
		uint num_nodes = max + 1;
		uint num_edges = edgeCounter;
		uint *nodePointer = new uint[num_nodes+1];
		OutEdgeWeighted *edgeList = new OutEdgeWeighted[num_edges];
		uint *degree = new uint[num_nodes];
		for(uint i=0; i<num_nodes; i++)
			degree[i] = 0;
		for(uint i=0; i<num_edges; i++)
			degree[edges[i].source]++;
		
		uint counter=0;
		for(uint i=0; i<num_nodes; i++)
		{
			nodePointer[i] = counter;
			counter = counter + degree[i];
		}
		uint *outDegreeCounter  = new uint[num_nodes];
		uint location;  
		for(uint i=0; i<num_edges; i++)
		{
			uint location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
			edgeList[location].end = edges[i].end;
			edgeList[location].w8 = edges[i].w8;
			outDegreeCounter[edges[i].source]++;  
		}
		edges.clear();
		delete[] degree;
		delete[] outDegreeCounter;
		
		std::ofstream outfile(input.substr(0, input.length()-3)+"bwcsr", std::ofstream::binary);
		
		outfile.write((char*)&num_nodes, sizeof(unsigned int));
		outfile.write((char*)&num_edges, sizeof(unsigned int));
		outfile.write ((char*)nodePointer, sizeof(unsigned int)*num_nodes);
		outfile.write ((char*)edgeList, sizeof(OutEdgeWeighted)*num_edges);
		
		outfile.close();
	}
	else
	{
		cout << "\nInput file format is not supported.\n";
	}

}

PYBIND11_MODULE(Tools, m){
	m.def("convert", &convert);
}
