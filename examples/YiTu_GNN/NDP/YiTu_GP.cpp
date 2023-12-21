#include<pybind11/pybind11.h>
// #include "distributed/bfs_dist/bfs_test.h"
#include<iostream>

#include <mpi.h>

#include "ndp/bfs_sig.h"
#include "ndp/bc_sig.h"
#include "ndp/cc_sig.h"
#include "ndp/pr_sig.h"
#include "ndp/sssp_sig.h"
#include "ndp/sswp_sig.h"

using std::cout;
using std::cerr;
using std::endl;

namespace py = pybind11;

void YiTu_GP(py::str method, py::list argulist){
	int argc = argulist.size();
	char** argv = (char**)malloc(argc*sizeof(char*));
    for(int i = 1; i < argc; i++){
        char* a = (char*)malloc(100*sizeof(char));
        strncpy(a, std::string(py::str(argulist[i])).data(),100);
        argv[i] = a;
        //std::cout<<argv[i]<<std::endl;
    }
	std::string algorithm = std::string(method);
	if(algorithm == "bfs") {
		ArgumentParser arguments(argc, argv, true, false);
		cout << "bfs single-async!" << endl;
		bfs_sig_async(arguments);
	}
	else if(algorithm == "bc") {
		ArgumentParser arguments(argc, argv, true, false);
		cout << "bc single-async!" << endl;
		bc_sig_async(arguments);
	}
	else if(algorithm == "cc") {
		ArgumentParser arguments(argc, argv, true, false);
		cout << "cc single-async!" << endl;
		cc_sig_async(arguments);
	}
	else if(algorithm == "pr") {
		ArgumentParser arguments(argc, argv, true, false);
		cout << "pr single-async!" << endl;
		pr_sig_async(arguments);
	}
	else if(algorithm == "sssp") {
		ArgumentParser arguments(argc, argv, true, false);
		cout << "sssp single-async!" << endl;
		sssp_sig_async(arguments);
	}
	else if(algorithm == "sswp") {
		ArgumentParser arguments(argc, argv, true, false);
		cout << "sswp single-async!" << endl;
		sswp_sig_async(arguments);
	}
	else {
		cout << "algorithm " << algorithm << " not support!" << endl;
		cout << "support algorithm: bfs, bc, cc, pr, sssp, sswp!" << endl;
	}
    
    free(argv);
}

void YiTu_GNN(py::str method, py::str augrStr){
	std::string model = std::string(method);
	std::string cmdArgu = std::string(augrStr);
	std::string cmd;
	if(model == "gcn") {
		cmd = "python examples/profile/NDP_gcn.py " + cmdArgu;
	}
	else if(model == "gin") {
		cmd = "python examples/profile/NDP_gin.py " + cmdArgu;
	}
	else if(model == "gs") {
		cmd = "python examples/profile/NDP_gs.py " + cmdArgu;
	}
	else if(model == "iso") {
		cmd = "python examples/profile/NDP_iso.py " + cmdArgu;
	}
	else {
		cout << "model " << model << " not support!" << endl;
		cout << "support model: gcn, gin, gs, iso!" << endl;
	}

	FILE* pp = popen(cmd.data(), "r"); // build pipe
	if (!pp) return;
	// collect cmd execute result
	char tmp[1024];
	while (fgets(tmp, sizeof(tmp) * 1024, pp) != NULL)
		std::cout << tmp << std::endl; // can join each line as string
	pclose(pp);
}

PYBIND11_MODULE(YiTu_GP, m){
	m.def("YiTu_GP", &YiTu_GP);
	m.def("YiTu_GNN", &YiTu_GNN);
}