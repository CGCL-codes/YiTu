#include "argument_parsing.cuh"


    
ArgumentParser::ArgumentParser(int argc, char **argv, bool canHaveSource, bool canHaveItrs)
{
	this->argc = argc;
	this->argv = argv;
	this->canHaveSource = canHaveSource;
	this->canHaveItrs = canHaveItrs;
	
	this->sourceNode = 0;
	this->deviceID = 0;
	this->numberOfItrs = 1;
	
	hasInput = false;
	hasSourceNode = false;
	hasOutput = false;
	hasDeviceID = false;
	hasNumberOfItrs = false;
	isDistributed = false;
	isSynchronize = false;

	Parse();
}
	
bool ArgumentParser::Parse()
{
	try
	{
		if(argc == 1)
		{
			cout << GenerateHelpString();
			exit(0);
		}
		
		if(argc == 2) 
			if ((strcmp(argv[1], "--help") == 0) || 
				(strcmp(argv[1], "-help") == 0) || 
				(strcmp(argv[1], "--h") == 0) || 
				(strcmp(argv[1], "-h") == 0))
			{
				cout << GenerateHelpString();
				exit(0);
			}
		
		if(argc%2 == 0)
		{
			cout << "\nThere was an error parsing command line arguments\n";
			cout << GenerateHelpString();
			exit(0);
		}
		
			
		for(int i=1; i<argc-1; i=i+2)
		{
			//argv[i]
			
			if (strcmp(argv[i], "--input") == 0) {
				input = string(argv[i+1]);
				hasInput = true;
			}
			else if (strcmp(argv[i], "--output") == 0) {
				output = string(argv[i+1]);
				hasOutput = true;
			}
			else if (strcmp(argv[i], "--source") == 0 && canHaveSource) {
				sourceNode = atoi(argv[i+1]);
				hasSourceNode = true;
			}
			else if (strcmp(argv[i], "--device") == 0) {
				deviceID = atoi(argv[i+1]);
				hasDeviceID = true;
				cudaSetDevice(deviceID);
			}
			else if (strcmp(argv[i], "--iteration") == 0 && canHaveItrs) {
				numberOfItrs = atoi(argv[i+1]);
				hasNumberOfItrs = true;
			}
			else if (strcmp(argv[i], "--sync") == 0) {
				if (atoi(argv[i + 1]) == 1) isSynchronize = true;
			}
			else if (strcmp(argv[i], "--dist") == 0) {
				if (atoi(argv[i + 1]) == 1) isDistributed = true;
			}
			else
			{
				cout << "\nThere was an error parsing command line argument <" << argv[i] << ">\n";
				cout << GenerateHelpString();
				exit(0);
			}
		}
		
		if(hasInput)
			return true;
		else
		{
			cout << "\nInput graph file argument is required.\n";
			cout << GenerateHelpString();
			exit(0);
		}
	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n";
		GenerateHelpString();
		exit(0);
	}
	catch(...) {
		std::cerr << "An exception has occurred.\n";
		GenerateHelpString();
		exit(0);
	}
}

string ArgumentParser::GenerateHelpString(){
	string str = "\nRequired arguments:";
	str += "\n    [--input]: Input graph file. E.g., --input FacebookGraph.txt";
	str += "\nOptional arguments";
	if(canHaveSource)
		str += "\n    [--source]:  Begins from the source (Default: 0). E.g., --source 10";
	str += "\n    [--output]: Output file for results. E.g., --output results.txt";
	str += "\n    [--device]: Select GPU device (default: 0). E.g., --device 1";
	if(canHaveItrs)
		str += "\n    [--iteration]: Number of iterations (default: 1). E.g., --iterations 10";
	str += "\n\n";
	return str;
}

