#ifndef SUBWAY_UTILITIES_HPP
#define SUBWAY_UTILITIES_HPP


#include "globals.hpp"

namespace utilities {
	void PrintResults(uint *results, uint n);
	void PrintResults(float *results, uint n);
	void PrintResults(double *results, uint n);
	void SaveResults(string filepath, uint *results, uint n);
	void SaveResults(string filepath, float *results, uint n);
	void SaveResults(string filepath, double *results, uint n);
}

#endif	//	SUBWAY_UTILITIES_HPP
