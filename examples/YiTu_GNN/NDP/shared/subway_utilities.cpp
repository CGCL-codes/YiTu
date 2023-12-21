
#include "subway_utilities.hpp"

void utilities::PrintResults(uint *results, uint n)
{
	cout << "Results of first "<< n << " nodes:\n[";
	for(int i=0; i<n; i++)
	{
		if(i>0)
			cout << " ";
		cout << i << ":" << results[i];
	}
	cout << "]\n";
}

void utilities::PrintResults(float *results, uint n)
{
	cout << "Results of first "<< n << " nodes:\n[";
	for(int i=0; i<n; i++)
	{
		if(i>0)
			cout << " ";
		cout << i << ":" << results[i];
	}
	cout << "]\n";
}

void utilities::PrintResults(double *results, uint n)
{
	cout << "Results of first "<< n << " nodes:\n[";
	for(int i=0; i<n; i++)
	{
		if(i>0)
			cout << " ";
		cout << i << ":" << results[i];
	}
	cout << "]\n";
}

void utilities::SaveResults(string filepath, uint *results, uint n)
{
	cout << "Saving the results into the following file:\n";
	cout << ">> " << filepath << endl;
	ofstream outfile;
	outfile.open(filepath);
	for(int i=0; i<n; i++)
		outfile << i << " " << results[i] << endl;
	outfile.close();
	cout << "Done saving.\n";
}

void utilities::SaveResults(string filepath, float *results, uint n)
{
	cout << "Saving the results into " << filepath << " ...... " << flush;
	ofstream outfile;
	outfile.open(filepath);
	for(int i=0; i<n; i++)
		outfile << i << " " << results[i] << endl;
	outfile.close();
	cout << " Completed.\n";
}

void utilities::SaveResults(string filepath, double *results, uint n)
{
	cout << "Saving the results into " << filepath << " ...... " << flush;
	ofstream outfile;
	outfile.open(filepath);
	for(int i=0; i<n; i++)
		outfile << i << " " << results[i] << endl;
	outfile.close();
	cout << " Completed.\n";
}
