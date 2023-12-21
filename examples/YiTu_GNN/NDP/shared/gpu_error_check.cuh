#ifndef	GPU_ERROR_CHECK_CUH
#define	GPU_ERROR_CHECK_CUH

//#include <string>
//#include <sstream>
//#include <stdexcept>

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif	//	GPU_ERROR_CHECK_CUH
