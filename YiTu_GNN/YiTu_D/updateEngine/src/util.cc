#include <iostream>
#include <sstream>
#include <fstream>

#include <vector>
#include <set>
#include <unordered_set>
#include <bitset>
#include <cassert>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>

#include "util.h"

namespace graphstore {
	/*return the integer representation of the base */

	float cal_time_elapsed(struct timeval* start, struct timeval* end)
	{
		struct timeval elapsed;
		if (start->tv_usec > end->tv_usec) {
			end->tv_usec += 1000000;
			end->tv_sec--;
		}
		elapsed.tv_usec = end->tv_usec - start->tv_usec;
		elapsed.tv_sec = end->tv_sec - start->tv_sec;
		return (elapsed.tv_sec * 1000000 + elapsed.tv_usec)/1000000.f;
	}

	void print_time_elapsed(std::string desc, struct timeval* start, struct
													timeval* end)
	{
		struct timeval elapsed;
		if (start->tv_usec > end->tv_usec) {
			end->tv_usec += 1000000;
			end->tv_sec--;
		}
		elapsed.tv_usec = end->tv_usec - start->tv_usec;
		elapsed.tv_sec = end->tv_sec - start->tv_sec;
		float time_elapsed = (elapsed.tv_sec * 1000000 + elapsed.tv_usec)/1000000.f;
		std::cout << desc << "Total Time Elapsed: " << std::to_string(time_elapsed) << 
			" seconds" << std::endl;
  std::cout << std::flush;
	}

	std::vector<uint32_t> get_random_permutation(uint32_t num) {
		std::vector<uint32_t> perm(num);
		std::vector<uint32_t> vec(num);

		for (uint32_t i = 0; i < num; i++)
			vec[i] = i;

		uint32_t cnt{0};
		while (vec.size()) {
			uint32_t n = vec.size();
			srand(time(NULL));
			uint32_t idx = rand() % n;
			uint32_t val = vec[idx];
			std::swap(vec[idx], vec[n-1]);
			vec.pop_back();
			perm[cnt++] = val;
		}
		return perm;
	}
}
