#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <stdlib.h>

#include <set>
#include <unordered_set>
#include <vector>
#include <iostream>

#include <string.h>

// #include "PMA.hpp"

namespace graphstore {

#define PREFETCH 1

#if WEIGHTED
#define NUM_IN_PLACE_NEIGHBORS 14
#else
#define NUM_IN_PLACE_NEIGHBORS 13
#endif
// #define MEDIUM_DEGREE (1ULL << 30)
#define MEDIUM_DEGREE (1ULL << 10)

#define LOCK_MASK (1ULL << 31)
#define UNLOCK_MASK ~(1ULL << 31)

// A 0 bit means the lock is free
// A 1 bit means the lock is currently acquired
static inline void lock(uint32_t *data)
{
   while ((__sync_fetch_and_or(data, LOCK_MASK) & (1ULL << 31)) != 0) {}
}

static inline void unlock(uint32_t *data)
{
   __sync_fetch_and_and(data, UNLOCK_MASK);
}

}
#endif