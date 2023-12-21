#ifndef TIMER_HPP
#define TIMER_HPP


#include "globals.hpp"
#include <stdlib.h>
#include <sys/time.h>


class Timer
{
private:
	//chrono::time_point<chrono::system_clock> A, B;
	timeval StartingTime;
public:
    void Start();
    float Finish();
};

#endif	//	TIMER_HPP



