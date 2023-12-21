
#include "timer.hpp"


void Timer::Start()
{
	//A = chrono::system_clock::now();
	gettimeofday( &StartingTime, NULL );
}


float Timer::Finish()
{
	//B = std::chrono::system_clock::now();
	//chrono::duration<double> elapsed_seconds = B - A;
	//time_t finish_time = std::chrono::system_clock::to_time_t(B);
	//cout << "title" << elapsed_seconds.count()*1000;
	timeval PausingTime, ElapsedTime;
	gettimeofday( &PausingTime, NULL );
	timersub(&PausingTime, &StartingTime, &ElapsedTime);
	float d = ElapsedTime.tv_sec*1000.0+ElapsedTime.tv_usec/1000.0;
	return d;
}
