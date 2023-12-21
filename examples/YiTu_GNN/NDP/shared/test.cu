#include "test.cuh"


template <class T>
Test<T>::Test()
{
	this->a = 1;
	this->b = 1;
}

template <class T>
int Test<T>::sum(int a, int b)
{
	return a + b;
}

