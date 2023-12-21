
CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_60

ROOT_DIR= $(shell pwd)
CXX?= g++
CXXFLAGS?= -O3 -Wall -std=c++11 -g -fopenmp -I$(ROOT_DIR)

SHARED=../shared

all: converter

converter: converter.cpp
	$(CC) converter.cpp -o converter $(CFLAGS)

clean:
	rm -f converter
