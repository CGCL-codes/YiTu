TARGETS= LSGraph

ifdef D
	DEBUG=-g -DDEBUG_MODE
	OPT=
else
	DEBUG=
# OPT=-Ofast -DNDEBUG
	OPT=-O3 -fno-finite-math-only -DNDEBUG
endif

ifdef NH
	ARCH=
else
	#ARCH=-msse4.2 -D__SSE4_2_
	ARCH=-march=native
endif

ifdef P
	PROFILE=-pg -no-pie # for bug in gprof.
endif

# sudo apt install libstdc++-8-dev
CXX = clang++ -std=c++17
CC = clang -std=gnu11
LD= clang++ -std=c++17

LOC_INCLUDE=include
LOC_LIB=lib
LOC_SRC=src
OBJDIR=obj
SER=ser
CILK=1

CXXFLAGS += -mcx16 -Wall $(DEBUG) -g $(PROFILE) $(OPT) $(ARCH) -DOPENMP=$(OPENMP) -DCILK=$(CILK) -DPARLAY_OPENCILK -m64 -I. -I$(LOC_INCLUDE)

CFLAGS += -mcx16 -Wall $(DEBUG) $(PROFILE) $(OPT) $(ARCH) -DOPENMP=$(OPENMP) -DCILK=$(CILK) -DPARLAY_OPENCILK -m64 -I. -I$(LOC_INCLUDE)

LDFLAGS += $(DEBUG) $(PROFILE) $(OPT) -L$(LOC_LIB) -lm -lpthread -lssl -lcrypto -ldl

OPENMP?=0
ifeq ($(OPENMP),1)
  CILK=0
else
  CILK?=0
endif

ifeq ($(CILK),1)
  CFLAGS += -fcilkplus
  CXXFLAGS += -fcilkplus
  LDFLAGS += -lcilkrts
endif

ifeq ($(OPENMP),1)
  CFLAGS += -fopenmp
  CXXFLAGS += -fopenmp
  LDFLAGS += -lomp
endif

LDFLAGS +="-Wl,-rpath,lib/"
all: LSGraph
LSGraph:							$(OBJDIR)/LSGraph.o \
												$(OBJDIR)/util.o
# dependencies between .o files and .cc (or .c) files
$(OBJDIR)/LSGraph.o: 					$(LOC_SRC)/LSGraph.cc \
																	$(LOC_INCLUDE)/graph.h \
																	$(LOC_INCLUDE)/util.h
#
# generic build rules
#

$(TARGETS):
	$(LD) $^ $(LDFLAGS) -o $@ 

$(OBJDIR)/%.o: $(LOC_SRC)/%.cc | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $@ $<

$(OBJDIR)/%.o: $(LOC_SRC)/%.c | $(OBJDIR)
	$(CXX) $(CFLAGS) $(INCLUDE) -c -o $@ $<

$(OBJDIR)/%.o: $(LOC_SRC)/gqf/%.c | $(OBJDIR)
	$(CXX) $(CFLAGS) $(INCLUDE) -c -o $@ $<

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(SER):
	@mkdir -p $(SER)

clean:
	rm -rf $(OBJDIR) core $(TARGETS)
