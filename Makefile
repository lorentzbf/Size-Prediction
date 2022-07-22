CXX = icc

CFLAGS = -qopenmp -std=c++17

#CFLAGS += -g
CFLAGS += -O2

#CFLAGS += -DCPP
CFLAGS += -DTBB


LDFLAGS = -L "/opt/intel/oneapi/mkl/latest/lib/intel64" -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm

LDFLAGS += -ltbbmalloc

BIN = ./bin
SRC = ./src
OBJ = ./obj
INC = ./inc

INCLUDE = -I/opt/intel/oneapi/tbb/latest/include/tbb
INCLUDE += -I$(INC)


OBJ_LIB = $(OBJ)/CSR.o $(OBJ)/Timings.o

COMMON_DEP = common.h define.h 

$(OBJ)/%.o : $(SRC)/%.cpp 
	mkdir -p $(dir $@)
	$(CXX) -c $(CFLAGS) $(INCLUDE) -o $@ $<


estimate_perf : $(OBJ_LIB) $(OBJ)/estimate_perf.o
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

estimate_accu : $(OBJ_LIB) $(OBJ)/estimate_accu.o
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

$(target) : $(OBJ_LIB) $(OBJ)/$(target).o
	$(CXX)  -o $@ $^ $(CFLAGS) $(LDFLAGS) $(INCLUDE)

all: $(target)


clean :
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*
