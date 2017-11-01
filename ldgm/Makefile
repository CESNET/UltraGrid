OPTS	  = -Wall -g -pg
CXXFLAGS += ${OPTS} -std=gnu++0x

NVCC	 ?= nvcc
NVCCFLAGS += -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35

TARGET    = ldgm-encode
SRC_DIR   = src
OBJ_DIR   = obj
LINKER 	 ?= ${NVCC} ${NVCCFLAGS}
LIBS	 += -lrt
LDFLAGS  +=

.PHONY: all
.PHONY: clean

CPP_FILES = $(wildcard $(SRC_DIR)/*.cpp)
CU_FILES  = $(wildcard $(SRC_DIR)/*.cu)

H_FILES   = $(wildcard $(SRC_DIR)/*.h)
CUH_FILES = $(wildcard $(SRC_DIR)/*.cuh)

OBJ_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
	CUO_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.

OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))
OBJS += $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CU_FILES)))

all: obj ${TARGET}

obj:
	mkdir -p obj

clean:
	clear
	rm -f obj/*.o ${TARGET}
	
$(TARGET) : $(OBJS)
	${LINKER} ${LDFLAGS} $? -o $@ ${LIBS}

$(OBJ_DIR)/%.cu.o : $(SRC_DIR)/%.cu
	${NVCC} ${NVCCFLAGS} -c -o $@ $<

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	${CXX} ${CXXFLAGS} ${OPTS} -c -o $@ $?
