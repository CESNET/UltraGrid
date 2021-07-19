OPTS	  = -Wall -g -pg
CXXFLAGS += ${OPTS} -std=gnu++0x

CUDA_COMPILER	 ?= nvcc
CUDA_FLAGS ?=

TARGET    = ldgm-encode
SRC_DIR   = src
OBJ_DIR   = obj
LINKER 	 ?= ${CUDA_COMPILER} ${CUDA_FLAGS}
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
	${CUDA_COMPILER} ${CUDA_FLAGS} -c -o $@ $<

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	${CXX} ${CXXFLAGS} ${OPTS} -c -o $@ $?
