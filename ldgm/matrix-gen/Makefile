TARGET = matrix-gen

.PHONY: all
.PHONY: clean

all: ${TARGET}

${TARGET} : matrix-generator.cpp ldpc-matrix.cpp main.cpp
	g++ -std=gnu++0x $? -o $@

clean:
	rm -f ${TARGET}

