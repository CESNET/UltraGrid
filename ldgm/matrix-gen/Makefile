TARGET = matrix-gen

.PHONY: all
.PHONY: clean

all: ${TARGET}

${TARGET} :
	g++ -std=gnu++0x matrix-generator.cpp ldpc-matrix.cpp main.cpp -o $@

clean:
	rm -f ${TARGET}

