

APPS := rgb2dxt1 rgb2dxt6 dxt62tga

all: $(APPS)
	@echo "DONE"

rgb2dxt%: rgb2dxt%.o cuda_dxt.o
	nvcc -o $@ $^

rgb2dxt%.o: rgb2dxt%.c cuda_dxt.h
	gcc -I /usr/local/cuda/include/ -std=c89 -pedantic -Wextra -Wno-long-long -O3 -c $<

dxt62tga: dxt62tga.c
	gcc -std=c89 -pedantic -Wextra -O3 -o $@ $<

cuda_dxt.o: cuda_dxt.cu cuda_dxt.h
	nvcc -O3 --ptxas-options=-v -arch=sm_20 -c -o $@ $< 

clean:
	rm -f *.o $(APPS)
