

demo: demo.o cuda_dxt.o
	nvcc -o demo demo.o cuda_dxt.o

demo.o: demo.c cuda_dxt.h
	gcc -I /usr/local/cuda/include/ -std=c89 -pedantic -Wextra -Wno-long-long -O3 -c demo.c

cuda_dxt.o: cuda_dxt.cu cuda_dxt.h
	nvcc $(INCLUDES) -O3 --ptxas-options=-v -arch=sm_20 -c -o cuda_dxt.o cuda_dxt.cu 

clean:
	rm -f *.o demo
