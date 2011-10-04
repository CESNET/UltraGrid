TARGET=dxt_compress
CC=gcc
CXX=g++
CFLAGS=-std=c99
OBJECTS=dxt_common.c.o dxt_encoder.c.o dxt_decoder.c.o dxt_display.c.o

build: $(TARGET) 

$(TARGET): main.c.o $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -lglut -lGLU -lGLEW -lCg -lCgGL -lpthread $< -o $@

main.c.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

dxt_common.c.o: dxt_common.c dxt_common.h
	$(CC) $(CFLAGS) -c $< -o $@
    
dxt_encoder.c.o: dxt_encoder.c dxt_encoder.h
	$(CC) $(CFLAGS) -c $< -o $@

dxt_decoder.c.o: dxt_decoder.c dxt_decoder.h
	$(CC) $(CFLAGS) -c $< -o $@
	
dxt_display.c.o: dxt_display.c dxt_display.h
	$(CC) $(CFLAGS) -c $< -o $@
	
clean:
	rm -f *.o $(TARGET) $(SHADER)
