
default: help

help:
	@echo "Targets: testperf, tetquality, testperf4k, or testquality4k"

testperf: main
	convert lena.png -depth 8 RGBA:lena.rgba
	./main 256 256 lena.rgba

testquality: 2dxt
	convert lena.png -depth 8 RGBA:lena.rgba
	./2dxt 256 256 1 lena.rgba lena.dxt
	./viewdxt -w 256 -h 256 -p -o -r lena.rgba lena.dxt
	./2dxt 256 256 6 lena.rgba lena.dxt
	./viewdxt -w 256 -h 256 -6 -p -o -r lena.rgba lena.dxt


testperf2k: main
	convert 2k.png -depth 8 RGBA:2k.rgba
	./main 1920 1080 2k.rgba

testquality2k: 2dxt
	convert 2k.png -depth 8 RGBA:2k.rgba
	./2dxt 1920 1080 1 2k.rgba 2k.dxt
	./viewdxt -w 1920 -h 1080 -o -p -r 2k.rgba 2k.dxt
	./2dxt 1920 1080 6 2k.rgba 2k.dxt
	./viewdxt -w 1920 -h 1080 -6 -o -p -r 2k.rgba 2k.dxt

testperf4k: main
	convert 4k.png -depth 8 RGBA:4k.rgba
	./main 4096 2048 4k.rgba

testquality4k: 2dxt
	convert 4k.png -depth 8 RGBA:4k.rgba
	./2dxt 4096 2048 1 4k.rgba 4k.dxt
	./viewdxt -w 4096 -h 2048 -o -p -r 4k.rgba 4k.dxt
	./2dxt 4096 2048 6 4k.rgba 4k.dxt
	./viewdxt -w 4096 -h 2048 -6 -o -p -r 4k.rgba 4k.dxt

clean:
	/bin/rm -f *.rgba *.dxt out.*

