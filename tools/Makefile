astat_test: astat.cpp ../src/compat/platform_pipe.cpp
	c++ -g -DASTAT_DEBUG -DDEFINE_TEST_MAIN astat.cpp ../src/compat/platform_pipe.cpp -I../src -pthread -o astat_test

astat_lib: astat.a

astat.a: astat.cpp ../src/compat/platform_pipe.cpp
	c++ -fpic -c -std=c++11 astat.cpp ../src/compat/platform_pipe.cpp -I../src -pthread
	ar rcs astat.a astat.o platform_pipe.o

convert:../src/video_codec.c ../src/compat/platform_time.c convert.cpp ../src/debug.cpp ../src/utils/misc.cpp
	$(CXX) -std=gnu++17 -D_GNU_SOURCE -I../src/ -xc ../src/video_codec.c ../src/compat/platform_time.c -xc++ convert.cpp ../src/utils/color_out.cpp ../src/debug.cpp ../src/utils/misc.cpp -o convert

decklink_temperature: decklink_temperature.cpp ../ext-deps/DeckLink/Linux/DeckLinkAPIDispatch.cpp
	$(CXX) $^ -o $@

uyvy2yuv422p: uyvy2yuv422p.c
	$(CC) -g -std=c99 -Wall $< -o $@

TARGETS=astat_lib astat_test convert decklink_temperature uyvy2yuv422p

all: $(TARGETS)

clean:
	$(RM) $(TARGETS) *o
