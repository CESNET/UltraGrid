FLAGS ?= -O2 -g -msse4.1
SRCDIR ?= ..
COMMON_FLAGS = $(FLAGS) -D_GNU_SOURCE -I$(SRCDIR)/src/
MKDIR_P = mkdir -p
vpath %.c $(SRCDIR) $(SRCDIR)/tools
vpath %.cpp $(SRCDIR) $(SRCDIR)/tools

TARGETS=astat_lib astat_test convert decklink_temperature uyvy2yuv422p thumbnailgen

all: $(TARGETS)

%.o : %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(COMMON_FLAGS) -c $< -o $@

%.o : %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -std=c++17 $(COMMON_FLAGS) -c $< -o $@

astat_test: astat.cpp src/compat/platform_pipe.o
	c++ -g -DASTAT_DEBUG -DDEFINE_TEST_MAIN $(COMMON_FLAGS) $^ -pthread -o astat_test

astat_lib: astat.a

astat.a: astat.o src/compat/platform_pipe.o
	ar rcs astat.a $^

convert: src/pixfmt_conv.o src/video_codec.o convert.o src/debug.o \
        src/utils/color_out.o src/utils/misc.o src/video_frame.o \
        src/utils/pam.c src/utils/y4m.c
	$(CXX) $^ -o convert

decklink_temperature: decklink_temperature.cpp ext-deps/DeckLink/Linux/DeckLinkAPIDispatch.o
	$(CXX) $^ -o $@

uyvy2yuv422p: uyvy2yuv422p.c
	$(CC) -g -std=c99 -Wall $< -o $@

thumbnailgen: thumbnailgen.o ipc_frame.o ipc_frame_unix.o
	$(CXX) $^ -o $@ -ljpeg

clean:
	$(RM) $(TARGETS) *o
