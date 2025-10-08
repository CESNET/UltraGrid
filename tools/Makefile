FLAGS ?= -Ofast -g
SRCDIR ?= ..
COMMON_FLAGS = $(FLAGS) -D_GNU_SOURCE -I$(SRCDIR)/src/
MKDIR_P = mkdir -p
vpath %.c $(SRCDIR) $(SRCDIR)/tools
vpath %.cpp $(SRCDIR) $(SRCDIR)/tools

ARCH := $(shell uname -m)
ifeq ($(ARCH), x86_64)
    COMMON_FLAGS += -msse4.1
endif

TARGETS=astat_lib astat_test benchmark_ff_convs convert decklink_temperature \
	mux_ivf \
	thumbnailgen uyvy2yuv422p

COMMON_OBJS = src/color_space.o src/debug.o src/video_codec.o src/pixfmt_conv.o \
	src/compat/aligned_malloc.o \
	src/utils/color_out.o src/utils/misc.o src/video_frame.o \
	src/utils/pam.o src/utils/y4m.o \
	ug_stub.o

OBJS=$(shell find . -name '*.o')

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

benchmark_ff_convs.o: benchmark_ff_convs.c ../src/libavcodec/from_lavc_vid_conv.c \
	../src/libavcodec/to_lavc_vid_conv.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(COMMON_FLAGS) -c $< -o $@

benchmark_ff_convs: benchmark_ff_convs.o $(COMMON_OBJS) \
	src/libavcodec/utils.o src/libavcodec/lavc_common.o \
	src/utils/parallel_conv.o src/utils/worker.o src/utils/thread.o
	$(CXX) $^ -o $@ -lavutil -lavcodec -pthread

convert: convert.o $(COMMON_OBJS)
	$(CXX) $^ -o convert

decklink_temperature: decklink_temperature.cpp ext-deps/DeckLink/Linux/DeckLinkAPIDispatch.o
	$(CXX) $^ -o $@

mux_ivf: mux_ivf.o
	$(CC) $^ -o $@

uyvy2yuv422p: uyvy2yuv422p.c
	$(CC) -g -std=c99 -Wall $< -o $@

thumbnailgen: thumbnailgen.o ipc_frame.o ipc_frame_unix.o
	$(CXX) $^ -o $@ -ljpeg

clean:
	$(RM) $(TARGETS) $(OBJS)
