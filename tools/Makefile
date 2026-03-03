FLAGS ?= -O3 -g
SRCDIR ?= ..
LAVC_CFLAGS = $(shell pkg-config -cflags libavcodec 2>/dev/null)
LAVC_LIBS = $(shell pkg-config -libs libavcodec 2>/dev/null)
COMMON_FLAGS = $(FLAGS) $(LAVC_CFLAGS) -D_GNU_SOURCE -I$(SRCDIR)/src/
MKDIR_P = mkdir -p
vpath %.c $(SRCDIR) $(SRCDIR)/tools
vpath %.cpp $(SRCDIR) $(SRCDIR)/tools

CFLAGS   := $(CFLAGS) -std=gnu2x
CXXFLAGS := $(CXXFLAGS) -std=gnu++17

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
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c $< -o $@

%.o : %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(COMMON_FLAGS) -c $< -o $@

astat_test: astat.cpp src/compat/platform_pipe.o
	c++ -g $(CXXFLAGS) -DASTAT_DEBUG -DDEFINE_TEST_MAIN $(COMMON_FLAGS) $^ -pthread -o astat_test

astat_lib: astat.a

astat.a: astat.o src/compat/platform_pipe.o
	ar rcs astat.a $^

benchmark_ff_convs.o: benchmark_ff_convs.c ../src/libavcodec/from_lavc_vid_conv.c \
	../src/libavcodec/to_lavc_vid_conv.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) $(COMMON_FLAGS) -c $< -o $@

benchmark_ff_convs: benchmark_ff_convs.o $(COMMON_OBJS) \
	src/libavcodec/utils.o src/libavcodec/lavc_common.o \
	src/utils/parallel_conv.o src/utils/worker.o src/utils/thread.o \
	src/from_planar.o src/to_planar.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LAVC_LIBS) -lavutil -pthread

convert: convert.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o convert

decklink_temperature: decklink_temperature.cpp ext-deps/DeckLink/Linux/DeckLinkAPIDispatch.o
	$(CXX) $(CXXFLAGS) $^ -o $@

mux_ivf: mux_ivf.o
	$(CC) $(CFLAGS) $^ -o $@

uyvy2yuv422p: uyvy2yuv422p.c
	$(CC) $(CFLAGS) -g -Wall $< -o $@

thumbnailgen: thumbnailgen.o ipc_frame.o ipc_frame_unix.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -ljpeg

clean:
	$(RM) $(TARGETS) $(OBJS)
