AUTOMAKE_OPTIONS = subdir-objects no-dependencies

ACLOCAL_AMFLAGS=-I m4
CLEANFILES=

SUBDIRS = . # test/memcheck test/opengl_interop

CUDA_INSTALL_PATH=@CUDA_INSTALL_PATH@
NVCC=${CUDA_INSTALL_PATH}/bin/nvcc

NVCC_CFLAGS= -Xcompiler -fPIC -Xcompiler '@COMMON_FLAGS@' \
	-gencode arch=compute_20,code=sm_20 \
	-gencode arch=compute_11,code=sm_11 \
	-gencode arch=compute_10,code=sm_10 \
	@CUDA_EXTRA_ARCH@

SUFFIXES=.cu

pkgconfigdir = $(libdir)/pkgconfig
library_includedir=$(includedir)/libgpujpeg/
gpujpeg_libincludedir = $(libdir)/libgpujpeg

lib_LTLIBRARIES = libgpujpeg.la
bin_PROGRAMS = gpujpeg
pkgconfig_DATA = libgpujpeg.pc

library_include_HEADERS = libgpujpeg/*.h
nodist_gpujpeg_libinclude_HEADERS = config.h

gpujpeg_SOURCES = src/main.c
gpujpeg_CFLAGS = -std=c99 @COMMON_FLAGS@
gpujpeg_LDADD = libgpujpeg.la
gpujpeg_LDFLAGS = @GPUJPEG_LDFLAGS@

# gpu jpeg library sources
libgpujpeg_la_SOURCES = src/gpujpeg_common.cpp \
			src/gpujpeg_dct_cpu.cpp \
			src/gpujpeg_decoder.cpp \
			src/gpujpeg_encoder.cpp \
			src/gpujpeg_huffman_cpu_decoder.cpp \
			src/gpujpeg_huffman_cpu_encoder.cpp \
			src/gpujpeg_reader.cpp \
			src/gpujpeg_table.cpp \
			src/gpujpeg_writer.cpp

libgpujpeg_la_DEPENDENCIES = @LIBGPUJPEG_CUDA_OBJS@

libgpujpeg_la_LIBADD = $(libgpujpeg_la_DEPENDENCIES)
libgpujpeg_la_LDFLAGS = -export-dynamic -version-info $(GPUJPEG_LIBRARY_VERSION) @GPUJPEG_LDFLAGS@ @GPUJPEG_LIBS@
libgpujpeg_la_CFLAGS = -std=c99 -fPIC @COMMON_FLAGS@
libgpujpeg_la_CXXFLAGS = -fPIC @COMMON_FLAGS@
#libgpujpeg_la_LINK = g++ -fPIC

check-TESTS: tests
tests:
	for testdir in `find ./test -type d` ; do \
		( cd $${testdir} ; make ) \
	done

if DARWIN
static: $(libgpujpeg_la_DEPENDENCIES) all
	[ -d build/tmp/i386 ] || mkdir -p build/tmp/i386
	[ -d build/tmp/x86_64 ] || mkdir -p build/tmp/x86_64
	[ -d build/tmp/universal ] || mkdir -p build/tmp/universal
	for arch in i386 x86_64; do \
		lipo -thin $$arch  .libs/libgpujpeg.a  -output build/tmp/$$arch/libgpujpeg.a ; \
		cd build/tmp/$$arch; ar x libgpujpeg.a; rm libgpujpeg.a; cd -; \
		for file in $(libgpujpeg_la_DEPENDENCIES); do \
			BASENAME=$$(basename $$file) ; \
			lipo -thin $$arch  $$file  -output build/tmp/$$arch/$$BASENAME ; \
			ar ru build/tmp/$$arch/libgpujpeg.a build/tmp/$$arch/$$BASENAME ; \
		done ; \
	done
	for arch in i386 x86_64; do \
		$(AR) rcu build/tmp/$$(arch)/libgpujpeg.a build/tmp/$$(arch)/* ; \
	done
	$(RM) .libs/libgpujpeg.a
	lipo -create build/tmp/i386/libgpujpeg.a build/tmp/x86_64/libgpujpeg.a -output .libs/libgpujpeg.a
else
static: $(libgpujpeg_la_DEPENDENCIES) all
	$(AR) ru .libs/libgpujpeg.a $(libgpujpeg_la_DEPENDENCIES)
	$(RANLIB) .libs/libgpujpeg.a
endif

# Pattern rule for compiling CUDA files
%.cu.o: %.cu
	$(NVCC) $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) -c $< -o $@

build/universal/%.o: build/i386/%.cu.o build/x86_64/%.cu.o
	[ -d build/universal ] || mkdir -p build/universal
	lipo -create $? -output $@

build/i386/%.cu.o: src/%.cu
	[ -d build/i386 ] || mkdir -p build/i386
	$(NVCC) -m32 $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) -c $< -o $@

build/x86_64/%.cu.o: src/%.cu
	[ -d build/x86_64 ] || mkdir -p build/x86_64
	$(NVCC) -m64 $(NVCC_CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) -c $< -o $@

clean-local:
	rm -rf src/*.cu.lo src/*.cu.o
	rm -rf build
