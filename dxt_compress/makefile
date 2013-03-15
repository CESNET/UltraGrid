TARGET=dxt_compress
CC=gcc
CXX=g++
CFLAGS=-std=c99 
#-DDEBUG
OBJECTS=dxt_common.c.o dxt_encoder.c.o dxt_decoder.c.o dxt_display.c.o dxt_util.c.o

# Ultragrid changes
ARCH := $(shell uname -s)
ifeq ($(ARCH),Darwin)
    LIBS += /opt/local/lib/libGLEW.a  -framework GLUT -framework OpenGL -framework Cocoa
    CFLAGS += -I/opt/local/include -DHAVE_GLUT
else
    LIBS += -lGL -lglut -lGLU -lGLEW
endif

################################################################################
# Build compressor
#
build: $(TARGET) 

$(TARGET): main.c.o dxt_glsl.h $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(LIBS) -lpthread $< -o $@

main.c.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

dxt_common.c.o: dxt_common.c dxt_common.h
	$(CC) $(CFLAGS) -c $< -o $@
    
dxt_encoder.c.o: dxt_encoder.c dxt_encoder.h dxt_glsl.h
	$(CC) $(CFLAGS) -c $< -o $@

dxt_decoder.c.o: dxt_decoder.c dxt_decoder.h dxt_glsl.h
	$(CC) $(CFLAGS) -c $< -o $@
	
dxt_display.c.o: dxt_display.c dxt_display.h dxt_glsl.h
	$(CC) $(CFLAGS) -c $< -o $@
    
dxt_util.c.o: dxt_util.c dxt_util.h
	$(CC) $(CFLAGS) -c $< -o $@
    
dxt_glsl.h:compress_vp.glsl \
		compress_dxt1_fp.glsl compress_dxt5ycocg_fp.glsl \
		display_fp.glsl display_dxt5ycocg_fp.glsl \
		yuv422_to_yuv444.glsl display_dxt1_yuv_fp.glsl \
		rgba_to_yuv422.glsl
	echo "/**" > $@
	echo " * GLSL source codes for DXT compressions" >> $@
	echo " *" >> $@
	echo " * @author Martin Srom" >> $@
	echo " */" >> $@
	# Write vp_compress
	echo "static const char vp_compress[] = " >> $@
	printf "\"#version 140\\\n\"\n" >> $@
	printf "\"#define legacy 0\\\n\"\n" >> $@
	cat compress_vp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write vp_compress_legacy
	echo "static const char vp_compress_legacy[] = " >> $@
	printf "\"#define legacy 1\\\n\"\n" >> $@
	cat compress_vp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_compress_dxt1
	echo "static const char fp_compress_dxt1[] = " >> $@
	printf "\"#version 140\\\n\"\n" >> $@
	printf "\"#define legacy 0\\\n\"\n" >> $@
	printf "\"#define FORMAT_YUV 0\\\n\"\n" >> $@
	cat  compress_dxt1_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_compress_dxt1 yuv
	echo "static const char fp_compress_dxt1_yuv[] = " >> $@
	printf "\"#version 140\\\n\"\n" >> $@
	printf "\"#define legacy 0\\\n\"\n" >> $@
	printf "\"#define FORMAT_YUV 1\\\n\"\n" >> $@
	cat  compress_dxt1_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_compress_dxt1_legacy
	echo "static const char fp_compress_dxt1_legacy[] = " >> $@
	printf "\"#define legacy 1\\\n\"\n" >> $@
	printf "\"#define FORMAT_YUV 0\\\n\"\n" >> $@
	cat  compress_dxt1_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_compress_dxt1_legacy
	echo "static const char fp_compress_dxt1_yuv_legacy[] = " >> $@
	printf "\"#define legacy 1\\\n\"\n" >> $@
	printf "\"#define FORMAT_YUV 1\\\n\"\n" >> $@
	cat  compress_dxt1_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_compress_dxt5ycocg
	echo "static const char fp_compress_dxt5ycocg[] = " >> $@
	printf "\"#version 140\\\n\"\n" >> $@
	printf "\"#define legacy 0\\\n\"\n" >> $@
	printf "\"#define FORMAT_YUV 0\\\n\"\n" >> $@
	cat compress_dxt5ycocg_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_compress_dxt5ycocg yuv
	echo "static const char fp_compress_dxt5ycocg_yuv[] = " >> $@
	printf "\"#version 140\\\n\"\n" >> $@
	printf "\"#define legacy 0\\\n\"\n" >> $@
	printf "\"#define FORMAT_YUV 1\\\n\"\n" >> $@
	cat compress_dxt5ycocg_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_compress_dxt5ycocg_legacy
	echo "static const char fp_compress_dxt5ycocg_legacy[] = " >> $@
	printf "\"#define legacy 1\\\n\"\n" >> $@
	printf "\"#define FORMAT_YUV 0\\\n\"\n" >> $@
	cat compress_dxt5ycocg_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_compress_dxt5ycocg_legacy yuv
	echo "static const char fp_compress_dxt5ycocg_yuv_legacy[] = " >> $@
	printf "\"#define legacy 1\\\n\"\n" >> $@
	printf "\"#define FORMAT_YUV 1\\\n\"\n" >> $@
	cat compress_dxt5ycocg_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_display_dxt1
	echo "static const char fp_display[] = " >> $@
	printf "\"#version 140\\\n\"\n" >> $@
	printf "\"#define legacy 0\\\n\"\n" >> $@
	cat display_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_display_dxt1 legacy
	echo "static const char fp_display_legacy[] = " >> $@
	printf "\"#define legacy 1\\\n\"\n" >> $@
	cat display_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_display_dxt5ycocg
	echo "static const char fp_display_dxt5ycocg[] = " >> $@
	printf "\"#version 140\\\n\"\n" >> $@
	printf "\"#define legacy 0\\\n\"\n" >> $@
	cat display_dxt5ycocg_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# Write fp_display_dxt5ycocg (legacy)
	echo "static const char fp_display_dxt5ycocg_legacy[] = " >> $@
	printf "\"#define legacy 1\\\n\"\n" >> $@
	cat display_dxt5ycocg_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# yuv 422 to yuv 444 shader
	echo "static const char fp_yuv422_to_yuv_444[] = " >> $@
	printf "\"#version 140\\\n\"\n" >> $@
	printf "\"#define legacy 0\\\n\"\n" >> $@
	cat yuv422_to_yuv444.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# yuv 422 to yuv 444 shader (legacy)
	echo "static const char fp_yuv422_to_yuv_444_legacy[] = " >> $@
	printf "\"#define legacy 1\\\n\"\n" >> $@
	cat yuv422_to_yuv444.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# display YUV dxt1
	echo "static const char fp_display_dxt1_yuv[] = " >> $@
	printf "\"#version 140\\\n\"\n" >> $@
	printf "\"#define legacy 0\\\n\"\n" >> $@
	cat display_dxt1_yuv_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# display YUV dxt1 (legacy)
	echo "static const char fp_display_dxt1_yuv_legacy[] = " >> $@
	printf "\"#define legacy 1\\\n\"\n" >> $@
	cat display_dxt1_yuv_fp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# rgba to yuv 422 shader
	echo "static const char fp_display_rgba_to_yuv422[] = " >> $@
	printf "\"#version 140\\\n\"\n" >> $@
	printf "\"#define legacy 0\\\n\"\n" >> $@
	cat rgba_to_yuv422.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# rgba to yuv 422 shader legacy
	echo "static const char fp_display_rgba_to_yuv422_legacy[] = " >> $@
	printf "\"#define legacy 1\\\n\"\n" >> $@
	cat rgba_to_yuv422.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	echo ";" >> $@
	# rgba to yuv 422 vertex
	#echo "static const char vp_display_rgba_to_yuv422[] = " >> $@
	#cat rgba_to_yuv422_vp.glsl | sed 's/\(.*\)/    \"\1\\n\"/' >> $@
	#echo ";" >> $@


#
################################################################################
    
    
################################################################################
# Clean compressor
#
clean:
	rm -f *.o $(TARGET) dxt_glsl.h
#
################################################################################


################################################################################
# Build glsl sources from Cg
#
GLSL_FILES=compress_dxt1_fp.glsl.build compress_dxt5ycocg_fp.glsl.build display_fp.glsl.build display_dxt5ycocg_fp.glsl.build

glsl: $(GLSL_FILES)
    
compress_dxt1_fp.glsl.build: dxt.cg
	#cgc -profile glslf -po version=130 -entry compress_DXT1_fp -o compress_dxt1_fp.glsl dxt.cg 
	# Replace all "_imageFormat#" by "imageFormat"
	#sed -i s/`cat compress_dxt1_fp.glsl | grep "var int imageFormat" | sed "s/.*\(_imageFormat\).*/\1/"`/imageFormat/g compress_dxt1_fp.glsl
    # Replace all "_imageSize#" by "imageSize"
	#sed -i s/`cat compress_dxt1_fp.glsl | grep "var float2 imageSize" | sed "s/.*\(_imageSize[0-9]\).*/\1/"`/imageSize/g compress_dxt1_fp.glsl
	# Remove comments lines
	#sed -i 's/^ *\/\/.*//g' compress_dxt1_fp.glsl
	# Remove empty lines
	#sed -i '/^$$/d' compress_dxt1_fp.glsl
    
compress_dxt5ycocg_fp.glsl.build: dxt.cg
	#cgc -profile glslf -po version=130 -entry compress_YCoCgDXT5_fp -o compress_dxt5ycocg_fp.glsl dxt.cg	
	# Replace all "_imageFormat#" by "imageFormat"
	#sed -i s/`cat compress_dxt5ycocg_fp.glsl | grep "var int imageFormat" | sed "s/.*\(_imageFormat\).*/\1/"`/imageFormat/g compress_dxt5ycocg_fp.glsl
	# Replace all "_imageSize#" by "imageSize"
	#sed -i s/`cat compress_dxt5ycocg_fp.glsl | grep "var float2 imageSize" | sed "s/.*\(_imageSize[0-9]\).*/\1/"`/imageSize/g compress_dxt5ycocg_fp.glsl
	# Remove comments lines
	#sed -i 's/^ *\/\/.*//g' compress_dxt5ycocg_fp.glsl
	# Remove empty lines
	#sed -i '/^$$/d' compress_dxt5ycocg_fp.glsl
    
display_fp.glsl.build: dxt.cg
	cgc -profile glslf -entry display_fp -o display_fp.glsl dxt.cg 
	# Remove comments lines
	sed -i 's/^ *\/\/.*//g' display_fp.glsl
	# Remove empty lines
	sed -i '/^$$/d' display_fp.glsl
    
display_dxt5ycocg_fp.glsl.build: dxt.cg
	cgc -profile glslf -entry display_YCoCgDXT5_fp -o display_dxt5ycocg_fp.glsl dxt.cg 
	# Remove comments lines
	sed -i 's/^ *\/\/.*//g' display_dxt5ycocg_fp.glsl
	# Remove empty lines
	sed -i '/^$$/d' display_dxt5ycocg_fp.glsl
#
################################################################################
