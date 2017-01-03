Name:		ultragrid
Version:	1.3
Release:	20150604.00
Summary:	Software for real-time transmissions of high-definition video
Group:		Applications/Multimedia

License:	GPL
URL:		https://sitola.fi.muni.cz/igrid/index.php/UltraGrid
Source0:	ultragrid-%{version}.tar.bz2
#%if %{defined fedora}
%if 0%{?fedora} < 21
Patch0:		cuda-6.5-configure.patch
%else
Patch0:		cuda-7.0-configure.patch
%endif
#%endif

BuildRequires:	gcc-c++,make,automake,autoconf,git
BuildRequires:	ImageMagick-devel,libjpeg-turbo-devel,freeglut-devel,mesa-libGL-devel,glew-devel
BuildRequires:	SDL-devel,SDL_mixer-devel,SDL_ttf-devel
BuildRequires:	qt-x11,qt-devel,libX11-devel
BuildRequires:	portaudio-devel,jack-audio-connection-kit-devel,alsa-lib-devel,libv4l-devel
BuildRequires:	zip,kernel-headers
BuildRequires:	ffmpeg-devel, live555-devel, opencv-devel, openssl-devel
BuildRequires:	libgpujpeg-devel

BuildRequires: 	cairo >= 1.14.0-2
BuildRequires: 	ultragrid-proprietary-drivers
%if %{defined fedora}
%if 0%{?fedora} < 21
BuildRequires:	cuda-core-6-5,cuda-command-line-tools-6-5,cuda-cudart-dev-6-5
%else
BuildRequires:	cuda-core-7-0,cuda-command-line-tools-7-0,cuda-cudart-dev-7-0
%endif
%endif
BuildRequires:	glib2-devel, libcurl-devel

#BuildArch:	x86_64 i686

# dummy, protoze commit
# main package - ultragrid
# old 1.0 release
Requires:	ultragrid-audio-alsa = %{version}-%{release}, ultragrid-audio-jack = %{version}-%{release}
Requires:	ultragrid-audio-portaudio = %{version}-%{release}, ultragrid-card-decklink = %{version}-%{release}
Requires:	ultragrid-card-deltacast = %{version}-%{release}, ultragrid-card-dvs = %{version}-%{release}
#Requires:	ultragrid-card-linsys = %{version}-%{release}, ultragrid-card-screen = %{version}-%{release}
Requires:	ultragrid-compress-rtdxt = %{version}-%{release}
Requires:	ultragrid-core = %{version}-%{release}
#, ultragrid-debuginfo = %{version}-%{release}
Requires:	ultragrid-display-opengl = %{version}-%{release}
Requires:	ultragrid-display-sdl = %{version}-%{release}, ultragrid-gui = %{version}-%{release}
Requires:	ultragrid-plugin-scale = %{version}-%{release}

# new in 1.2 release
Requires:	ultragrid-card-bluefish444 = %{version}-%{release}, ultragrid-card-v4l2 = %{version}-%{release}
Requires:	ultragrid-plugin-swmix = %{version}-%{release}, ultragrid-compress-libavcodec = %{version}-%{release}
Requires:	ultragrid-compress-uyvy = %{version}-%{release}



%description
Ultragrid is a low-latency (As low as 83ms end-to-end)
flexible RTP-based format for HD/2K/4K video transmission tool.
It provides support for various video standards (PAL/NTSC, HD, 4K - both tiled and untiled),
Scalable Adaptive Graphics Environment (SAGE) visualization clusters, several compression
formats (DXT, JPEG, etc.) and many hardware providers.

Our work is supported by CESNET research intent "Optical Network of National
Research and Its New Applications" and partially also by Masaryk University
research intent "Parallel and Distributed Systems". It is a fork of the original
UltraGrid developed by Colin Perkins, Ladan Gharai, et al..

# hack over the way fedora ignores dependences in /usr/lib/dir/*.so
%define _use_internal_dependency_generator 0
%define __find_requires	/bin/bash -c "/usr/lib/rpm/find-requires | sed -e '\
/^libvideomasterhd/d; \
/^libBlueVelvet/d; \
/^libBlueANCUtils64/d; \
/^libsail\.so/d; \
/^libquanta\.so/d; \
'"

#define __find_requires	/bin/bash -c "/usr/lib/rpm/find-requires | sed -e '\
#/^libvideomasterhd/d; \
#/^libcudart\.so/d; \
#/^libsail\.so/d; \
#/^libquanta\.so/d; \
#/^libnpp\.so/d; \
#/^librt\.so/d; \
#'"

%package	full
Requires: 	ultragrid
#Requires:	ultragrid-display-sage3svn = %{version}-%{release}
Requires:	ultragrid-compress-jpeg = %{version}-%{release}
Summary:	Full ultragrid installation
%description	full
Full ultragrid installation containing all drivers including the ones
requiring external libraries outside rpm repositories to operate (currently cuda).
Some might require manual installation using rpm --nodeps.

%package	audio-alsa
Requires:	ultragrid-core = %{version}-%{release}
Summary:	ALSA input/output modules for Ultragrid.
%description	audio-alsa
Advanced Linux Sound Architecture (ALSA) support for Ultragrid.

%package	core
Summary:	The core ultragrid functionality
%description	core
The very core of ultragrid. For individual codecs and cards install respective packages.

%package    openssl
Requires:	ultragrid-core = %{version}-%{release}
Summary:    SSL functionality for Ultragrid.
%description    openssl
SSL functionality for Ultragrid.

#%package	compress-fastdxt
#Requires:	ultragrid-core = %{version}-%{release}
#Summary:	FastDXT compression plugin for Ultragrid.
#%description	compress-fastdxt
#FastDXT DXT compression & decompression plugin for Ultragrid.

%package	plugin-lgdm
Requires:	ultragrid-core = %{version}-%{release}
Summary:	ldgm module for Ultragrid
%description	plugin-lgdm
ldgm module for Ultragrid.

%package 	compress-cudadxt
Requires:	ultragrid-core = %{version}-%{release}
Summary:	DXT compression and decompression on CUDA-capable graphics card
%description	compress-cudadxt
DXT compression and decompression on CUDA-capable graphics card.

%package	compress-jpeg2dxt
Requires:	ultragrid-core = %{version}-%{release}
Summary:	video compresion from jpeg to dxt
%description	compress-jpeg2dxt
Video compresion from jpeg to dxt.

%package	plugin-rtsp
Requires:	ultragrid-core = %{version}-%{release}
Summary:	rtsp plugin for Ultragrid
%description	plugin-rtsp
rtsp plugin for Ultragrid.

%package	 card-aja
Requires:	ultragrid-core = %{version}-%{release}
Summary:	AJA ideo capture adapters support for Ultragrid
%description	card-aja
AJA ideo capture adapters support for Ultragrid.

%package	audio-jack
Requires:	ultragrid-core = %{version}-%{release}
Summary:	JACK input/output modules for Ultragrid
%description	audio-jack
Jack Audio Connection Kit (JACK) support for Ultragrid.

%package	compress-jpeg
Requires:	ultragrid-core = %{version}-%{release}
Summary:	JPEG  compression plugin for Ultragrid. Requires CUDA.
%description	compress-jpeg
GPU JPEG compression & decompression plugin for Ultragrid.
Requires the NVIDIA CUDA repository (see https://developer.nvidia.com/cuda-downloads)

%package	display-opengl
Requires:	ultragrid-core = %{version}-%{release}
Summary:	OpenGL display plugin for Ultragrid.
%description	display-opengl
OpenGL display plugin for Ultragrid.

%package	audio-portaudio
Requires:	ultragrid-core = %{version}-%{release}
Summary:	Portaudio input/output modules for Ultragrid.
%description	audio-portaudio
Portaudio support for ultragrid.

%package	compress-rtdxt
Requires:	ultragrid-core = %{version}-%{release}
Summary:	RTDXT compression plugin for Ultragrid.
%description	compress-rtdxt
RTDXT DXT compression/decompression plugin for Ultragrid.

%package	compress-rxtx-h264
Requires:	ultragrid-core = %{version}-%{release}
Summary:	Video compression plugin rxtx h264 for Ultragrid.
%description	compress-rxtx-h264
Video compression plugin rxtx h264 for Ultragrid.

%package	gui
Requires:	ultragrid-core = %{version}-%{release}
Summary:	Simplistic GUI for Ultragrid.
%description	gui
QT gui for ultragrid.

%package	display-sdl
Requires:	ultragrid-core = %{version}-%{release}
Summary:	SDL display plugin for Ultragrid.
%description	display-sdl
SDL display plugin for ultragrid.

%package	plugin-scale
Requires:	ultragrid-core = %{version}-%{release}
Summary:	Image scaling plugin for Ultragrid.
%description	plugin-scale
OpenGL image scaling plugin for Ultragrid.

%package	plugin-vcapfilter
Requires:	ultragrid-core = %{version}-%{release}
Summary:	Plugin for VCAP filtering functionality for Ultragrid.
%description	plugin-vcapfilter
Plugin for VCAP filtering functionality.

%package	card-dvs
Requires:	ultragrid-core = %{version}-%{release}
Summary:	DVS video capture adapters support for Ultragrid.
%description	card-dvs
Requires the DVS SDK installed (proprietary). Setup symbolic link
/opt/ultragrid-externals/dvs_sdk pointing to your DVS SDK installation.

%package	card-deltacast
Requires:	ultragrid = %{version}-%{release}
Summary:	Deltacast video capture adapters support for Ultragrid.
%description	card-deltacast
Requires the Deltacast drivers installed (VideoMasterHd, proprietary).

%package	card-decklink
Requires:	ultragrid-core = %{version}-%{release}
Summary:	Decklink video capture adapters support for Ultragrid.
%description	card-decklink
Decklink video capture adapters support for Ultragrid.

#%package	card-linsys
#Requires:	ultragrid-core = %{version}-%{release}
#Summary:	Linsys video capture adapters support for Ultragrid.
#%description	card-linsys
#Linsys video capture adapters support for Ultragrid.

#%package	display-sage3svn
#Requires:	ultragrid-core = %{version}-%{release}
#Summary:	SAGE 3.x video output for Ultragrid.
#%description	display-sage3svn
#Scalable Adaptive Graphics Environment (SAGE) 3.x display for Ultragrid.

%package	card-screen
Requires:	ultragrid-core = %{version}-%{release}
Summary:	Screen capture adapter for Ultragrid.
%description	card-screen
OpenGL/X11 screen capture adapter for Ultragrid.

%package	card-bluefish444
Requires:	ultragrid-core = %{version}-%{release}
Summary:	BlueFish444 card input/output driver
%description	card-bluefish444
Requires the BlueFish444 drivers installed (Epoch driver, proprietary).

%package	card-v4l2
Requires:	ultragrid-core = %{version}-%{release}
Summary:	Video4Linux input driver
%description	card-v4l2
Video4Linux v2 input driver

%package	plugin-swmix
Requires:	ultragrid-core = %{version}-%{release}
Summary:	Software videomixer for Ultragrid
%description	plugin-swmix
This plugin allows to grab multiple cards and place the grabbed frames
to arbitrary positions in output video stream

%package	compress-libavcodec
Requires:	ultragrid-core = %{version}-%{release}
Summary:	Libavcodec (ffmpeg) compression/decompression wrapper for Ultragrid.
%description	compress-libavcodec
Allows to acces the ffmpeg enabled copression, expecially h264.

%package	compress-uyvy
Requires:	ultragrid-core = %{version}-%{release}
Summary:	UYVY compression for Ultragrid.
%description	compress-uyvy
UYVY compression for Ultragrid.

%prep
%setup -q
%patch0 -p1
#%patch1 -p1


%build

# need to run autogen as configure.ac is patched during setup
./autogen.sh

# fastdxt requires objects from sage build, but these are not available
# jack-transport is broken since 1.2 release
%configure --docdir=%{_docdir} --enable-rt --disable-profile --disable-debug --enable-ipv6 --enable-plugins \
	--enable-sdl --enable-gl --enable-rtdxt --enable-jpeg \
	--enable-portaudio --disable-jack-transport --enable-jack \
	--enable-alsa --enable-scale --enable-qt --disable-quicktime \
	--disable-coreaudio --enable-dvs --enable-decklink \
	--enable-deltacast --disable-sage --enable-screen\
    --enable-v4l2 --enable-bsd --enable-libavcodec --enable-scale --enable-uyvy \
    --enable-swmix --enable-bluefish444 --enable-blue-audio --enable-ihdtv \
    %if 0%{?fedora} < 21
    --with-cuda=/usr/local/cuda-6.5 \
    %else
    --with-cuda=/usr/local/cuda-7.0 \
    %endif
	--with-dvs=/opt/ultragrid-externals/dvs_sdk \
	--with-bluefish444=/opt/ultragrid-externals/bluefish_sdk --with-deltacast=/opt/ultragrid-externals/deltacast_sdk \
    --with-aja=/opt/ultragrid-externals/aja_sdk
 # --enable-testcard-extras \
#make %{?_smp_mflags}
%install
rm -rf ${RPM_BUILD_ROOT}
make install DESTDIR=${RPM_BUILD_ROOT}
echo %{version}-%{release} > ${RPM_BUILD_ROOT}/%{_datadir}/%{name}/ultragrid.version
echo %{version}-%{release} > ${RPM_BUILD_ROOT}/%{_datadir}/%{name}/ultragrid-full.version

%files
%defattr(-,root,root,-)
%{_datadir}/%{name}/ultragrid.version

%files full
%defattr(-,root,root,-)
%{_datadir}/%{name}/ultragrid-full.version

%files core
%defattr(-,root,root,-)
%{_bindir}/uv
%{_bindir}/hd-rum-transcode
#%{_libdir}/%{name}/ug_lib_common.so.*
%{_libdir}/%{name}/*vidcap_testcard*.so*
#%{_libdir}/%{name}/vidcap_testcard2.so.*
#%{_docdir}/%{name}-%{version}/*
%{_defaultdocdir}/%{name}
%{_datadir}/%{name}

%files openssl
%defattr(-,root,root,-)
%{_libdir}/%{name}/module_openssl.so*

#%files compress-fastdxt
#%defattr(-,root,root,-)
#%{_libdir}/%{name}/*vcompress_fastdxt.so*

%files plugin-lgdm
%defattr(-,root,root,-)
%{_libdir}/%{name}/module_ldgm_gpu.so*

%files compress-cudadxt
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vcompress_cuda_dxt.so*

%files compress-jpeg2dxt
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vdecompress_jpeg_to_dxt.so*

%files plugin-rtsp
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vidcap_rtsp.so*

%files card-aja
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vidcap_aja.so*

%files audio-alsa
%defattr(-,root,root,-)
%{_libdir}/%{name}/*acap_alsa.so*
%{_libdir}/%{name}/*aplay_alsa.so*

%files audio-jack
%defattr(-,root,root,-)
%{_libdir}/%{name}/*acap_jack.so*
%{_libdir}/%{name}/*aplay_jack.so*

%files compress-jpeg
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vcompress_jpeg.so*
%{_libdir}/%{name}/*vdecompress_jpeg.so*

%files display-opengl
%defattr(-,root,root,-)
%{_libdir}/%{name}/*display_gl.so*

%files audio-portaudio
%defattr(-,root,root,-)
%{_libdir}/%{name}/*acap_portaudio.so*
%{_libdir}/%{name}/*aplay_portaudio.so*

%files gui
%defattr(-,root,root,-)
%{_bindir}/uv-qt

%files compress-rtdxt
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vcompress_rtdxt.so*
%{_libdir}/%{name}/*vdecompress_rtdxt.so*

%files compress-rxtx-h264
%defattr(-,root,root,-)
%{_libdir}/%{name}/*video_rxtx_h264.so*

%files plugin-scale
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vo_pp_scale.so*

%files plugin-vcapfilter
%defattr(-,root,root,-)
%{_libdir}/%{name}/module_vcapfilter_resize.so*
%{_libdir}/%{name}/module_vcapfilter_blank.so*

%files display-sdl
%defattr(-,root,root,-)
%{_libdir}/%{name}/*display_sdl.so*

%files card-decklink
%defattr(-,root,root,-)
%{_libdir}/%{name}/*aplay_decklink.so*
%{_libdir}/%{name}/*display_decklink.so*
%{_libdir}/%{name}/*vidcap_decklink.so*

%files card-dvs
%defattr(-,root,root,-)
%{_libdir}/%{name}/*display_dvs.so*
%{_libdir}/%{name}/*vidcap_dvs.so*

#%files card-linsys
#%defattr(-,root,root,-)
#%{_libdir}/%{name}/*vidcap_linsys.so*

#%files display-sage3svn
#%defattr(-,root,root,-)
#%{_libdir}/%{name}/*display_sage.so*

%files card-screen
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vidcap_screen.so*

%files card-deltacast
%defattr(-,root,root,-)
%{_libdir}/%{name}/*display_deltacast.so*
%{_libdir}/%{name}/*vidcap_deltacast.so*

%files card-bluefish444
%defattr(-,root,root,-)
%{_libdir}/%{name}/*display_bluefish444.so*
%{_libdir}/%{name}/*vidcap_bluefish444.so*

%files card-v4l2
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vidcap_v4l2.so*

%files plugin-swmix
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vidcap_swmix.so*

%files compress-libavcodec
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vcompress_libavcodec.so*
%{_libdir}/%{name}/*vdecompress_libavcodec.so*
%{_libdir}/%{name}/*acodec_libavcodec.so*

%files compress-uyvy
%defattr(-,root,root,-)
%{_libdir}/%{name}/*vcompress_uyvy.so*

%changelog
* Wed Apr 2 2014 Ultragrid Development Team <ultragrid-dev@cesnet.cz>
- Fixed dependency issues for BlueFish444 cards

* Wed Oct 23 2013 Ultragrid Development Team <ultragrid-dev@cesnet.cz>
- Ultragrid 1.2 release
- Disabled binary package containing fastdxt
- Added support for BlueFish444 cards, v4l2 and swmix (software videomixer)
- Added support for libavcodec compression

* Fri Jan 6 2012 Ultragrid Development Team <ultragrid-dev@cesnet.cz>
- Ultragrid 1.0 christmas release with bugfixes
- The packet format is now incompatible with the pre-1.0 versions.
