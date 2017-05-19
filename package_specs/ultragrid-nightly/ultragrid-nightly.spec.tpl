Name:		ultragrid-nightly
Version:	1.4
Release:	20170401.00
Summary:	Software for real-time transmissions of high-definition video
Group:		Applications/Multimedia

License:	GPL
URL:		http://ultragrid.cz
Source0:	ultragrid-nightly-%{version}.tar.bz2

BuildRequires:	gcc-c++,make,automake,autoconf,git,libtool
BuildRequires:	ImageMagick-devel,freeglut-devel,glew-devel
BuildRequires:	SDL-devel,SDL_mixer-devel,SDL_ttf-devel
BuildRequires:	qt-x11,qt-devel,libX11-devel
BuildRequires:	portaudio-devel,jack-audio-connection-kit-devel,alsa-lib-devel,libv4l-devel
BuildRequires:	zip,kernel-headers
BuildRequires:	openssl-devel
BuildRequires:	opencv-devel

# bug
#BuildRequires: live555-devel

BuildRequires: 	cairo >= 1.14.0-2
BuildRequires: 	ultragrid-proprietary-drivers
%if %{defined fedora}
BuildRequires:	libjpeg-turbo-devel,mesa-libGL-devel
BuildRequires:	ffmpeg-devel
%else
# suse_version
BuildRequires:	libavcodec-devel, libswscale-devel
BuildRequires:	libjpeg62-devel,Mesa-libGL-devel
%endif
BuildRequires:	glib2-devel, libcurl-devel

#####################################################
# > cuda
#####################################################
%define cuda 1
#####################################################
# < cuda
#####################################################

%if 0%{?cuda} > 0
%if 0%{?fedora} > 1 && 0%{?fedora} < 21
BuildRequires:	cuda-core-6-5,cuda-command-line-tools-6-5,cuda-cudart-dev-6-5
%define cudaconf --with-cuda=/usr/local/cuda-6.5
%else
BuildRequires:	cuda-core-8-0,cuda-command-line-tools-8-0,cuda-cudart-dev-8-0,clang
%define cudaconf --with-cuda=/usr/local/cuda-8.0 --with-cuda-host-compiler=clang
%endif
BuildRequires:	libgpujpeg-devel
%else
%define cudaconf --disable-cuda
%endif

%define build_conference 1
# bug OpenCV3 in Fedora 25 does not contain gpu support stuff
%if 0%{?fedora} > 24
%define build_conference 0
%endif
# nor Leap 42.2
%if 0%{?leap_version} >= 420200 || 0%{?sle_version} >= 120200
%define build_conference 0
%endif

Conflicts:	ultragrid-core, ultragrid
Provides:	ultragrid

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
/^libcudart\.so.*/d; \
'"

%define __find_provides	/bin/bash -c "/usr/lib/rpm/find-provides | sed -e '\
/^libvideomasterhd/d; \
/^libBlueVelvet/d; \
/^libBlueANCUtils64/d; \
/^libsail\.so/d; \
/^libquanta\.so/d; \
/^libcudart\.so.*/d; \
'"

#####################################################
# > bluefish
#####################################################
%define build_bluefish 1
#####################################################
# < bluefish
#####################################################
#####################################################
# > dvs
#####################################################
%define build_dvs 1
#####################################################
# < dvs
#####################################################
#####################################################
# > blackmagick
#####################################################
%define build_blackmagick 1
#####################################################
# < blackmagick
#####################################################
#####################################################
# > deltacast
#####################################################
%define build_deltacast 1
#####################################################
# < deltacast
#####################################################
#####################################################
# > aja
#####################################################
%define build_aja 1
#####################################################
# < aja
#####################################################

%define UGLIBDIR %{_libdir}/ultragrid

%prep
%setup -q

%build
# need to run autogen as configure.ac is patched during setup
./autogen.sh || true

# fastdxt requires objects from sage build, but these are not available
# jack-transport is broken since 1.2 release
# rstp is broken with current live555
%configure --docdir=%_docdir --disable-profile --disable-debug --enable-ipv6 --enable-plugins \
	--enable-sdl --enable-gl --enable-rtdxt \
	--enable-portaudio --disable-jack-transport --enable-jack \
	--enable-alsa --enable-scale --enable-qt --disable-quicktime \
	--disable-coreaudio --disable-sage --enable-screen\
	--enable-v4l2 --enable-gpl-build --enable-libavcodec --enable-scale --enable-uyvy \
	--disable-rtsp \
	--enable-swmix --enable-ihdtv \
	%if 0%{?build_conference} > 0
		--enable-video-mixer \
	%else
		--disable-video-mixer \
	%endif
	%{?cudaconf} \
	%if 0%{?cuda} > 0
		--enable-jpeg \
	%else
		--disable-jpeg \
	%endif
	%if 0%{?build_bluefish} > 0
		--enable-bluefish444 --enable-blue-audio --with-bluefish444=/usr/src/ultragrid-externals/bluefish_sdk  \
	%else
		--disable-bluefish444 --disable-blue-audio \
	%endif
	%if 0%{?build_dvs} > 0
		--enable-dvs --with-dvs=/usr/src/ultragrid-externals/dvs_sdk \
	%else
		--disable-dvs \
	%endif
	%if 0%{?build_blackmagick} > 0
		--enable-decklink \
	%else
		--disable-decklink \
	%endif
	%if 0%{?build_deltacast} > 0
		--enable-deltacast --with-deltacast=/usr/src/ultragrid-externals/deltacast_sdk \
	%else
		--disable-deltacast \
	%endif
	%if 0%{?build_aja} > 0
		--enable-aja --with-aja=/usr/src/ultragrid-externals/aja_sdk \
	%else
		--disable-aja \
	%endif
	LDFLAGS="$LDFLAGS -Wl,-rpath=%{UGLIBDIR}" \
# --enable-testcard-extras \

make %{?_smp_mflags}

%install
rm -rf ${RPM_BUILD_ROOT}
make install DESTDIR=${RPM_BUILD_ROOT}
mkdir -p ${RPM_BUILD_ROOT}/%{_datadir}/ultragrid
echo %{version}-%{release} > ${RPM_BUILD_ROOT}/%{_datadir}/ultragrid/ultragrid-nightly.version
%if 0%{?cuda} > 0
# copy the real cudart to our rpath
sh -c "$(ldd bin/uv $(find . -name '*.so*') 2>/dev/null | grep cudart | grep -E '^[[:space:]]+' | sed -r "s#[[:space:]]+([^[:space:]]+)[[:space:]]+=>[[:space:]]+([^[:space:]].*)[[:space:]]+[(][^)]+[)]#cp \"\$(realpath '\2')\" '${RPM_BUILD_ROOT}/%{UGLIBDIR}/\1'#g" | uniq | tr $'\n' ';')"
%endif

%files
%defattr(-,root,root,-)
%dir %{_datadir}/ultragrid
%{_datadir}/ultragrid/*
%dir %{_docdir}/ultragrid
%{_docdir}/ultragrid/*
%{_bindir}/uv
%{_bindir}/hd-rum-transcode
%{_bindir}/uv-qt
%dir %{_libdir}/ultragrid
%if 0%{?build_dvs} > 0
%{_libdir}/ultragrid/module_vidcap_dvs.so
%{_libdir}/ultragrid/module_display_dvs.so
%endif
%if 0%{?build_blackmagick} > 0
%{_libdir}/ultragrid/module_vidcap_decklink.so
%{_libdir}/ultragrid/module_display_decklink.so
%{_libdir}/ultragrid/module_aplay_decklink.so
%endif
%if 0%{?build_bluefish} > 0
%{_libdir}/ultragrid/module_vidcap_bluefish444.so
%{_libdir}/ultragrid/module_display_bluefish444.so
%endif
%if 0%{?build_aja} > 0
%{_libdir}/ultragrid/module_vidcap_aja.so
%endif
%if 0%{?build_deltacast} > 0
%{_libdir}/ultragrid/module_vidcap_deltacast.so
%{_libdir}/ultragrid/module_display_deltacast.so
%endif
%{_libdir}/ultragrid/module_display_sdl.so
# rtsp is broken with current live555
#%{_libdir}/ultragrid/module_vidcap_rtsp.so
#%{_libdir}/ultragrid/module_video_rxtx_h264.so
%{_libdir}/ultragrid/module_vcapfilter_resize.so
%{_libdir}/ultragrid/module_vcapfilter_blank.so
%{_libdir}/ultragrid/module_vidcap_testcard.so
%{_libdir}/ultragrid/module_display_gl.so
%{_libdir}/ultragrid/module_vidcap_swmix.so
# Fedora 25 OpenCV3
%if 0%{?build_conference} > 0
%{_libdir}/ultragrid/module_display_video_mix.so
%endif
%{_libdir}/ultragrid/module_vidcap_screen.so
%{_libdir}/ultragrid/module_vcompress_rtdxt.so
%{_libdir}/ultragrid/module_vdecompress_rtdxt.so
%{_libdir}/ultragrid/module_vcompress_uyvy.so
%{_libdir}/ultragrid/module_acap_portaudio.so
%{_libdir}/ultragrid/module_aplay_portaudio.so
%{_libdir}/ultragrid/module_acap_jack.so
%{_libdir}/ultragrid/module_aplay_jack.so
%{_libdir}/ultragrid/module_acap_alsa.so
%{_libdir}/ultragrid/module_aplay_alsa.so
%{_libdir}/ultragrid/module_vo_pp_scale.so
%{_libdir}/ultragrid/module_vo_pp_text.so
%{_libdir}/ultragrid/module_vidcap_v4l2.so
%{_libdir}/ultragrid/module_vcompress_libavcodec.so
%{_libdir}/ultragrid/module_vdecompress_libavcodec.so
%{_libdir}/ultragrid/module_acompress_libavcodec.so
%{_libdir}/ultragrid/module_openssl.so
%if 0%{?cuda} > 0
%{_libdir}/ultragrid/module_vcompress_jpeg.so
%{_libdir}/ultragrid/module_vdecompress_jpeg.so
%{_libdir}/ultragrid/module_vcompress_cuda_dxt.so
%{_libdir}/ultragrid/module_vdecompress_jpeg_to_dxt.so
%{_libdir}/ultragrid/module_ldgm_gpu.so
# cudart
%{_libdir}/ultragrid/*cudart*
%endif

%changelog
* Fri Mar 31 2017 Ultragrid Development Team <ultragrid-dev@cesnet.cz> 1.4-20170401
- Switching to 1.4 release

* Thu Feb 2 2017 Ultragrid Development Team <ultragrid-dev@cesnet.cz>
- Integrated package definitions int main git repository

* Thu Sep 17 2015 Ultragrid Development Team <ultragrid-dev@cesnet.cz>
- Fixed the package specification to reflect changes in module naming

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
