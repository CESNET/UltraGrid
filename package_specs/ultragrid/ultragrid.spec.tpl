# for embedding cuda
%undefine _missing_build_ids_terminate_build

Name:		ultragrid
Version:	1.6
Release:	20200601.00
Summary:	Software for real-time transmissions of high-definition video
Group:		Applications/Multimedia

License:	GPL
URL:		http://ultragrid.cz
Source0:	ultragrid-%{version}.tar.bz2

BuildRequires:	gcc-c++,make,automake,autoconf,git,libtool
BuildRequires:	ImageMagick-devel,freeglut-devel,glew-devel
BuildRequires:	SDL2-devel,SDL2_mixer-devel,SDL2_ttf-devel
BuildRequires:	libX11-devel
BuildRequires:	portaudio-devel,jack-audio-connection-kit-devel,alsa-lib-devel,libv4l-devel
BuildRequires:	zip,kernel-headers
BuildRequires:	openssl-devel
BuildRequires:	opencv-devel

# bug
#BuildRequires: live555-devel

BuildRequires: 	cairo >= 1.14.0-2
BuildRequires: 	ultragrid-proprietary-drivers-release-1.6
%if %{defined fedora}
BuildRequires:	libjpeg-turbo-devel, mesa-libGL-devel
BuildRequires:	ffmpeg-devel
BuildRequires:	qt5-devel
%else
# suse_version
BuildRequires:	libavcodec-devel, libswscale-devel
BuildRequires:	libjpeg62-devel, Mesa-libGL-devel
BuildRequires:	libqt5-qtbase-devel
%endif
BuildRequires:	glib2-devel, libcurl-devel
BuildRequires:  cppunit-devel

#####################################################
# > cuda
#####################################################
%define cuda 1
#####################################################
# < cuda
#####################################################
#####################################################
# > cineform
#####################################################
%define build_cineform 1
#####################################################
# < cineform
#####################################################
#####################################################
# > ximea
#####################################################
%define build_ximea 1
#####################################################
# < ximea
#####################################################
#####################################################
# > ndi
#####################################################
#define build_ndi 1
#####################################################
# < ndi
#####################################################
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
%define build_dvs 0
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

%if 0%{?cuda} > 0
	%if 0%{?fedora} > 1 && 0%{?fedora} < 21
BuildRequires:	cuda-core-6-5,cuda-command-line-tools-6-5,cuda-cudart-dev-6-5
		%define cudaconf --with-cuda=/usr/local/cuda-6.5
	%endif
	%if 0%{?fedora} >= 21 && 0%{?fedora} <= 25
BuildRequires:	cuda-core-9-1,cuda-command-line-tools-9-1,cuda-cudart-dev-9-1,clang
		%define cudaconf --with-cuda=/usr/local/cuda-9.1 --with-cuda-host-compiler=clang
	%endif
	%if 0%{?fedora} >= 26
BuildRequires:	cuda-core-9-2,cuda-command-line-tools-9-2,cuda-cudart-dev-9-2,clang
		%define cudaconf --with-cuda=/usr/local/cuda-9.2 --with-cuda-host-compiler=clang
	%endif
	%if 0%{?sle_version} >= 150000
BuildRequires:	cuda-core-10-2,cuda-command-line-tools-10-2,cuda-cudart-dev-10-2
		%define cudaconf --with-cuda=/usr/local/cuda-10.2
	%endif
	%if 0%{?leap_version} >= 420000 && 0%{?leap_version} < 430000
BuildRequires:	cuda-core-9-2, cuda-command-line-tools-9-2, cuda-cudart-dev-9-2, clang
		%define cudaconf --with-cuda=/usr/local/cuda-9.2 --with-cuda-host-compiler=clang
	%endif
BuildRequires:	libgpujpeg-devel
%else
	%define cudaconf --disable-cuda
%endif

%if 0%{?build_cineform} > 0
BuildRequires:	libuuid-devel
%endif

%if 0%{?build_ximea} > 0
BuildRequires:	ultragrid-proprietary-drivers-ximea-release-1.6
%endif

%if 0%{?build_ndi} > 0
BuildRequires:	ultragrid-proprietary-drivers-ndi-release-1.6
%endif

%define build_conference 1
%define build_gui 1

%if 0%{build_gui} > 0
	%if 0%{?leap_version} > 1
BuildRequires:  update-desktop-files
Requires(post): update-desktop-files
Requires(postun): update-desktop-files
	%else
BuildRequires:	desktop-file-utils
	%endif
%endif


%define hwaccel 1

%if 0%{?hwaccel} > 0
#BuildRequires: libvdpau-devel, vaapi-intel-driver, libva-devel 
BuildRequires: libva-devel, libvdpau-devel
%if 0%{?fedora} >= 26
%define vaapi 1
%define vdpau 1
%endif
%if 0%{?leap_version} >= 420200 || 0%{?sle_version} >= 120200
%define vaapi 1
%define vdpau 1
BuildRequires: libva-vdpau-driver, vaapi-intel-driver
%endif
%endif


Conflicts:	ultragrid-nightly
Requires:	jack-audio-connection-kit

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
%define __find_requires bash -c 'cd %{_builddir}/%{name}-%{version} ; /usr/lib/rpm/find-requires | (grep -v -F -f install-provides || true) | (grep -v -f norequires || true)'
%define __find_provides bash -c 'cd %{_builddir}/%{name}-%{version} ; /usr/lib/rpm/find-provides | (grep -v -f noprovides || true)'

%define UGLIBDIR %{_libdir}/ultragrid

%prep
%setup -q

%build
%if 0%{?build_cineform} > 0
	pushd cineform-sdk && cmake -DCMAKE_INSTALL_PREFIX=%{_prefix} . && make && popd
%endif

# need to run autogen as configure.ac is patched during setup
./autogen.sh || true

# fastdxt requires objects from sage build, but these are not available
# jack-transport is broken since 1.2 release
# rstp is broken with current live555
%configure --docdir=%_docdir --disable-profile --disable-debug --enable-ipv6 --enable-plugins \
	--enable-sdl2 --enable-gl --enable-gl-display --enable-rtdxt \
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
	%if 0%{?build_ndi} > 0
		--enable-ndi \
	%else
		--disable-ndi \
	%endif
	%if 0%{?build_ximea} > 0
		--enable-ximea \
	%else
		--disable-ximea \
	%endif
	%if 0%{?build_cineform} > 0
		--enable-cineform \
	%else
		--disable-cineform \
	%endif
	%if 0%{?cuda} > 0
		--enable-gpujpeg \
		--enable-ldgm-gpu \
	%else
		--disable-gpujpeg \
		--disable-ldgm-gpu \
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
	%if 0%{?vdpau} > 0
		--enable-lavc-hw-accel-vdpau \
	%else
		--disable-lavc-hw-accel-vdpau \
	%endif
	%if 0%{?vaapi} > 0
		--enable-lavc-hw-accel-vaapi \
	%else
		--disable-lavc-hw-accel-vaapi \
	%endif
	%if 0%{?build_ximea} > 0
		GENICAM_GENTL64_PATH=/usr/src/ultragrid-externals/ximea_sdk/lib \
	%endif
	LDFLAGS="$LDFLAGS -Wl,-rpath=%{UGLIBDIR}" \
# --enable-testcard-extras \

make %{?_smp_mflags}

%check
env UG_SKIP_NET_TESTS=1 make check
env UG_SKIP_NET_TESTS=1 make distcheck

%install
rm -rf ${RPM_BUILD_ROOT}
make install DESTDIR=${RPM_BUILD_ROOT}
mkdir -p ${RPM_BUILD_ROOT}/%{_datadir}/ultragrid
mkdir -p ${RPM_BUILD_ROOT}/%{_docdir}/ultragrid
echo %{version}-%{release} > ${RPM_BUILD_ROOT}/%{_datadir}/ultragrid/ultragrid.version
%if 0%{?cuda} > 0
# copy the real cudart to our rpath
sh -c "$(ldd bin/uv $(find . -name '*.so*') 2>/dev/null | grep cudart | grep -E '^[[:space:]]+' | sed -r "s#[[:space:]]+([^[:space:]]+)[[:space:]]+=>[[:space:]]+([^[:space:]].*)[[:space:]]+[(][^)]+[)]#cp \"\$(realpath '\2')\" '${RPM_BUILD_ROOT}/%{UGLIBDIR}/\1'#g" | uniq | tr $'\n' ';')"
%endif

%if 0%{?build_cineform} > 0
cp cineform-sdk/LICENSE.txt ${RPM_BUILD_ROOT}/%{_docdir}/ultragrid/LICENSE-cineform.txt
%endif
cp speex-*/COPYING ${RPM_BUILD_ROOT}/%{_docdir}/ultragrid/COPYING-speex
cp dxt_compress/LICENSE ${RPM_BUILD_ROOT}/%{_docdir}/ultragrid/LICENSE-dxt_compress
cp dxt_compress/LICENSE-rtdxt ${RPM_BUILD_ROOT}/%{_docdir}/ultragrid/

# dependencies
find ${RPM_BUILD_ROOT}/ -type f | /usr/lib/rpm/find-provides > install-provides
find ${RPM_BUILD_ROOT}/ -type f | /usr/lib/rpm/find-requires > install-requires

echo '^libsail\.so.*$' >> norequires
echo '^libxfixes3-ndi(\.so.*)?$' >> norequires
echo '^libquanta\.so.*$' >> norequires
echo '^libcudart\.so.*$' >> norequires
echo '^libcudart\.so.*$' >> noprovides
rpm -q --provides ultragrid-proprietary-drivers | sed -r -e 's#([()\][.])#\\\1#g' -e 's#^(.*)$#^\1$#g' >> norequires

#grep -v -F -f install-provides install-requires > install-requires-noself || true
#grep -v -f noprovides install-provides > install-provides-result || true
#grep -v -f norequires install-requires-noself > install-requires-result || true

# postinstalls

%post
%if 0%{?fedora} > 0
/usr/bin/update-desktop-database -q %{_datadir}/applications &>/dev/null || :
/usr/bin/gtk-update-icon-cache -qf %{_datadir}/pixmaps &> /dev/null || :
%endif

%postun
%if 0%{?fedora} > 0
/usr/bin/update-desktop-database -q %{_datadir}/applications &>/dev/null || :
/usr/bin/gtk-update-icon-cache -qf %{_datadir}/pixmaps &> /dev/null || :
%endif

%files
%defattr(-,root,root,-)
%dir %{_datadir}/ultragrid
%{_datadir}/ultragrid/*
%{_mandir}/man1/hd-rum-transcode.1.gz
%{_mandir}/man1/uv.1.gz
%if 0%{?build_gui} > 0
%dir %{_datadir}/applications
%{_datadir}/applications/uv-qt.desktop
%dir %{_datadir}/pixmaps
%{_datadir}/pixmaps/*ultragrid*
%endif
%dir %{_docdir}/ultragrid
%{_docdir}/ultragrid/*
%{_bindir}/uv
%{_bindir}/hd-rum-transcode
%if 0%{?build_gui} > 0
%{_bindir}/uv-qt
%endif
%dir %{_libdir}/ultragrid
%if 0%{?build_dvs} > 0
%{_libdir}/ultragrid/ultragrid_vidcap_dvs.so
%{_libdir}/ultragrid/ultragrid_display_dvs.so
%endif
%if 0%{?build_blackmagick} > 0
%{_libdir}/ultragrid/ultragrid_vidcap_decklink.so
%{_libdir}/ultragrid/ultragrid_display_decklink.so
%{_libdir}/ultragrid/ultragrid_aplay_decklink.so
%endif
%if 0%{?build_bluefish} > 0
%{_libdir}/ultragrid/ultragrid_vidcap_bluefish444.so
%{_libdir}/ultragrid/ultragrid_display_bluefish444.so
%endif
%if 0%{?build_aja} > 0
# as of e3e481243
%{_libdir}/ultragrid/ultragrid_aja.so
%endif
%if 0%{?build_deltacast} > 0
%{_libdir}/ultragrid/ultragrid_vidcap_deltacast.so
%{_libdir}/ultragrid/ultragrid_display_deltacast.so
%endif
%if 0%{?build_cineform} > 0
%{_libdir}/ultragrid/ultragrid_vcompress_cineform.so
%{_libdir}/ultragrid/ultragrid_vdecompress_cineform.so
%endif
%if 0%{?build_ximea} > 0
%{_libdir}/ultragrid/ultragrid_vidcap_ximea.so
%endif
%if 0%{?build_ndi} > 0
%{_libdir}/ultragrid/ultragrid_vidcap_ndi.so
%{_libdir}/ultragrid/ultragrid_display_ndi.so
%endif
%{_libdir}/ultragrid/ultragrid_display_sdl*.so
# rtsp is broken with current live555
#{_libdir}/ultragrid/ultragrid_vidcap_rtsp.so
#{_libdir}/ultragrid/ultragrid_video_rxtx_h264.so
%{_libdir}/ultragrid/ultragrid_vcapfilter_resize.so
%{_libdir}/ultragrid/ultragrid_vcapfilter_blank.so
%{_libdir}/ultragrid/ultragrid_vidcap_testcard.so
%{_libdir}/ultragrid/ultragrid_display_gl.so
%{_libdir}/ultragrid/ultragrid_display_preview.so
%{_libdir}/ultragrid/ultragrid_capture_filter_preview.so
%{_libdir}/ultragrid/ultragrid_vidcap_swmix.so
# Fedora 25 OpenCV3
%if 0%{?build_conference} > 0
%{_libdir}/ultragrid/ultragrid_display_video_mix.so
%endif
%{_libdir}/ultragrid/ultragrid_vidcap_screen.so
%{_libdir}/ultragrid/ultragrid_vcompress_rtdxt.so
%{_libdir}/ultragrid/ultragrid_vdecompress_rtdxt.so
%{_libdir}/ultragrid/ultragrid_vcompress_uyvy.so
%{_libdir}/ultragrid/ultragrid_acap_portaudio.so
%{_libdir}/ultragrid/ultragrid_aplay_portaudio.so
%{_libdir}/ultragrid/ultragrid_acap_jack.so
%{_libdir}/ultragrid/ultragrid_aplay_jack.so
%{_libdir}/ultragrid/ultragrid_acap_alsa.so
%{_libdir}/ultragrid/ultragrid_aplay_alsa.so
%{_libdir}/ultragrid/ultragrid_vo_pp_scale.so
%{_libdir}/ultragrid/ultragrid_vo_pp_text.so
%{_libdir}/ultragrid/ultragrid_vidcap_v4l2.so
%{_libdir}/ultragrid/ultragrid_vcompress_libavcodec.so
%{_libdir}/ultragrid/ultragrid_vdecompress_libavcodec.so
%{_libdir}/ultragrid/ultragrid_acompress_libavcodec.so
%{_libdir}/ultragrid/ultragrid_openssl.so
%if 0%{?cuda} > 0
%{_libdir}/ultragrid/ultragrid_vcompress_gpujpeg.so
%{_libdir}/ultragrid/ultragrid_vdecompress_gpujpeg.so
%{_libdir}/ultragrid/ultragrid_vcompress_cuda_dxt.so
%{_libdir}/ultragrid/ultragrid_vdecompress_gpujpeg_to_dxt.so
%{_libdir}/ultragrid/ultragrid_ldgm_gpu.so
# cudart
%{_libdir}/ultragrid/*cudart*
%endif
%if ( 0%{?vaapi} + 0%{?vdpau} ) > 0
%{_libdir}/ultragrid/ultragrid_hw_accel.so
%endif

%changelog
* Mon Jun 01 2020 Ultragrid Development Team <ultragrid-dev@cesnet.cz> 1.6-20200601
- Switching to 1.6 release

* Thu Oct 04 2018 Ultragrid Development Team <ultragrid-dev@cesnet.cz> 1.5-20181004
- Switching to 1.5 release

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
