Name:		ultragrid-proprietary-drivers
Version:	20200109
Release:	1%{?dist}
Summary:	Ultragrid drivers pseudometapackage
Group:		Applications/Multimedia

License: 	GPL
URL:		http://ultragrid.cz

# replace this line with generated conflicts
Provides:	ultragrid-proprietary-drivers-nightly
Provides:	ultragrid-proprietary-drivers-release-1.6

BuildRequires:	gcc-c++, make, automake, autoconf, coreutils
BuildRequires: 	libX11-devel, glew-devel, libXext-devel, glibc, ncurses-devel, qt-devel
BuildRequires:	%kernel_module_package_buildreqs , kernel
#####################################################
# > blackmagick (DesktopVideo)
#####################################################
Requires:	desktopvideo
#####################################################
# < blackmagick (Desktopvideo)
#####################################################

%if 0%{?fedora}
BuildRequires: mesa-libGLU-devel, libgcc, qt5-qtbase-devel, mesa-libGL-devel
%else
BuildRequires: freeglut-devel, libqt5-qtbase-devel
%endif

#BuildArch:	x86_64

Source0:	drivers.tar.gz
Source1:	ultragrid-proprietary-drivers-rpmlintrc
#####################################################
# > deltacast (videoMasterHD)
#####################################################
Patch0:		videoMasterHD-destdir.patch
#Patch1:		videoMasterHD-linux4.6-get-user-pages.patch
#Patch9:		videoMasterHD-kernel-backports-opensuse-423.patch
# somehow, provides detection seems to be broken on Fedora
%if 0%{?fedora}
Provides:	libFlxComm64.so()(64bit)
%endif
#####################################################
# < deltacast (videoMasterHD)
#####################################################
#####################################################
# > bluefish (EpochLinuxDriver)
#####################################################
Patch10:	bluefish-uname.patch
#Patch11:	bluefish-g++.patch
Patch12:	bluefish-destdir.patch
#Patch13:	bluefish-linux4.6-get-user-pages.patch
#Patch19:	bluefish-kernel-backports-opensuse-423.patch
#####################################################
# < bluefish (EpochLinuxDriver)
#####################################################
#####################################################
# > aja (ntv2sdklinux)
#####################################################
Patch20:	AJA-linuxdriver-uname.patch
Patch21:	AJA-nodemo.patch
#Patch22:	AJA-qmake.patch
#Patch23:	AJA-qt5.patch
#Patch24:	AJA-gcc-explicit-constructors.patch
#Patch25:	AJA-linux4.16-flush-write-buffers.patch
Patch26:	AJA-clang-cxx14.patch
#Patch29:	AJA-kernel-backports-opensuse-423.patch
#####################################################
# < aja (ntv2sdklinux)
#####################################################

%description
Proprietary ultragrid drivers that are 3th party.
Drivers currently managed by this specfile:
Bluefish_linux_driver_6_0_1_21          --bluefish
ntv2sdklinux_15.5.1.1                   --aja
VideoMasterHD_SDK_Linux_v6.13.0.1       --deltacast

# hack over the way fedora ignores dependences in /usr/lib/dir/*.so
%define _use_internal_dependency_generator 0
%define __find_requires bash -c 'cd %{_builddir}/%{name}-%{version} ; /usr/lib/rpm/find-requires | (grep -v -F -f install-provides || true) | (grep -v -f norequires || true)'
%define __find_provides bash -c 'cd %{_builddir}/%{name}-%{version} ; /usr/lib/rpm/find-provides | (grep -v -f noprovides || true)'

%prep
%setup -q
#####################################################
# > deltacast
#####################################################
%patch0 -p1
#%patch1 -p1
#%if 0%{?is_opensuse} >= 1 && 0%{?sle_version} == 120300
#%patch9 -p1
#%endif
#####################################################
# < deltacast
#####################################################
#####################################################
# > bluefish
#####################################################
%patch10 -p1
#%patch11 -p1
%patch12 -p1
#%patch13 -p1
#%if 0%{?is_opensuse} >= 1 && 0%{?sle_version} == 120300
#%patch19 -p1
#%endif
#####################################################
# < bluefish
#####################################################
#####################################################
# > aja
#####################################################
%patch20 -p1
%patch21 -p1
#%patch22 -p1
#%patch23 -p1
#%patch24 -p1
%patch26 -p1
#%if 0%{?is_opensuse} >= 1 && 0%{?sle_version} == 120300
#%patch29 -p1
#%endif
#on intention
#%patch25 -p1
#####################################################
# < aja
#####################################################

%build
%define debug_package %{nil}
find . -type f -iname "*.pri" -exec chmod -x {} \;
find . -type f -name "Makefile" -exec chmod -x {} \;
find . -type f -name "*~" -exec rm {} \;

#####################################################
# > aja
#####################################################
pushd ntv2sdk*
env libdir=%{_libdir} make QTDIR=/usr/lib/qt5 AJA_NO_FLTK=1
popd
#####################################################
# < aja
#####################################################

#####################################################
# > bluefish
#####################################################
pushd Bluefish_linux_driver_6*/apis/BlueVelvetC/lin
env libdir=%{_libdir} make
popd

rm -rf Bluefish_linux_driver_6*/firmware/x86
#####################################################
# < bluefish
#####################################################

# relativize all symlinks (build phase)
find ./ -type l -print0 | xargs -0 -I '{}' sh -c 'mv "{}" "{}.pkgbkp" ; pushd "$(dirname "{}")" > /dev/null ; echo ln -s "$(realpath --relative-to=. "$(readlink "$(basename "{}").pkgbkp")")" "$(basename "{}")" ; ln -s "$(realpath --relative-to=. "$(readlink "$(basename "{}").pkgbkp")")" "$(basename "{}")" ; popd > /dev/null ; rm "{}.pkgbkp"'

%install
mkdir -p $RPM_BUILD_ROOT/usr/src/ultragrid-externals/

#####################################################
# > bluefish
#####################################################

# classical make install must be done first
pushd Bluefish_linux_driver_6*/apis/BlueVelvetC/lin
env libdir=%{_libdir} make install DESTDIR=$RPM_BUILD_ROOT
popd

ln -s Bluefish_linux_driver_6* bluefish_sdk
tar -c bluefish_sdk Bluefish_linux_driver_6* -f - | tar -C $RPM_BUILD_ROOT/usr/src/ultragrid-externals/ -xf -

#####################################################
# < bluefish
#####################################################
#####################################################
# > aja
#####################################################
ln -s ntv2sdklinux_* aja_sdk
tar -c aja_sdk ntv2sdklinux_* -f - | tar -C $RPM_BUILD_ROOT/usr/src/ultragrid-externals/ -xf -
#####################################################
# < aja
#####################################################
#####################################################
# > deltacast
#####################################################
ln -s VideoMasterHD* deltacast_sdk
tar -c deltacast_sdk VideoMasterHD* -f - | tar -C $RPM_BUILD_ROOT/usr/src/ultragrid-externals/ -xf -

mkdir -p $RPM_BUILD_ROOT%{_libdir}
pushd VideoMasterHD_*/Library/
for i in $(ls -d */x64/)
do
    pushd $i
    env libdir=%{_libdir} %makeinstall
    popd
done
popd

# fix bad arch binaries
find ${RPM_BUILD_ROOT}/usr/src/ultragrid-externals/deltacast_sdk/Library/ -maxdepth 2 -name x86 -exec rm -r {} \;
find ${RPM_BUILD_ROOT}/ -executable -type f -exec file {} \; | grep -i 'elf 32' | cut -d':' -f1 | while read -r -d $'\n' filename ; do rm "$filename" ; done
find ${RPM_BUILD_ROOT}/ -iregex '.*\.so\(\.[0-9]+\)*$' -type f -exec file {} \; | grep -i 'elf 32' | cut -d':' -f1 | while read -r -d $'\n' filename ; do rm "$filename" ; done
#####################################################
# < deltacast
#####################################################

for pattern in "*.so" "*.so.*" "*.sh" ; do find ${RPM_BUILD_ROOT}/ -name "$pattern" -exec chmod +x {} \; ; done
for pattern in "*.c" "*.cu" "*.cpp" "*.h" Makefile "*.bin" "*.pdf" ; do find ${RPM_BUILD_ROOT}/ -name "$pattern" -exec chmod -x {} \; ; done
# new AJA installs broken symlinks
#rm -rf ${RPM_BUILD_ROOT}/usr/src/ultragrid-externals/ntv*/bin/qtlibs
# remove messed diffs if patches otherwise patched successfuly

echo '^libQt.*$' >> noprovides
echo '^libQt.*$' >> norequires


find ${RPM_BUILD_ROOT}/ -name "*.orig" -exec rm {} \;

find ${RPM_BUILD_ROOT}/ -iregex '.*\.so\(\.[0-9]+\)*$' -o -type f -executable | /usr/lib/rpm/find-provides > install-provides
find ${RPM_BUILD_ROOT}/ -iregex '.*\.so\(\.[0-9]+\)*$' -o -type f -executable | /usr/lib/rpm/find-requires > install-requires

#grep -v -F -f install-provides install-requires > install-requires-noself || true
#grep -v -f noprovides install-provides > install-provides-result || true
#grep -v -f norequires install-requires-noself > install-requires-result || true

export NO_BRP_CHECK_RPATH=true

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%defattr(-,root,root,-)
%{_libdir}/*
%{_prefix}/src/ultragrid-externals

%changelog
* Thu Jan 16 2020 Lukas Rucka <ultragrid-dev@cesnet.cz>
- 20200109
- Upgrade all drivers, except dvs, which is dropped

* Fri Aug 3 2018 Lukas Rucka <ultragrid-dev@cesnet.cz>
- 20180803
- Upgrade Aja drivers to compensate for kernel api changes

* Sat May 5 2018 Lukas Rucka <ultragrid-dev@cesnet.cz>
- 20180511
- Upgrade Bluefish drivers to compensate for kernel api changes

* Tue Jan 3 2017 Lukas Rucka <xrucka@fi.muni.cz>
- 20170103
- Marked up sections to enable public specification release available

* Fri Oct 28 2016 Lukas Rucka <xrucka@fi.muni.cz>
- 20161022
- Upgraded all drivers

* Fri Apr 29 2016 Lukas Rucka <xrucka@fi.muni.cz>
- Upgraded VideoMasterHD & ntv2

* Tue Mar 17 2015 Matej Minarik <xminari4@mail.muni.cz>
- Creating this package
