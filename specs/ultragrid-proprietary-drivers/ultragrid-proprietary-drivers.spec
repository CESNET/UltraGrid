Name:		ultragrid-proprietary-drivers
Version:	20170103
Release:	1%{?dist}
Summary:	Ultragrid drivers pseudometapackage
Group:		Applications/Multimedia

License: 	GPL
URL:		http://ultragrid.cz

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
Patch1:		videoMasterHD-linux4.6-get-user-pages.patch
#####################################################
# < deltacast (videoMasterHD)
#####################################################
#####################################################
# > bluefish (EpochLinuxDriver)
#####################################################
Patch10:	bluefish-uname.patch
Patch11:	bluefish-g++.patch
Patch12:	bluefish-destdir.patch
Patch13:	bluefish-linux4.6-get-user-pages.patch
#####################################################
# < bluefish (EpochLinuxDriver)
#####################################################
#####################################################
# > aja (ntv2sdklinux)
#####################################################
Patch20:	AJA-linuxdriver-uname.patch
Patch21:	AJA-nodemo.patch
Patch22:	AJA-qmake.patch
Patch23:	AJA-qt5.patch
Patch24:	AJA-gcc-explicit-constructors.patch
Patch25:	AJA-linux4.6-get-user-pages.patch
#####################################################
# < aja (ntv2sdklinux)
#####################################################
#####################################################
# > dvs (sdk)
#####################################################
Patch30:	dvs-linux4.6-get-user-pages.patch
#####################################################
# < dvs (sdk)
#####################################################

%description
Proprietary ultragrid drivers that are 3th party.
Drivers currently managed by this specfile:
EpochLinuxDriver_V5     --bluefish
ntv2sdklinux_12.4.2.1	--aja
sdk4.3.5.21 		--dvs
VideoMasterHD		--deltacast

%prep
%setup -q
#####################################################
# > deltacast
#####################################################
%patch0 -p1
%patch1 -p1
#####################################################
# < deltacast
#####################################################
#####################################################
# > bluefish
#####################################################
%patch10 -p1
%patch11 -p1
%patch12 -p1
%patch13 -p1
#####################################################
# < bluefish
#####################################################
#####################################################
# > aja
#####################################################
%patch20 -p1
%patch21 -p1
%patch22 -p1
%patch23 -p1
%patch24 -p1
%patch25 -p1
#####################################################
# < aja
#####################################################
#####################################################
# > dvs (sdk)
#####################################################
%patch30 -p1
#####################################################
# < dvs (sdk)
#####################################################

%build
%define debug_package %{nil}

#####################################################
# > aja
#####################################################
ls
pushd ntv2sdklinux_*/ntv2projects
env libdir=%{_libdir} make QTDIR=/usr/lib/qt5
popd
#####################################################
# < aja
#####################################################

#####################################################
# > bluefish
#####################################################
pushd EpochLinuxDriver_V5*/drivers/orac
env libdir=%{_libdir} make
popd
pushd EpochLinuxDriver_V5*/apis/BlueVelvet
env libdir=%{_libdir} make
popd
#####################################################
# < bluefish
#####################################################

# relativize all symlinks (build phase)
find ./ -type l -print0 | xargs -0 -I '{}' sh -c 'mv "{}" "{}.pkgbkp" ; ln -s "$(realpath --relative-to="{}.pkgbkp" "$(readlink "{}.pkgbkp")")" "{}" ; rm "{}.pkgbkp"'

%install
mkdir -p $RPM_BUILD_ROOT/usr/src/ultragrid-externals/

#####################################################
# > bluefish
#####################################################
cp -r EpochLinuxDriver_V5* $RPM_BUILD_ROOT/usr/src/ultragrid-externals/
ln -s EpochLinuxDriver_V5* $RPM_BUILD_ROOT/usr/src/ultragrid-externals/bluefish_sdk

pushd EpochLinuxDriver_V5*/drivers/orac
env libdir=%{_libdir} make install DESTDIR=$RPM_BUILD_ROOT
popd
pushd EpochLinuxDriver_V5*/apis/BlueVelvet
env libdir=%{_libdir} make install DESTDIR=$RPM_BUILD_ROOT
popd
#####################################################
# < bluefish
#####################################################
#####################################################
# > dvs
#####################################################
cp -r sdk4.3.* $RPM_BUILD_ROOT/usr/src/ultragrid-externals/
ln -s sdk4.3* $RPM_BUILD_ROOT/usr/src/ultragrid-externals/dvs_sdk

rm -r $RPM_BUILD_ROOT/usr/src/ultragrid-externals/dvs_sdk/linux-x86
#####################################################
# < dvs
#####################################################
#####################################################
# > aja
#####################################################
cp -r ntv2sdklinux_* $RPM_BUILD_ROOT/usr/src/ultragrid-externals/
ln -s ntv2sdklinux_* $RPM_BUILD_ROOT/usr/src/ultragrid-externals/aja_sdk
#####################################################
# < aja
#####################################################
#####################################################
# > deltacast
#####################################################
cp -r VideoMasterHD* $RPM_BUILD_ROOT/usr/src/ultragrid-externals/
ln -s VideoMasterHD* $RPM_BUILD_ROOT/usr/src/ultragrid-externals/deltacast_sdk

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
#####################################################
# < deltacast
#####################################################

for pattern in "*.so" "*.so.*" "*.sh" ; do find ${RPM_BUILD_ROOT}/ -name "$pattern" -exec chmod +x {} \; ; done
for pattern in "*.cpp" "*.h" Makefile "*.bin" "*.pdf" ; do find ${RPM_BUILD_ROOT}/ -name "$pattern" -exec chmod -x {} \; ; done

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%defattr(-,root,root,-)
%{_libdir}/*
%{_prefix}/src/ultragrid-externals

%changelog
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
