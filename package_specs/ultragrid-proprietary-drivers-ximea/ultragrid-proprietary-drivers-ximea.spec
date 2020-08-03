Name:		ultragrid-proprietary-drivers-ximea
Version:	20200109
Release:	1%{?dist}
Summary:	Ultragrid drivers ximea pseudometapackage
Group:		Applications/Multimedia

# not really GPL
License: 	Proprietary
URL:		http://www.ximea.com

# replace this line with generated conflicts
Provides:	ultragrid-proprietary-drivers-ximea-nightly
Provides:	ultragrid-proprietary-drivers-ximea-release-1.6

BuildRequires:	gcc-c++, make, automake, autoconf, coreutils
BuildRequires: 	libX11-devel, glew-devel, libXext-devel, glibc, ncurses-devel, qt-devel
BuildRequires:	%kernel_module_package_buildreqs , kernel

Source0:	ximea-linux-upstream.tar.gz
Source1:        ultragrid-proprietary-drivers-ximea-rpmlintrc
Source2:        debian.buildscript.sh

%description
Rewrite ximea installer to be compatible with packages,
essentially create rpm for ultragrid
+ udev rules for plugdev
- pci
- firewire
+ usb
* libusb copied to sdk, but not overriding system defaults


# hack over the way fedora ignores dependences in /usr/lib/dir/*.so
%define _use_internal_dependency_generator 0
%define __find_requires bash -c 'cd %{_builddir}/ ; cat final-requires'
%define __find_provides bash -c 'cd %{_builddir}/ ; cat final-provides'
%define _missing_build_ids_terminate_build 0

%prep
#%setup -q
tar -xvf %{SOURCE0} && cd package

%build
%define debug_package %{nil}
cd package
#env DESTDIR=${RPM_BUILD_ROOT} EUID=0 ./scripts/install_steps -x64 -noudev -nopci

%install
chmod +x %{SOURCE2}
pushd package
env RPM_BUILD_ROOT=${RPM_BUILD_ROOT} SYSCONFDIR=%{_sysconfdir} LIBDIR=%{_libdir} DATADIR=%{_datadir} INCLUDEDIR=%{_includedir} PREFIX=%{_prefix} %{SOURCE2}
popd

# generate dependencies
pushd $RPM_BUILD_ROOT
find . -iregex '.*\.so\(\.[0-9]+\)*$' -o -type f -executable | grep -vE '^\./opt' | /usr/lib/rpm/find-provides > ${RPM_BUILD_DIR}/package/install-provides
find . -iregex '.*\.so\(\.[0-9]+\)*$' -o -type f -executable | /usr/lib/rpm/find-provides > ${RPM_BUILD_DIR}/package/install-provides-full
find . -iregex '.*\.so\(\.[0-9]+\)*$' -o -type f -executable | /usr/lib/rpm/find-requires > ${RPM_BUILD_DIR}/package/install-requires
popd
#grep -vE 'lib.*\.so.*' install-provides > install-provides-nolibs

echo 'libopenh264.*.so.*' >> norequires
echo 'libjasper.*.so.*' >> norequires
echo 'libgst.*.0.10.*.so.*' >> norequires
echo 'libQt5EglDeviceIntegration.so.5.*' >> norequires

grep -v -F -f install-provides-full install-requires > install-requires-noself || true
grep -v -Ef noprovides install-provides > install-provides-result || true
grep -v -Ef norequires install-requires-noself > install-requires-result || true

cp install-requires-result final-requires
cp install-provides-result final-provides 

export NO_BRP_CHECK_RPATH=true

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files
%defattr(-,root,root,-)
%{_sysconfdir}/udev/rules.d/*.rules
%{_sysconfdir}/modprobe.d/*.conf
%{_sysconfdir}/profile.d/*.sh
/opt/XIMEA
%{_libdir}/*
%{_includedir}/*
%{_prefix}/lib*/python*
%{_prefix}/src/ultragrid-externals
%{_prefix}/src/ultragrid-externals/ximea_sdk
%{_datadir}/applications/*.desktop

%changelog
* Thu Feb 27 2020 Lukas Rucka <ultragrid-dev@cesnet.cz>
- 20200227
- Create RPM such that mirrors ximea installed
