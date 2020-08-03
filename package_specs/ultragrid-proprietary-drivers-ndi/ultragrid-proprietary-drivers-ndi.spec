Name:		ultragrid-proprietary-drivers-ndi
Version:	20200227
Release:	1%{?dist}
Summary:	Ultragrid drivers ndi pseudometapackage
Group:		Applications/Multimedia

License: 	GPL
URL:		http://www.ndi.com

# replace this line with generated conflicts
Provides:	ultragrid-proprietary-drivers-ndi-nightly
Provides:	ultragrid-proprietary-drivers-ndi-release-1.6

BuildRequires:	gcc-c++, make, automake, autoconf, coreutils
BuildRequires: 	libX11-devel, glew-devel, libXext-devel, glibc, ncurses-devel, qt-devel
BuildRequires:	%kernel_module_package_buildreqs , kernel

Source0:	ndi4.tar.gz
Source1:	ultragrid-proprietary-drivers-ndi-rpmlintrc

%description
Unpack NDI and remove irrelevant architectures
+ instal systematic symlink

# hack over the way fedora ignores dependences in /usr/lib/dir/*.so
%define _use_internal_dependency_generator 0
%define _missing_build_ids_terminate_build 0

%prep
#%setup -q
tar -xvf %{SOURCE0}

%build
%define debug_package %{nil}
yes | env PAGER=cat ./InstallNDISDK_v4_Linux.sh
cd 'NDI SDK for Linux'
rm -rf $(find lib bin -mindepth 1 -maxdepth 1 -type d | grep -v x86.64)

%install
mkdir -p ${RPM_BUILD_ROOT}/%{_prefix}/src/ultragrid-externals/
mv 'NDI SDK for Linux'  ${RPM_BUILD_ROOT}/%{_prefix}/src/ultragrid-externals/ndi_sdk_v4
ln -s ndi_sdk_v4 ${RPM_BUILD_ROOT}/%{_prefix}/src/ultragrid-externals/ndi_sdk

# now, link to common paths
mkdir -p ${RPM_BUILD_ROOT}/%{_libdir} ${RPM_BUILD_ROOT}/%{_includedir}
pushd ${RPM_BUILD_ROOT}/
find usr/src/ultragrid-externals/ndi_sdk/lib -iname "*.so" -exec ln -s /{} ./%{_libdir}/ \;
find usr/src/ultragrid-externals/ndi_sdk/lib -iname "*.so.*" -exec ln -s /{} ./%{_libdir}/ \;
find usr/src/ultragrid-externals/ndi_sdk/include -iname "*.h" -exec ln -s /{} ./%{_includedir}/ \;
popd

export NO_BRP_CHECK_RPATH=true

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files
%defattr(-,root,root,-)
%{_prefix}/src/ultragrid-externals
%{_prefix}/src/ultragrid-externals/ndi_sdk_v4
%{_prefix}/src/ultragrid-externals/ndi_sdk
%{_libdir}/*
%{_includedir}/*


%changelog
* Thu Feb 27 2020 Lukas Rucka <ultragrid-dev@cesnet.cz>
- 20200227
- Create RPM such that mirrors ndi installed
