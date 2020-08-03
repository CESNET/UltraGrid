Name:		ultragrid-proprietary-drivers-ndi-placeholder
Version:	20200227
Release:	1%{?dist}
Summary:	Ultragrid drivers ndi pseudo package
Group:		Applications/Multimedia

License: 	GPL
URL:		http://www.ultragrid.cz

# replace this line with generated conflicts
Provides:	ultragrid-proprietary-drivers-ndi-nightly
Provides:	ultragrid-proprietary-drivers-ndi-release-1.6

BuildRequires:	make, coreutils

Source0:	dummy.tar.gz

%description
Inject dummy dependency
And create empty .keep file in corresponding location

# hack over the way fedora ignores dependences in /usr/lib/dir/*.so
%define _use_internal_dependency_generator 0
%define _missing_build_ids_terminate_build 0

%prep
#%setup -q
tar -xvf %{SOURCE0}

%build
%define debug_package %{nil}
echo "dummy build"

%install
mkdir -p ${RPM_BUILD_ROOT}/%{_prefix}/src/ultragrid-externals/ndi_sdk_v4_placeholder
touch    ${RPM_BUILD_ROOT}/%{_prefix}/src/ultragrid-externals/ndi_sdk_v4_placeholder/.keep
ln -s ndi_sdk_v4_placeholder ${RPM_BUILD_ROOT}/%{_prefix}/src/ultragrid-externals/ndi_sdk

export NO_BRP_CHECK_RPATH=true

%files
%defattr(-,root,root,-)
%{_prefix}/src/ultragrid-externals
%{_prefix}/src/ultragrid-externals/ndi_sdk_v4_placeholder
%{_prefix}/src/ultragrid-externals/ndi_sdk_v4_placeholder/.keep
%{_prefix}/src/ultragrid-externals/ndi_sdk


%changelog
* Thu Feb 27 2020 Lukas Rucka <ultragrid-dev@cesnet.cz>
- 20200227
- Create RPM such that proprietary driver requirements are satisfied
