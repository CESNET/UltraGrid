#!/bin/bash

optpath=${RPM_BUILD_ROOT}/opt/XIMEA
platform_bits=64
api_version=2

mkdir -p ${RPM_BUILD_ROOT}/${SYSCONFDIR}/udev/rules.d/
cp libs/libusb/*.rules ${RPM_BUILD_ROOT}/${SYSCONFDIR}/udev/rules.d/

mkdir -p $optpath/bin $optpath/lib $optpath/src $optpath/CamTool $optpath/backup 2>/dev/null
mkdir -p "$optpath/data"
cp README* uninstall version_LINUX_SP.txt "$optpath"
touch $optpath/backup/files.txt

# libusb copy
cp -r libs/libusb/* $optpath/lib/


mkdir -p ${RPM_BUILD_ROOT}/${LIBDIR}

# install api
cp api/X$platform_bits/libm3api.so.$api_version ${RPM_BUILD_ROOT}/${LIBDIR}/libm3api.so.${api_version}.0.0 && \
ln -snf libm3api.so.$api_version ${RPM_BUILD_ROOT}/${LIBDIR}/libm3api.so

cp libs/gentl/X$platform_bits/libXIMEA_GenTL.cti $optpath/lib/
cp libs/gentl/X$platform_bits/libxiFapi_GenTL.cti $optpath/lib/

[ -e libs/xiapi_dng_store/X$platform_bits ] && cp libs/xiapi_dng_store/X$platform_bits/libxiapi_dng_store.so $optpath/lib/ || true

cp bin/xiSample.$platform_bits $optpath/bin/xiSample
cp "bin/xiCOP.$platform_bits" "$optpath/bin/xiCOP"
cp "data/fw_update_tools_map.xml" "$optpath/data/"

cp bin/streamViewer.$platform_bits $optpath/bin/streamViewer

mkdir -p ${RPM_BUILD_ROOT}/${DATADIR}/applications

newlauncher=${RPM_BUILD_ROOT}/${DATADIR}/applications/streamViewer.desktop
echo "[Desktop Entry]" >$newlauncher
echo "Version=1.0" >>$newlauncher
echo "Type=Application" >>$newlauncher
echo "Terminal=false" >>$newlauncher
echo "Icon[en_US]=gnome-panel-launcher" >>$newlauncher
echo "Name=streamViewer" >>$newlauncher
echo "Exec=$optpath/bin/streamViewer" >>$newlauncher
echo "Categories=AudioVideo;Player;Recorder;AudioVideo;Video;" >> $newlauncher
echo "GenericName=Stream Viewer Tool" >> $newlauncher
chmod a+x $newlauncher

echo Installing CamTool
libh264=libopenh264-1.4.0-linux"$platform_bits".so
# Remove files from previous versions since they can cause CamTool crashes
find "$optpath"/CamTool -mindepth 1 \! -name "$libh264" -delete
cp bin/xiCamTool $optpath/bin/
cp -R CamTool.$platform_bits/* $optpath/CamTool

# should be handled by download service
# if [! -e "$optpath"/CamTool/"$libh264" ] ; then
#wget -O "$optpath"/CamTool/"$libh264".bz2 http://ciscobinary.openh264.org/"$libh264".bz2
#bunzip2 "$optpath"/CamTool/"$libh264".bz2
#fi 

newlauncher=${RPM_BUILD_ROOT}/${DATADIR}/applications/xiCamTool.desktop
echo "[Desktop Entry]" >$newlauncher
echo "Version=1.0" >>$newlauncher
echo "Type=Application" >>$newlauncher
echo "Terminal=false" >>$newlauncher
echo "Icon=/opt/XIMEA/CamTool/icon.png" >>$newlauncher
echo "Name=xiCamTool" >>$newlauncher
echo "Exec=$optpath/bin/xiCamTool" >>$newlauncher
echo "Categories=AudioVideo;Player;Recorder;AudioVideo;Video;" >> $newlauncher
echo "GenericName=Ximea Camera Viewer Tool" >> $newlauncher
chmod a+x $newlauncher

mkdir -p ${RPM_BUILD_ROOT}/${SYSCONFDIR}/modprobe.d/
echo install usbcore usbfs_memory_mb=0 > ${RPM_BUILD_ROOT}/${SYSCONFDIR}/modprobe.d/99-ximea-usbcore.conf

mkdir -p ${RPM_BUILD_ROOT}/${SYSCONFDIR}/profile.d
echo "export GENICAM_GENTL${platform_bits}_PATH=/opt/XIMEA/lib/" >> ${RPM_BUILD_ROOT}/${SYSCONFDIR}/profile.d/99-ximea-genicam.sh

# sample code
cp -R include $optpath/
cp -R examples $optpath/
cp -R samples $optpath/

# include dir
mkdir -p ${RPM_BUILD_ROOT}/${INCLUDEDIR}
ln -snf /opt/XIMEA/include ${RPM_BUILD_ROOT}/${INCLUDEDIR}/m3api


for d in api/Python/v*/ximea/libs/ ; do
	pushd $d
	rm -rf $(find .  -mindepth 1 -maxdepth 1 -type d | grep -iv X64)
	popd
done

# python wrappers - copypasted from the install script, except for ${RPM_BUILD_ROOT}
for py in python python2 python3
do
        if type $py &>/dev/null
        then
                py_ver="$($py -c "import sys; pyver=sys.version_info[0]; print(pyver)")"
                py_site_dir="$($py -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")"
                if [ -n "$py_site_dir" ]
                then
                        mkdir -p "${RPM_BUILD_ROOT}/$py_site_dir"/ximea
                        if [ "$py_ver" -gt 2 ]
                        then
                                echo "Installing python package version 3.x"
                                cp -R api/Python/v3/ximea "${RPM_BUILD_ROOT}/$py_site_dir"
                        else
                                echo "Installing python package version 2.x"
                                cp -R api/Python/v2/ximea "${RPM_BUILD_ROOT}/$py_site_dir"
                        fi
                fi
        fi
done

mkdir -p ${RPM_BUILD_ROOT}/${PREFIX}/src/ultragrid-externals/
ln -s /opt/XIMEA ${RPM_BUILD_ROOT}/${PREFIX}/src/ultragrid-externals/ximea_sdk

pushd ${RPM_BUILD_ROOT}/opt/XIMEA/lib
rm -rf $(find .  -mindepth 1 -maxdepth 1 -type d | grep -v X64)
popd

