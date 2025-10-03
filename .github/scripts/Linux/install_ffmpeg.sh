#!/bin/bash -eux

cd /var/tmp/ffmpeg
( cd libvpx && sudo make install )
( cd x264 && sudo make install )
( cd nv-codec-headers && sudo make install )
( cd aom/build && sudo cmake --install . )
( cd dav1d/build && sudo ninja install )
sudo cmake --install SVT-AV1/Build
sudo cmake --install SVT-HEVC/Build/linux/Release
sudo cmake --install SVT-VP9/Build
sudo cmake --build oneVPL/build --config Release --target install

sudo make install
sudo ldconfig
