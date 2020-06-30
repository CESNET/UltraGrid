#!/bin/bash -eux

cd /var/tmp/ffmpeg
( cd nv-codec-headers && sudo make install )
( cd aom/build && sudo make install )
sudo make install
sudo ldconfig
