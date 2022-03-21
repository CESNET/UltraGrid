#### 1.7.1
* fixed testcard pattern gradient2
* build fixes (OpenGL, SDP/HTTP, SVT VP9)
* broken JPEG from libavcodec if threads disabled (threads=no)
* AppImage: bundle i965-va-driver-shaders instead of i965-va-driver
* fixed AppImage GUI crash on Ubuntu 21.10 + compat with Rocky Linux 8.5

#### 1.7.2
* fixed potential problem with multichannel audio when there is a BPS change
* build GitHub builds with NDI 5 by default (user still needs to obtain NDI
  library by itself)

#### 1.7.3
* fixed channel mapping from lesser amount of input channels to higher (eg.
  `--audio-channel-map 0:1` while receiving only mono)
* do not set BMD conversion mode (issue #215)
