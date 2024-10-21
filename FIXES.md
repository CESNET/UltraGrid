#### 1.9.7

* fixed vdisp/file crash
* fixes for irregular (eg. odd) image widths
* added warnings for potentially wrong configs (alsa, R-S audio)
* fixed some color conversion
* NDI fixes - reconnect, 16-bit YCbCr blank if produced in MSW

#### 1.9.6

* fixed not working GPUJPEG decoder
* vcap/rtsp small fixes; but mainly advised to use the development build
because of plenty of changes/fixes that are not to be backported

#### 1.9.5

* CI: updated dependencies (x265, NDI6 et al.)
* libsvtav1 - hint to use crf instead of bitrate (more comprehensive enhancement in the development version)
* rtsp client freeze fix
* fixed capture from OBS virtual cam in Windows
* fixes some (rare) crashes

#### 1.9.4

* new GPUJPEG compat
* build fixes
* miscellaneous fixes, but no high severity

#### 1.9.3

* fixed crashes in edge cases
* support for current DELTACAST driver/SDK
* deinterlace: fixed running without options

#### 1.9.2

* CI: build CUDA stuff for CC 3.5 (compat with Kepler/Maxwell devices)
* build fixes
* DShow: support for H.264 capture
* AV Foundation: fixed running without options
* DeckLink disp.: fixed crash with BMD Studio 4K Studio and '-r analog'
* fixed transparency in GUI on Wayland

#### 1.9.1

* CI build fixes
* fixed libavcodec conversion
* fixed testcard:size=4k (option parse)
