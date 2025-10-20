#### 1.9.12

* AppImage comatibility improvement (fix missing gnutls, maybe also gdk[_pixbuf])
* vcap/file: fixes (raw file rewinding + others)
* vulkan_sdl2: fix (allow) integrated/discrete card selection
* vcap/decklink: fix color-space change detection (regression)
* CI build fixes

#### 1.9.11

* improve NVENC intra-refresh behavior (ported from master)
* suppress `--cuda-devices help` returned error
* fix multi-tile compress support

#### 1.9.10

* fix macOS >= 15.4 init crashes (LC_RPATH dups)
* build fixes (XIMEA, PCP)
* fix manual port assignment with send+recv over lo in single process
* few other less important fixes

#### 1.9.9

* build fixes: switch to U22, PCP
* fix SDL2 I420
* improve Firejail AppImage compatibility with various params
* fix WASAPI device selection with UUID
* CoInitialize: use COINIT_APARTMENTTHREADE (compat with PA/ASIO build)
* select libvpx-vp9 for VP9 if AVX2 not present (avoid crasn with SVT impl)
* portaudio: improved sample rate handling
* Vulkan fixes

#### 1.9.8

* build fixes (SVT-VP9, Vulkan)
* re-enable DELTACAST

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
