#### 1.10.3

* screen_avf: fix swapped name/dev in probe (for GUI)
* compliance with Apple Dev Program (explicit recording notice
if screen capture or Core Audio device is used)
* reenable CFHD decoding fallback with lavd
* fixed some small lavd regression when some decoders unavail

#### 1.10.2

* fix Vulkan display crashing on keypresses in console
* fix GL display show/hide cursor keybind
* fix control port argument in hd-rum-translator (GH-482)
* added 2 conversions for Videotoolbox 10b encode (GH-479, GH-480)
* modified hevc_videotoolbox not to default to 4:2:0 (GH-480)
* bundle libdecor so that the windows (GL/SDL/Vulkan) have decorations
  in Wayland
* make SPOUT running again (GH-487)
* fix VDPAU being advertised/used with GL even when undesirable
* fix GPUJPEG subsampling detection (probe, by updated GJ upstream)
* ported AV Foundation vcap from master so that it can be used as
  a screen capture for macOS Intel builds (the old Core Graphics cannot
  be built anymore) (GH-485)

#### 1.10.1

* fix temporarily broken r12l_to_gbrp16le (GH-476)
* missing arm64 macOS dependency on brotlicommon (GH-478)
* add libOpenGL.0.so library fallback to Linux AppImage
* fix Pipewire screen capture not starting
* build fixes
* reenable AV Foundation (not being enabled on macOS)
* fix decoding ProRes (GH-481)
