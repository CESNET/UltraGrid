#### 1.8.1

* CI build fixes
* fixed lavc QSV initialization fail
* fixed AVFoundation device order instability
* [Vulkan] fixed 16-bit RGB display
* fixed GUI crash
* double-framerate vo postprocess fixes
* Vulkan is now enabled in Linux builds; macOS builds link with
  MoltenVK, so that Vulkan works there as well

#### 1.8.2

* build fixes
* fixed lavc subsampling option parsing
* fixed inaccurate rendering of some codecs with Vulcan (R10k, UYVY, Y416)
* fixed inability to run portaudio + decklink at the same time in Windows
* fixed PortAudio default devce selection and capture channel count

#### 1.8.3

* build fixes
* fixed libavcodec compression of videos with very small resolution
* [GPUJPEG] fixed parsing of q=/restart= options
* Reed-Solomon - support for multiple tiles
* [dshow] fixed indexing of devices in help
* fixed reflector address overflow + conference participant removal
* fixed DeckLink incorrectly displaying interlaced video

#### 1.8.4

* build fixes
* fixed frame drop during RTP wrap-around
* file video capture - detect interlacing

#### 1.8.5

* fixed audio mixer segfault
* fixed file capture initialized from switcher
* fixed switcher with excl_init
* fixed holepunch

#### 1.8.6

* fixed zfec (Reed-Solomon) crash due to upstream changes
* build fixes
* show HW acceleration checkbox in GUI
