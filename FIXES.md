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

