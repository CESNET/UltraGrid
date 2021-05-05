UltraGrid - VR branch
=====================

See also [DEVELOPMENT.md](DEVELOPMENT.md).

UltraGrid as a library
----------------------
Library is compiled automatically to `lib/` as _libug_. Accompanying
header is `libug.h` (in _src/_).

## Samples
* `test_libug_sender.c`

   ### Compile

       cc -Isrc -o test_sender test_libug_sender.c -Llib -lug -pthread

   ### Run

       LD_LIBRARY_PATH=lib ./test_sender [-n] [address]

   **-n** - disables stripping mode

* `test_libug_receiver.c`

   ### Compile

       cc -Isrc -o test_receiver test_libug_receiver.c -Llib -lug

   ### Run

       LD_LIBRARY_PATH=lib ./test_sender [-n] [-d display]

   **-n** - disables stripping mode

For both `-h` displays other options.

## Notes

* <del>Sender binds to 5004 by default, therefore an receiver cannot run at the same machine.</del> -- no longer valid

VR-specific changes
---------------------

- added UltraGrid as a library
  * tests: `test_libug_sender` and `test_libug_receiver`
- added **SHM** capture (currently not used)
- added **VRG** display
- added **CUDA\_I420** and **CUDA\_RGBA** codecs allowing GPUJPEG decompression
  directly to CUDA buffers (need to be allocated by a display)
- increased maximal MTU to 16 KiB
- sending RenderPacket in the video buffer following actual video (data len
  is set to video frame length)
- "stripped" mode - video is stripped to 8 bars sent as one video (eg.
  1920x1080->240x8640) to increase number of parallel sections when encoded by
  libavcodec and decoded with GPUJPEG (equals `video_height / 16`) - for that
  is implemented a capture filter stripe and unstripping mode in GPUJPEG dec.
- changed behavior of UDP ports - sender uses always dynamic RX ports, as a result
  mixed sender/receiver mode in one process is prohibitted

Changes
---------

### 2021-05-05
- `ug_receiver_parameters`/`ug_sender_parameters` - `disable_strips` -> `enable_strips`
  rename (+ default is now disable - not sure if yet compatible with the pitches)

### 2021-05-03
- test sender now processes received RenderPacket and sends it back + option to generate
  YUV 4:2:0 instead of RGBA
- option ":malloc" for VRG display using malloc allocator instead of CUDA-allocated (either
  managed or host)

### 2021-04-23
- UltraGrid now honors `RenderPacket::dx_row_pitch` and `RenderPacket::dx_row_pitch_uv`
  for uncompressed stream (not for JPEG yet)
- `ug_send_frame` now accepts and sends RenderPacket

### 2021-03-22
- VRG now allocates buffers with cudaMallocHost, passed to library is with
  VrgMemory::CPU. Added option to use managed mem instead, aka `-d vrg:managed`
  or to `ug_receiver_parameters::display`.
- maximal MTU size is set to 16 KiB

