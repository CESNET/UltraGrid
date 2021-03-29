VR - Development and testing
============================

There is described ongoing development and TODO.

Windows uncompressed performance
--------------------------------

# Ideas
* multithreaded receiving (param "rtp-multithreaded")

  `--param rtp-multithreaded`

  - use multiple receiving threads - **Rationale:** utilize multiple memory channels, eg. on AMD
  - reuse malloc packet buffers, speed up the rest (in `udp_reader()`)

# Measurements

Following values were measured for receiving 9000 B packet on hd3¹

Malloc duration: ~1.8 ± 0.5 µs
Recv duration: ~11 ± 5 µs
Rest of the `udp_reader()`: ~1.5 ± 0.7 µs

¹ Intel I7-4930K, 1866 MHz DDR3
