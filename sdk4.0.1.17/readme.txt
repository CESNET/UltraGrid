Release notes for version 4.0.x.x

This is a combined beta SDK release for
- Atomix
- Atomix LT
- Centaurus II 
- Centaurus II LT

It is available for
- Windows 2000
- Windows XP
- Windows Vista
- Windows 7
- Linux Kernel 2.6
- Mac OS X

New Features
------------
* Support for Atomix.
* Render API for Atomix.

Changes 3.x -> 4.0
------------------
* sv_fifo_status() will now return the exact ring buffer size, see documentation.
* Removed support for SDStationOEM and SDStationOEM II.

Known Bugs / Limitations
------------------------
* Missing installation guide for Atomix.
* In new syncmode introduced in SDK 3.2.14.0 the output signal is shifted by one output line
  in case of syncing SMPTE296 with a SD signal (Centaurus II / Centaurus II LT).
* Multi-channel: Analog out (DVI/CVBS) is always assigned to the output channel 0 (Centaurus II / Centaurus II LT).
* Closed Caption will be available later.
* Rasters PAL24I, PALHR, NTSCHR, SMPTE240/29I and 30I, SMPTE295/25I, SMPTE296/71P and 72P are no longer supported.

Known Bugs / Limitations in the Example Programs
------------------------------------------------
* The 'dpxrender' example is not usable with Centaurus II and Centaurus II LT.
* Using dpxio example for field-based output is not implemented.
* The 'counter' and 'logo' examples do not support 12- and 16-bit modes.
* The 'logo' example program cannot be used in BOTTOM2TOP storage mode.
* The 'overlay' example requires Centaurus II with a firmware version of
  either exactly 3.2.68.11_12_9 or at least 3.2.70.6_12_9
  Centaurus II LT is currently not supported by this example program.
* The 'rs422test' example returns an error message on Centaurus II.
  To circumvent this use it with the 'base' command line option.
* The 'dmaspeed' example requires 8-bit storage modes to measure reasonable
  data rates. Blocksize commands are handled as frame commands.
* Currently there is no example for the hardware watchdog feature included.
  It is available on request.

Upgrade Information:
--------------------
The latest firmware revisions for the DVS video boards can be found on
our web site (http://www.dvs.de). All necessary upgrade
information, such as firmware recommendations, will be detailed there.

This SDK has been tested for the following firmware revisions:
  Atomix (DIGOUT module)  3.3.0.22_3.0.10
  Atomix (onboard MCX)    4.3.0.15_3.0.10
  Atomix LT               5.3.4.14_5.0.2
  Centaurus II            3.2.76.5_19_15
  Centaurus II LT         4.2.68.13_18_14
