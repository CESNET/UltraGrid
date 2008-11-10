This is a combined SDK release for 
- Centaurus II 
- Centaurus II LT

Changes 2.7 -> 3.2
------------------
* The timecode format is the same throughout the DVS SDK, i.e. binary timecodes
  outside the FIFO API are no longer swapped.
* The support of SV_MASTER_RAW/SV_MASTER_CODE has been abandoned. Instead use
  sv_vtrmaster_raw().
* The 64-bit libraries are named dvs<xxx>64.dll in the DVS SDK for Windows.
* Mapped FIFO no longer possible for SDStationOEM / SDStationOEM II.
* Migrated example projects to Microsoft Visual Studio 2005 solution files.
* Combined SDK for 32-bit and 64-bit OS architectures.
* Separate drivers for each hardware.
* Discontinued HDStationOEM support.
* Discontinued Windows NT 4 support.
* Discontinued dvsfile library.
* Discontinued support for linux kernel 2.4.x.
* New version numbering.

New Features
------------
For details about the new features in this SDK please see the file changelog.txt.

Known Bugs / Limitations
------------------------
* Initializing the board in analog genlock mode results in a wrong
  sync position in the analog green output. This can be solved by
  setting the analog genlock mode again.
* The function sv_fifo_wait() returns one frame too soon. For instance, the
  'dpxio' example program stops one field short before the sequence ends.
* Analog LTC (XLR connectors) is not fully reliable in FILM2K rasters.
* The 'Uninstall' button of the DVSconf progam does not work under Windows Vista.
  Instead use the uninstall option of the Windows Vista device manager.
* A system's suspend or standby mode to save power will be supported later.
* Multi-channel: Audio is always assigned to channel 0.
* Multi-channel: Analog out (DVI/CVBS) is always assigned to the output channel 0.
* Closed Caption will be available later.

Known Bugs / Limitations in the Example Programs
------------------------------------------------
* The 'counter' and 'logo' examples do not support 12- and 16-bit modes.
* The 'logo' example program cannot be used in BOTTOM2TOP storage mode.
* The 'overlay' example requires Centaurus II with a firmware version of
  either exactly 3.2.68.11_12_9 or at least 3.2.70.6_12_9
  Centaurus II LT is currently not supported by this example program.
* The 'rs422test' example returns an error message on Centaurus II.
  To circumvent this use it with the 'base' command line option.
* The 'dmaspeed' example requires 8-bit storage modes to measure reasonable
  data rates. Blocksize commands are handled as frame commands.
* Currently there is no exmple for the hardware watchdog feature included.
  It is available on request.

Upgrade Information:
--------------------
The latest firmware revisions for the DVS OEM products can be found on
our OEM web site (http://private.dvs.de/oem). All necessary upgrade
information, such as firmware recommendations, will be detailed there.

This SDK has been tested and released for the following firmware revisions:
  Centaurus II      3.2.70.5_12_9
  Centaurus II LT   4.2.68.11_12_8

