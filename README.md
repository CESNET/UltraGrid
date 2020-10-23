UltraGrid - VR branch
=====================

UltraGrid as a library
----------------------
Library is compiled automatically to `lib/` as _libug_. Accompanying
header is `libug.h` (in _src/_).

## Samples
* `test_libug_sender.c`

   ### Compile

       cc -o test_sender test_libug_sender.c -lug

   ### Run

       test_sender [address]

## Notes

* Sender binds to 5004 by default, therefore an receiver cannot run at the same machine.

