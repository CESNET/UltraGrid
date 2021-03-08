External dependencies
=====================

DeckLink
--------
**DeckLink SDK API** - only a wrapper dynamically calling actual library.

SpeexDSP
----------------
Essential for UltraGrid (i.e. it won't compile without).

If this can be detected with _pkg-config_, system library is used instead (preferred).
Otherwise `autogen.sh` tries to either pull the submodule, clone or a direct download.

Zfec
----
Currently required. Bootstrapped similarly to SpeexDSP.
