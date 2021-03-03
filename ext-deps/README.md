External dependencies
=====================

DeckLink
--------
**DeckLink SDK API** - only a wrapper dynamically calling actual library.

Speex + SpeexDSP
----------------
Essential for UltraGrid (i.e. it won't compile without).

If those can be detected with _pkg-config_, system library is used instead (preferred).
Otherwise `autogen.sh` tries to either pull the submodule, clone or a direct download.

Zfec
----
Currently required. Bootsrapped similarly to Speex/SpeexDSP.
