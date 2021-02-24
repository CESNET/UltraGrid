# Reporting bugs

There are couple of ways to report a problem:

  - primary channel is an e-mail conference ultragrid-dev@cesnet.cz
    which leads to developers of UltraGrid
  - you can use [GitHub issues](https://github.com/CESNET/UltraGrid/issues)
    for reporting bugs. You need to register, but it is only matter
    of filling username and password.

If the problem is a crash (segmentation fault, abort), if possible, attach
a core dump (if generated) and the binary (if you compiled by yourself, otherwise
the executable version). If core dump is not generated, a backtrace might have been
generated to standard error output so please attach this. Also the terminal output
containg the error context would be helpful.

If reporting a bug, please use the latest version of UltraGrid (release/nightly) if not
already using that - either a stable release (a release/X.Y branch) or master for source,
in case of a binary package its most recent version. If not using already, please retest
that version and report the bugs against the current one.

