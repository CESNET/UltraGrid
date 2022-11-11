# Reporting bugs

There are couple of ways to report a problem:

  - you can use [GitHub issues](https://github.com/CESNET/UltraGrid/issues)
    for reporting bugs. You need to register, but it is only matter
    of filling username and password.
  - alternativey you can use our an e-mail conference ultragrid-dev _\_at\__
    cesnet.cz which leads to developers of UltraGrid
  - if not sure if the problem is a bug, you can also use
    [GitHub discussions](https://github.com/CESNET/UltraGrid/discussions)
    to ask a question.

Please try to follow these _rules_ when reporting a bug if possible:

1. test the issue with latest version of UltraGrid (either release or
   continuous channel)
2. if relevant, attach information about your environment (SW/HW)
3. report a _minimal working example_ of the problem -- this mainly means
   removing excess parameters from UltraGrid command-line
4. if the problem is specific to some device, try to use either _testcard_ to
   generate the signal that isn't displayed correctly or _GL_/_SDL_ display to
   display the broken signal, eg.:

       uv -t testcard:codec=RGB -d ndi # problems with NDI display
       uv -t decklink[:args] -d gl     # problems with DeckLink capture (omit `-d gl` if the problem is not visual but eg. a crash)
       uv -t testcard -c libavcodec:encoder=libx264 -d gl # problems with compression
       uv -t decklink[:args] -d decklink[:args]  # WRONG: not sure what the signal actually is and if probiem is in capture or display

If you suspect that the issue may not be always replicable, you can use a
script
[ultragrid-bugreport-collect.sh](../data/ultragrid-bugreport-collect.sh)
to collect data about a computer and attach its result (Linux/macOS).

If the problem is a **crash** (segmentation fault, abort), if possible, attach
a _core dump_ (if generated) and the _binary_ (if you compiled by yourself,
otherwise the _executable version_). If core dump is not generated, a
_backtrace_ (see below) might have been generated to standard error output so
please attach this. Also the terminal output containg the _error context_ would
be helpful.

## Decoding stacktrace (Linux)
If reporting a bug, you can also decode the stacktrace for the bug report.
The following is intended mainly for _advanced users_ with _UltraGrid_ compiled from source,
don't bother with it not sure. Requires **addr2line** and **objdump**.

Eg. we have a stacktrace saved as _stacktrace.txt_:
```
Backtrace:
./bin/uv(+0x21b946)[0x562ebedee946]
/lib/x86_64-linux-gnu/libc.so.6(+0x41950)[0x7f7d9dfe4950]
/lib/x86_64-linux-gnu/libc.so.6(gsignal+0xcb)[0x7f7d9dfe48cb]
/lib/x86_64-linux-gnu/libc.so.6(abort+0x116)[0x7f7d9dfc9864]
./bin/uv(main+0x2b93)[0x562ebedf7141]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf2)[0x7f7d9dfcbcb2]
./bin/uv(_start+0x2e)[0x562ebecd7b9e]
```

In _UltraGrid_ sources there is a script `tools/stacktrace_addr2line.sh`.
You can use that to decode the stacktrace addresses to human-readable symbols:

```
$ ./tools/stacktrace_addr2line.sh < stacktrace.txt
Decoding ../build-devel-linux/bin/uv(+0x21b946)[0x563cded2e946]
crash_signal_handler(int)
/home/martin/Projects/ultragrid/build-devel-linux/../src/main.cpp:301
Decoding /lib/x86_64-linux-gnu/libc.so.6(+0x41950)[0x7fc826d85950]
killpg
??:?
Decoding /lib/x86_64-linux-gnu/libc.so.6(gsignal+0xcb)[0x7fc826d858cb]
??
??:0
Decoding /lib/x86_64-linux-gnu/libc.so.6(abort+0x116)[0x7fc826d6a864]
??
??:0
Decoding ../build-devel-linux/bin/uv(main+0x2b93)[0x563cded37141]
main
/home/martin/Projects/ultragrid/build-devel-linux/../src/main.cpp:1313
Decoding /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf2)[0x7fc826d6ccb2]
??
??:0
Decoding ../build-devel-linux/bin/uv(_start+0x2e)[0x563cdec17b9e]
_start
??:?
```

**Note**: The script needs to be called from the same directory as UltraGrid
was called for the script to be able to find the executable.

