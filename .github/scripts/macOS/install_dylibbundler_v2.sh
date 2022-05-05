#!/bin/sh -eu

# 2020-05-02
# Used dylib bundler v2 from https://github.com/SCG82/macdylibbundler instead of the
# original because it has far better execution time (and perhaps also other improvements)

# 2022-05-05
# Reverting to master because of this change:
#   https://github.com/auriamg/macdylibbundler/commit/9b3779e040c1436cedb0c75e1702105d94bc8a10
# that should fix failing CI builds (run 658). Relevant otool outputs:
# failed:
# /usr/local/Cellar/gcc/11.3.0/lib/gcc/11/libgcc_s.1.dylib:
#         /usr/local/opt/gcc/lib/gcc/11/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
#         /usr/local/opt/gcc/lib/gcc/11/libgcc_s.1.1.dylib (compatibility version 1.0.0, current version 1.1.0, reexport)
#         /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1281.100.1)
# vs previous (working):
# /usr/local/Cellar-bkp/gcc/11.2.0_3/lib/gcc/11/libgcc_s.1.dylib:
#         /usr/local/opt/gcc/lib/gcc/11/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
#         /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1281.100.1)
#
# Seems to no longer has a significant impact on execution time.
#
# TODO: remove when sure that it is not needed and replace its call by the Homebres installation

brew install dylibbundler

