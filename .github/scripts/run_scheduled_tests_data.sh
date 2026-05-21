# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
#
# test cases for run_scheduled_tests.sh
#
# For the documentation of add_test parameters see the
# run_scheduled_tests.sh file.

# UltraGrid
add_test -v                                                  # basic sanity test
add_test --nonexistent-param                   should_fail
add_test "-d gl"                               should_timeout,Linux_only
add_test "-d gl:unknown_param"                 should_fail
add_test "-d sdl"                              should_timeout
add_test "-t testcard -c lavc:e=libx265 -f rs -d dummy" should_timeout
add_test "-t spout:check_lib"                  Windows_only

# rxtx/compress help + incorrect options
add_test "-c help -t testcard"
add_test "-c lavc:help -t testcard"
add_test "-x help"
add_test "-x sdp:help"
add_test "-c nonexistent"                      should_fail
add_test "-c lavc:wrongopt -t testcard"        should_fail
add_test "-x nonexistent"                      should_fail
add_test "-x sdp:wrongopt"                     should_fail

# reflector
add_test -v                                    run_reflector # basic sanity test
add_test "8M 5004"                             run_reflector,should_timeout
add_test "8M 5004 -c lavc:c=H.264 -P 8000 ::1" run_reflector,should_timeout
