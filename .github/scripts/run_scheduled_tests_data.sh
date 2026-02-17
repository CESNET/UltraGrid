# test cases for run_scheduled_tests.sh
#
# For the documentation of add_test parameters see the
# run_scheduled_tests.sh file.

# UltraGrid
add_test -v                                                  # basic sanity test
add_test --nonexistent-param                   should_fail
add_test "-d sdl"                              should_timeout
add_test "-t testcard -c lavc:e=libx265 -f rs -d dummy" should_timeout
add_test "-t spout:check_lib"                  Windows_only

# reflector
add_test -v                                    run_reflector # basic sanity test
add_test "8M 5004"                             run_reflector,should_timeout
