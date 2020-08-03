#!/bin/bash
osc up
osc service localrun
env FORCE_REBUILD=yes ../../UltraGrid/package_specs/ultragrid-autobuild.sh $(realpath .) master $(realpath ../../UltraGrid_readonly/) $(realpath ../../UltraGrid_readonly/package_specs/)
