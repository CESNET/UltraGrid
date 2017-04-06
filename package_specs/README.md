This directory contains package specifications for ultragrid.

The binary packages are build using the Open Build Service
platform (https://openbuildservice.org). If you need to run
a manual build, e.g. without some feature, apply corresponding
__disable_(vendor)__.patch . To disable multiple features,
apply patches in lexicographical (or any other) order.

RPMs are built as usual, yet DEBs require a small change:
1) pick proper .dsc file and find which debian-patches-series
	it requires
2) rename the series file and move it into respective place
3) rename debian.rules to debian/rules
4) continue with normal build

The pseudoautomated build system deployed for nightly builds
uses template package specifications. To convert the
templates to specifications, launch script deploy-templates.sh,
shipped in this directory.
