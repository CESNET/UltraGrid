UltraGrid - A High Definition Collaboratory
===========================================
[![Facebook Follow](https://img.shields.io/badge/Facebook-follow-blue)](https://www.facebook.com/UltraGrid/)
[![Mastodon Follow](https://img.shields.io/badge/Mastodon-follow-blue)](https://mastodon.technology/@UltraGrid)
[![Twitter Follow](https://img.shields.io/badge/Twitter-follow-blue)](https://twitter.com/UltraGrid_CZ)
[![Web Visit](https://img.shields.io/badge/web-visit-orange)](http://www.ultragrid.cz)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/2851/badge.svg)](https://scan.coverity.com/projects/2851)
[![C/C++ CI](../../workflows/C%2FC%2B%2B%20CI/badge.svg)](../../actions)

![UltraGrid Logo](data/ultragrid-logo-text.png)

   * Copyright (c) 2001-2004 University of Southern California 
   * Copyright (c) 2003-2004 University of Glasgow
   * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
   * Copyright (c) 2005-2020 CESNET z.s.p.o.
   * All rights reserved.

   This software is distributed under license, see the file
   [COPYRIGHT](COPYRIGHT) for full terms and conditions.


Contents
--------

   - [About UltraGrid](#about-ultragrid)
   - [Hardware and Software Requirements](#hardware-and-software-requirements)
     * [Required Software Preliminaries](#required-software-preliminaries)
   - [Using the UltraGrid System](#using-the-ultragrid-system)
   - [Documentation](#documentation)

About UltraGrid
---------------

   **UltraGrid** brought by [CESNET's](https://www.cesnet.cz) Laboratory of
   Advanced Networking Technologies ([Sitola](https://www.sitola.cz)) is
   a software implementation of high-quality low-latency video and
   audio transmissions using commodity hardware. Supported resolutions range
   through *HD* (1920x1080) up to *8K* (7680x2160) with up to 60 frames per second.
   Other features are listed [here](https://github.com/CESNET/UltraGrid/wiki).

   The high-quality is achieved either by using uncompressed
   streams or streams with very low compression ratio. End-to-end transmission
   latency (i.e., all the way from the camera to the display) is about 100ms,
   but it varies based on camera and capture cards being used. UltraGrid was
   originally a research project used to demonstrate the possibilities of 10Gbps
   networks and to study multi-point data distribution in such environments.
   Recent advances in the field of GPU-accelerated low-latency codecs extend its
   usability also to Gigabit networks. High compression ratio compressions allow
   further use of any commodity network connection including a shared Internet
   connection.

   UltraGrid is supported on stations with
   Linux, Windows or macOS operating system. The software is open-source
   distributed under BSD license, i.e., we're interested in both
   research/academic and commercial applications. Nowadays, main application
   areas are collaborative environments, medical, cinematography and
   broadcasting applications, as well as various educational activities.

   It is a fork of the original UltraGrid developed by Colin Perkins, Ladan
   Gharai, et al..

   Our work is supported by CESNET research intents "Optical Network of National
   Research and Its New Applications" (MŠM 6383917201), CESNET Large
   Infrastructure (LM2010005), CESNET E-Infrastructure (LM2015042) and partially
   also by Masaryk University research intent "Parallel and Distributed Systems"
   (MŠM 0021622419). 

   The contents of this directory are as follows:

       COPYRIGHT         Full license terms and conditions
       INSTALL           Installation instructions
       NEWS              Change log and modification history
       README.md         This file
       bin/              Compiled binaries
       doc/              Documentation
       src/              Source code for the UltraGrid system
       test/             Source code and binaries for test routines
       Makefile.in       Build script
       acconfig.h        "       "
       config.guess      "       "
       config.sub        "       "
       configure         "       "
       configure.ac      "       "
       install-shx       "       "


Hardware and Software Requirements
----------------------------------

   Recommended Hardware Setup:
   - 64-bit CPU with at least 2 cores
   - **OpenGL** compatible graphics card
     - *DXT* compression on GPU is tested with **OpenGL 3.3**
     - *GPUJPEG* compression requires a **NVidia** card
     - various other HW accelerations supported with recent cards (**NVENC**,
       **QuickSync**, **VideoToolbox**)
   - For uncompressed 1.5Gbps streams (either sending or receiving), **10GbE**
     network interface card is needed
     - We test with PCIe Myrinet 10GbE 
   - For *SDI* send/receive capabilities, **AJA**, **Bluefish444**,
     **Blackmagic**, **DELTACAST** or **Magewell** card is required
     - Magewell module in UG support only capturing

   Video capture card should be located on a separate PCI bus from network card if possible.

### Required Software Preliminaries
   You will need this software (in brackets are optional features for which you'll need it):

   - AMD/NVidia proprietary drivers for optimal performance
   - AJA/Blackmagic/DELTACAST drivers
   - For macOS Homebrew or MacPorts are recommended

   To compile UltraGrid you will need to prepare build environment and
   install dependencies for various modules. For up-to-date information
   please refer to our
   [wiki](https://github.com/CESNET/UltraGrid/wiki/Compile-UltraGrid-%28Source%29).

Using the UltraGrid System
--------------------------

   The [INSTALL](INSTALL) gives instructions for building the UltraGrid system.
   Once the system has been built, the `uv` binary will be present. This
   can be invoked as follows:

       uv -t <capture_device> -c <compression> hostname     (on the sender)
       uv -d <display_device> hostname                      (on the receiver)

   The **\<display_device\>** is one of the list viewed with `-d help`.

   The **\<capture_device\>** is one of the list viewed with `-t help`. Name
   of capture device usually follows with configuration of video mode,
   video input etc. All options can be interactively dispayed using built-in
   help, eg. `-t decklink:help`.

   The **\<compression\>** specifies the selected video compression to be
   used. Similarly as for other options, available options can be viewed
   by `-c help`. If compression is not specified, video is transmitted
   *uncompressed* (in that case consider setting **MTU** with `-m <mtu>`).

   Further options follow UltraGrid command-line help (-h) or visit this
   [wiki page](https://github.com/CESNET/UltraGrid/wiki/Running-UltraGrid)
   for further information.

   As an example, if a user on host "ormal" wishes to send audio and video
   captured using a BMD DeckLink card another user on host "curtis" with
   a display using the OpenGL driver and Portaudio audio playback, then
   the user on host "ormal" would run:

       uv -t decklink -c libavcodec:codec=H.264 -s embedded --audio-codec OPUS curtis

   while the user on "curtis" would run:

       uv -d gl -r portaudio ormal

   The system requires access to UDP ports 5004 and 5005: you should open
   these ports on any firewall on the network path. Uncompressed high definition
   video formats require approximately 1 Gigabit per second of network capacity.
   Using different supported compression schemes, the needed network capacity
   can be as low as 10 Megabits per second for a high definition video.

Documentation
-------------
   Documentation can be found either _offline_ (apart from this document) and
   _online_. The online documentation is more comprehensive and up-to-date,
   offline is rather complementary.

   The **online** documentation is available in our GitHub
   [wiki](https://github.com/CESNET/UltraGrid/wiki).

   UltraGrid _built-in_ documentation can be found in [doc](doc) subdirectory,
   these documents are available:

   - [Adding modules](doc/ADDING-MODULES.md) (**developers only**) - information
     how to add new UltraGrid modules
   - [Performance tuning](doc/PERFORMANCE-TUNING.md) - various tweaks to improve
     UltraGrid performance
   - [Reporting bugs](doc/REPORTING_BUGS.md) - recommended steps for reporting
     bugs


