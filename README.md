UltraGrid - A High Definition Collaboratory
===========================================
[![Facebook Follow](https://img.shields.io/badge/Facebook-follow-blue)](https://www.facebook.com/UltraGrid/)
[![Mastodon Follow](https://img.shields.io/badge/Mastodon-follow-blue)](https://mastodon.technology/@UltraGrid)
[![Twitter Follow](https://img.shields.io/badge/Twitter-follow-blue)](https://twitter.com/UltraGrid_CZ)
[![@ultragrid:matrix.org](https://img.shields.io/badge/Matrix-chat-black)](https://matrix.to/#/!IrTYOLJOmZIoTBiITI:matrix.org?via=matrix.org)
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
       REPORTING-BUGS.md Recommendations for reporting bugs
       bin/              Compiled binaries
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


Performance Tuning: Network
---------------------------

   If transmitting *uncompressed video* stream to achieve optimal performance
   with high definition video, it may be necessary to tune your system's
   network parameters to more aggressive values than used by default.

   A key factor affecting performance is the path MTU. It is unlikely that
   the system will sustain gigabit rates with the 1500 octet Ethernet MTU. 
   If using a gigabit Ethernet you may be able to improve performance by
   setting an 8192 octet MTU on the interface, provided all intermediate
   hops on the path from sender to receiver support the large MTU. 

   UltraGrid attempts to increase the UDP receive socket buffer from the
   default value (typically 64 kilobytes) to 4/6 megabytes. If successful,
   this will make the system more robust to scheduling variations and
   better able to accept bursty packet arrivals. UltraGrid will notify
   you if it cannot increase buffers. You should follow those instructions
   and set your system according to it.

   Interrupt processing load on the receiver host may be significant when
   running at high rates. Depending on your network interface hardware it
   may be possible to coalesce interrupts to reduce this load, although
   the settings to do this are highly driver dependent. On FreeBSD, the
   use of network device polling may also help performance: see the man
   page for "polling" in section 4 of the manual.

   In many cases, the performance of your network interface card may be
   limited by host bus performance (this is particularly an issue at high
   rates, for example when using HD format video).


Performance Tuning: Display devices
-----------------------------------

   If using a HW grabbing card (eg. DVS) as a display device, the
   key factor limiting performance is PCI bus contention. Ensure that
   the grabbing card is on a separate PCI bus to the network card --
   this typically requires a server class motherboard. On Linux, the
   PCI bus topology can be displayed using "lspci -tv", for example:

        [root@ormal root]# lspci -tv
        -+-[03]---06.0  Xilinx, Inc.: Unknown device d150
         +-[01]-+-02.0-[02]--+-04.0  Adaptec 7899P
         |      |            \-04.1  Adaptec 7899P
         |      \-0e.0  3Com Corporation 3c985 1000BaseSX
         \-[00]-+-00.0  ServerWorks CNB20HE
                +-00.1  ServerWorks CNB20HE
                +-00.2  ServerWorks: Unknown device 0006
                +-00.3  ServerWorks: Unknown device 0006
                +-04.0  Intel Corporation 82557 [Ethernet Pro 100]
                +-0e.0  ATI Technologies Inc Rage XL
                +-0f.0  ServerWorks OSB4
                \-0f.1  ServerWorks: Unknown device 0211
        [root@ormal root]# 

   showing an DVS card on PCI bus [03] (the card shows as a Xilinx
   device) and a gigabit Ethernet card on PCI bus [02] (the 3Com entry).

   For software display, you can use SDL or OpenGL display. Both are
   accelerated (Mac and Linux) if you have properly configured video
   drivers. On Linux, basic operability can be checked with following
   commands. If configured properly, both should display driver
   properties:
        [root@ormal root]# glxinfo
        <-- output omitted -->
   and for SDL (accelerated through XVideo:
        [root@ormal root]# xvinfo
        <-- output omitted -->

   If you intend to use some of DXT compressions, recommended driver
   is OpenGL, which can display it natively. When using other display
   drivers, decompression is still done throught OpenGL and then displayed
   with requested video driver.


Performance Tuning: Other Factors
---------------------------------

   The UltraGrid system will attempt to enable POSIX real-time scheduling
   to improve performance. This behaviour is disabled by default now, because
   it can occupy the whole system when enabled, but it can be stil enabled by
   '--enable-rt' configure option. If you see the message:

        WARNING: Unable to set real-time scheduling

   when starting the application, this means that the operating system did
   not permit it to enable real-time scheduling. The application will run,
   but with reduced performance. The most likely reason for failure to set
   realtime scheduling is that the application has insufficient privilege:
   it should either be run by root, or be made setuid root. A similar
   message:

        WARNING: System does not support real-time scheduling

   indicates that your operating system does not support POSIX real-time
   scheduling. The application will still run, but performance may be less
   than desired.


   You can find more operating system tweaks at this page:
   https://github.com/CESNET/UltraGrid/wiki/OS-Setup-UltraGrid

                                  - * - 

