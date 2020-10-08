Performance Tuning
==================

Network
-------

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


Display devices
---------------

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


Other Factors
-------------

   **Note:** This is left only as a legacy behavior - currently UltraGrid
   is intended to run without _real-time_ priority (although still being
   able to compiled with it). This may not be recommended in a general case,
   however.

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

