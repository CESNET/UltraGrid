/*x*DVS*HEADER*
//      Copyright (c) 1996-2008 DVS Digital Video Systems AG
//
//      DVSDriver:  Common structures and defines
//
*/

#ifndef _PS_COMMON_H
#define _PS_COMMON_H

#ifndef TRUE
# define TRUE 1
#endif

#ifndef FALSE
# define FALSE 0
#endif

#define PCI_VENDORID_XILINX             0x10ee
#define PCI_VENDORID_DVS                0x1a55

#define PCI_DEVICEID_XILINX_SDIO        0xd151
#define PCI_DEVICEID_XILINX_HUGO        0xd152
#define PCI_DEVICEID_XILINX_IRIS        0xd153

#define PCI_DEVICEID_DVS_MASK           0xfff0
#define PCI_DEVICEID_DVS_SDIO           0x0010
#define PCI_DEVICEID_DVS_SDIO_EDDY      0x0011
#define PCI_DEVICEID_DVS_IRIS           0x0020
#define PCI_DEVICEID_DVS_IRIS_LUCY      0x0021
#define PCI_DEVICEID_DVS_HUGO           0x0030

#define PS_CARDTYPE_PSCONF		0
#define PS_CARDTYPE_CLIPBOARD		1
#define PS_CARDTYPE_PCIAUDIO		2
#define PS_CARDTYPE_HDIO		3
#define PS_CARDTYPE_SDIO		4
#define PS_CARDTYPE_HUGO		5
#define PS_CARDTYPE_IRIS		6


typedef struct {
  uintphysaddr physaddr;
  uint32       size;
  uint32       offset;
} ps_alenlist_alen;

/**
//  ps_alenlist -> Address length list for DMA
*/
typedef struct {
  int         allocated;        ///< Number of allocated elementes in alen
  int         count;            ///< Number of elements in alen
  void *      addr;             ///< Virtual address of buffer
  int         size;             ///< Size of buffer
  int         flags;            ///< Any of the PS_ALENLIST_ flags
  void *      mdl;              ///< Memory description list
  ps_alenlist_alen alen[1];     ///< Points to one address/size/offset element.
} ps_alenlist;


#define PS_ALENLIST_PHYSADDR	0x0001
#define PS_ALENLIST_VIRTADDR	0x0002
#define PS_ALENLIST_KERNEL	0x0004
#define PS_ALENLIST_OFFSET	0x0008
#define PS_ALENLIST_READ	0x0010
#define PS_ALENLIST_WRITE	0x0020

#define sizeof_alenlist(x)    (sizeof(ps_alenlist) + (x - 1) * sizeof(ps_alenlist_alen))


/*
//  ps_lock
*/
#ifdef WIN32
typedef struct {
  unsigned long spinlock;
  uint8         irql;
} ps_lock;
#else
typedef struct {
  void * lock;
  int    count;
#if defined(LINUX)
  unsigned long flags;		/* processor flags before locking */
#endif
} ps_lock;
#endif


#define PS_QUEUE_TYPE_VSYNC_RECORD	0x0001
#define PS_QUEUE_TYPE_VSYNC_DISPLAY	0x0002
#define PS_QUEUE_TYPE_AUDIO_RECORD	0x0004
#define PS_QUEUE_TYPE_AUDIO_DISPLAY	0x0008
#define PS_QUEUE_TYPE_FIFO_RECORD       0x0010
#define PS_QUEUE_TYPE_FIFO_DISPLAY      0x0020


#if !defined(WIN32)
# define HAL_MEMCPY_U2K		0
# define HAL_MEMCPY_K2U		1
# define HAL_MEMCPY_K2K		2
# define HAL_MEMCPY_U2U		3
#endif


#define JACK_DEFAULT  0
#define JACK_OUTPUT   0
#define JACK_INPUT    1
#define JACK_COUNT    2

#endif	/* !_PS_COMMON_H */
