/**DVS*HEADER*
 *	Copyright (c) 1996-2008 DVS Digital Video Systems AG
 *
 *	DVSDriver:  Debug header
 *
 */


#ifndef _PS_DEBUG_H
#define _PS_DEBUG_H

#ifndef DPF
# define DPF DPF_printf
#endif

#define DPF_MASK_AUDIO	        0x00000001
#define DPF_MASK_AUDIO2	        0x00000002
#define DPF_MASK_AUDIOIRQ	0x00000004
#define DPF_MASK_OPEN	        0x00000008
#define DPF_MASK_FIFO2  	0x00000010
#define DPF_MASK_IO	        0x00000020
#define DPF_MASK_CMD	        0x00000040
#define DPF_MASK_MALLOC	        0x00000080
#define DPF_MASK_CTRL 	        0x00000100
#define DPF_MASK_DMA	        0x00000200
#define DPF_MASK_DMA2	        0x00000400
#define DPF_MASK_DRAM	        0x00000800
#define DPF_MASK_CAPTURE        0x00001000
#define DPF_MASK_FIFO   	0x00002000
#define DPF_MASK_HEADER	        0x00004000
#define DPF_MASK_INIT	        0x00008000
#define DPF_MASK_IRQ	        0x00010000
#define DPF_MASK_VIDEO3         0x00020000
#define DPF_MASK_MASTER   	0x00040000
#define DPF_MASK_MASTER2   	0x00080000
#define DPF_MASK_QUEUEAPI       0x00100000
#define DPF_MASK_QUEUE          0x00200000
#define DPF_MASK_SCHED	        0x00400000
#define DPF_MASK_OFFSETS        0x00800000
#define DPF_MASK_ANC            0x01000000
#define DPF_MASK_VIDEO	        0x02000000
#define DPF_MASK_VIDEO2	        0x04000000
#define DPF_MASK_VTRSLAVE       0x08000000
#define DPF_MASK_VTRMASTER      0x10000000
#define DPF_MASK_VIDEOIRQ	0x20000000
#define DPF_MASK_TIMECODE       0x40000000
#define DPF_MASK_PULLDOWN       0x80000000

#define DPF_MASK2_I2C           0x00000001

#ifdef DRIVER
extern uint32 ps_debug;
extern uint32 ps_debug2;
#endif

#if defined(DEBUG)
# define DPF_CHECK(mask)        if(ps_debug & mask)
# define DPF_CHECK2(mask)       if(ps_debug2 & mask)
# define DPF_DEBUG              DPF
#elif defined(UNIX) || defined(macintosh)
# define DPF_CHECK(mask)        if(0) 
# define DPF_CHECK2(mask)       if(0) 
# define DPF_DEBUG              if(0) DPF
#else
# define DPF_CHECK(mask)        ;/ ## /
# define DPF_CHECK2(mask)       ;/ ## /
# define DPF_DEBUG              ;/ ## /
#endif

#if defined(UNIX) || defined(macintosh)
# define DPF_NOP                if(0) DPF
#else
# define DPF_NOP                ;/ ## /
#endif

#define DPF_ANC         DPF_CHECK(DPF_MASK_ANC)       DPF
#define DPF_AUDIO	DPF_CHECK(DPF_MASK_AUDIO)     DPF
#define DPF_AUDIOIRQ	DPF_CHECK(DPF_MASK_AUDIOIRQ)  DPF
#define DPF_AUDIO2   	DPF_CHECK(DPF_MASK_AUDIO2)    DPF
#define DPF_CAPTURE     DPF_CHECK(DPF_MASK_CAPTURE)   DPF
#define DPF_CMD    	DPF_CHECK(DPF_MASK_CMD)       DPF
#define DPF_CTRL   	DPF_CHECK(DPF_MASK_CTRL)      DPF
#define DPF_DMA   	DPF_CHECK(DPF_MASK_DMA)       DPF
#define DPF_DMA2  	DPF_CHECK(DPF_MASK_DMA2)      DPF
#define DPF_DRAM	DPF_CHECK(DPF_MASK_DRAM)      DPF
#define DPF_FIFO  	DPF_CHECK(DPF_MASK_FIFO)      DPF
#define DPF_FIFO2	DPF_CHECK(DPF_MASK_FIFO2)     DPF
#define DPF_HEADER	DPF_CHECK(DPF_MASK_HEADER)    DPF
#define DPF_INIT 	DPF_CHECK(DPF_MASK_INIT)      DPF
#define DPF_IO    	DPF_CHECK(DPF_MASK_IO)        DPF
#define DPF_I2C    	DPF_CHECK2(DPF_MASK2_I2C)     DPF
#define DPF_IRQ	  	DPF_CHECK(DPF_MASK_IRQ)       DPF
#define DPF_MALLOC  	DPF_CHECK(DPF_MASK_MALLOC)    DPF
#define DPF_MASTER	DPF_CHECK(DPF_MASK_MASTER)    DPF
#define DPF_MASTER2	DPF_CHECK(DPF_MASK_MASTER2)   DPF
#define DPF_OPEN    	DPF_CHECK(DPF_MASK_OPEN)      DPF
#define DPF_PULLDOWN    DPF_CHECK(DPF_MASK_PULLDOWN)  DPF
#define DPF_QUEUE	DPF_CHECK(DPF_MASK_QUEUE)     DPF
#define DPF_QUEUEAPI	DPF_CHECK(DPF_MASK_QUEUEAPI)  DPF
#define DPF_SCHED 	DPF_CHECK(DPF_MASK_SCHED)     DPF
#define DPF_OFFSETS     DPF_CHECK(DPF_MASK_OFFSETS)   DPF
#define DPF_TIMECODE    DPF_CHECK(DPF_MASK_TIMECODE)  DPF
#define DPF_VIDEO 	DPF_CHECK(DPF_MASK_VIDEO)     DPF
#define DPF_VIDEO2 	DPF_CHECK(DPF_MASK_VIDEO2)    DPF
#define DPF_VIDEO3 	DPF_CHECK(DPF_MASK_VIDEO3)    DPF
#define DPF_VIDEOIRQ	DPF_CHECK(DPF_MASK_VIDEOIRQ)  DPF
#define DPF_VTRSLAVE	DPF_CHECK(DPF_MASK_VTRSLAVE)  DPF
#define DPF_VTRMASTER	DPF_CHECK(DPF_MASK_VTRMASTER) DPF

#define DPF_WARN  	DPF
#define DPF_ERROR	DPF
#define DPF_FATAL 	DPF

#ifdef _DEBUG
# define PE()		DPF_WARN("PE %s:%d\n", __FILE__, __LINE__);
# define NI()		DPF_WARN("NI: %s:%d\n", __FILE__, __LINE__);
#else
# define PE()		;
# define NI()		;
#endif

#ifdef _DEBUG
# define Assert(a)	if(!(a)) DPF_printf("Assert '" #a "' @ %s:%d\r\n", __FILE__, __LINE__)
#else
# define Assert(a)
#endif

#endif	/* !_PS_DEBUG_H */
