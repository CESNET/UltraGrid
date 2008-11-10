/*
//	Copyright (c) 1996-2008 DVS Digital Video Systems AG
//
//	DVSDriver:  Debug header
//
*/


#ifndef _PS_DEBUG_H
#define _PS_DEBUG_H

#ifndef DPF
# define DPF DPF_printf
#endif

// ANC
#define DPF_ANCMASK_SETUP         0x00000001
#define DPF_ANCMASK_TABLE         0x00000002
#define DPF_ANCMASK_INPUT         0x00000004
#define DPF_ANCMASK_OUTPUT        0x00000008


// Audio
#define DPF_AUDIOMASK_1	          0x00000001
#define DPF_AUDIOMASK_2	          0x00000002


// DMA
#define DPF_DMAMASK_SETUP         0x00000001
#define DPF_DMAMASK_VERBOSE       0x00000002
#define DPF_DMAMASK_TIMING        0x00000004
#define DPF_DMAMASK_DRIVER        0x00000008
#define DPF_DMAMASK_ALENLIST      0x00000010

// Fifoapi
#define DPF_FIFOMASK_1            0x00000001
#define DPF_FIFOMASK_2            0x00000002

// JP2K
#define DPF_JP2KMASK_MISC         0x00000001
#define DPF_JP2KMASK_HEADER       0x00000002
#define DPF_JP2KMASK_ISSUE        0x00000004
#define DPF_JP2KMASK_TIMING       0x00000008
#define DPF_JP2KMASK_AES          0x00000010
#define DPF_JP2KMASK_ANY          0xffffffff

// IRQ
#define DPF_IRQMASK_AUDIO_DISPLAY 0x00000001
#define DPF_IRQMASK_AUDIO_RECORD  0x00000002
#define DPF_IRQMASK_VIDEO_DISPLAY 0x00000004
#define DPF_IRQMASK_VIDEO_RECORD  0x00000008
#define DPF_IRQMASK_RENDER        0x00000010
#define DPF_IRQMASK_LTC           0x00000020
#define DPF_IRQMASK_DMA           0x00000040
#define DPF_IRQMASK_TIME          0x00000080

// Sched
#define DPF_SCHEDMASK_MISC        0x00000001
#define DPF_SCHEDMASK_VR          0x00000002
#define DPF_SCHEDMASK_VO          0x00000004
#define DPF_SCHEDMASK_VI          0x00000008
#define DPF_SCHEDMASK_AO          0x00000010
#define DPF_SCHEDMASK_AI          0x00000020
#define DPF_SCHEDMASK_OUTDVI      0x00000040
#define DPF_SCHEDMASK_OUTSD       0x00000080
#define DPF_SCHEDMASK_PULLDOWN    0x00000100
#define DPF_SCHEDMASK_PHASE       0x00000200
#define DPF_SCHEDMASK_TRACE       0x00000400
#define DPF_SCHEDMASK_RENDER      0x00000800

// Render
#define DPF_RENDERMASK_STATE      0x00000001
#define DPF_RENDERMASK_MALLOC     0x00000002
#define DPF_RENDERMASK_ISSUE      0x00000004
#define DPF_RENDERMASK_HEADERDUMP 0x00000008


// Timecode
#define DPF_TCMASK_CLOSEDCAPTION  0x00000001
#define DPF_TCMASK_DVITC_INPUT    0x00000002
#define DPF_TCMASK_DVITC_OUTPUT   0x00000004
#define DPF_TCMASK_LTC_INPUT      0x00000008
#define DPF_TCMASK_LTC_OUTPUT     0x00000010
#define DPF_TCMASK_VITC_DECODE    0x00000020
#define DPF_TCMASK_VITC_INPUT     0x00000040
#define DPF_TCMASK_VITC_OUTPUT    0x00000080
#define DPF_TCMASK_VTRTC          0x00000100
#define DPF_TCMASK_RECORD         0x00000200
#define DPF_TCMASK_OPTIONAT       0x00000400

//Video
#define DPF_VIDEOMASK_VIDEO	  0x00000001
#define DPF_VIDEOMASK_VIDEO2	  0x00000002
#define DPF_VIDEOMASK_VIDEO3      0x00000004
#define DPF_VIDEOMASK_CLOCK       0x00000008
#define DPF_VIDEOMASK_DETECT      0x00000010

// Misc
#define DPF_MASK_CAPTURE          0x00000001
#define DPF_MASK_CMD	          0x00000002
#define DPF_MASK_CTRL 	          0x00000004
#define DPF_MASK_DRAM	          0x00000008
#define DPF_MASK_HEADER	          0x00000010
//#define DPF_MASK_HEADERJP2K       0x00000020
#define DPF_MASK_HEADERDECODE     0x00000040
#define DPF_MASK_HEADERENCODE     0x00000080
#define DPF_MASK_FLASH            0x00000100
#define DPF_MASK_I2C              0x00000200
#define DPF_MASK_INIT	          0x00000400
#define DPF_MASK_IO	          0x00000800
//#define DPF_MASK_JP2K             0x00001000
#define DPF_MASK_MALLOC	          0x00002000
#define DPF_MASK_OPEN	          0x00004000
#define DPF_MASK_MASTER   	  0x00008000
#define DPF_MASK_MASTER2   	  0x00010000
#define DPF_MASK_OFFSETS          0x00020000
#define DPF_MASK_PULLDOWN         0x00040000
#define DPF_MASK_QUEUE            0x00080000
#define DPF_MASK_QUEUEAPI         0x00100000
#define DPF_MASK_VTRSLAVE         0x00400000
#define DPF_MASK_VTRMASTER        0x00800000

#ifdef DRIVER
extern uint32 ps_debug;
extern uint32 ps_debug_anc;
extern uint32 ps_debug_audio;
extern uint32 ps_debug_dma;
extern uint32 ps_debug_fifo;
extern uint32 ps_debug_irq;
extern uint32 ps_debug_jp2k;
extern uint32 ps_debug_render;
extern uint32 ps_debug_sched;
extern uint32 ps_debug_timecode;
extern uint32 ps_debug_video;
#endif

#if defined(DEBUG)
# define DPF_CHECK(mask)          if(ps_debug & mask)
# define DPF_CHECK_ANC(mask)      if(ps_debug_anc & mask)
# define DPF_CHECK_AUDIO(mask)    if(ps_debug_audio & mask)
# define DPF_CHECK_DMA(mask)      if(ps_debug_dma & mask)
# define DPF_CHECK_FIFO(mask)     if(ps_debug_fifo & mask)
# define DPF_CHECK_IRQ(mask)      if(ps_debug_irq & mask)
# define DPF_CHECK_JP2K(mask)     if(ps_debug_jp2k & mask)
# define DPF_CHECK_RENDER(mask)   if(ps_debug_render & mask)
# define DPF_CHECK_SCHED(mask)    if(ps_debug_sched & mask)
# define DPF_CHECK_TIMECODE(mask) if(ps_debug_timecode & mask)
# define DPF_CHECK_VIDEO(mask)    if(ps_debug_video & mask)
# define DPF_DEBUG                DPF
#elif defined(UNIX) || defined(macintosh)
# define DPF_CHECK(mask)          if(0) 
# define DPF_CHECK_ANC(mask)      if(0) 
# define DPF_CHECK_AUDIO(mask)    if(0)
# define DPF_CHECK_DMA(mask)      if(0)
# define DPF_CHECK_FIFO(mask)     if(0)
# define DPF_CHECK_IRQ(mask)      if(0)
# define DPF_CHECK_JP2K(mask)     if(0)
# define DPF_CHECK_RENDER(mask)   if(0)
# define DPF_CHECK_SCHED(mask)    if(0)
# define DPF_CHECK_TIMECODE(mask) if(0)
# define DPF_CHECK_VIDEO(mask)    if(0)
# define DPF_DEBUG                if(0) DPF
#else
# define DPF_CHECK(mask)          ;/ ## /
# define DPF_CHECK_ANC(mask)      ;/ ## /
# define DPF_CHECK_AUDIO(mask)    ;/ ## /
# define DPF_CHECK_DMA(mask)      ;/ ## /
# define DPF_CHECK_FIFO(mask)     ;/ ## /
# define DPF_CHECK_IRQ(mask)      ;/ ## /
# define DPF_CHECK_JP2K(mask)     ;/ ## /
# define DPF_CHECK_RENDER(mask)   ;/ ## /
# define DPF_CHECK_SCHED(mask)    ;/ ## /
# define DPF_CHECK_TIMECODE(mask) ;/ ## /
# define DPF_CHECK_VIDEO(mask)    ;/ ## /
# define DPF_DEBUG                ;/ ## /
#endif

#if defined(UNIX) || defined(macintosh)
# define DPF_NOP                if(0) DPF
#else
# define DPF_NOP                ;/ ## /
#endif


#define DPF_CAPTURE     DPF_CHECK(DPF_MASK_CAPTURE)   DPF
#define DPF_CMD    	DPF_CHECK(DPF_MASK_CMD)       DPF
#define DPF_CTRL   	DPF_CHECK(DPF_MASK_CTRL)      DPF
#define DPF_DRAM	DPF_CHECK(DPF_MASK_DRAM)      DPF
#define DPF_HEADER	DPF_CHECK(DPF_MASK_HEADER)    DPF
#define DPF_FLASH       DPF_CHECK(DPF_MASK_FLASH)     DPF
#define DPF_INIT 	DPF_CHECK(DPF_MASK_INIT)      DPF
#define DPF_IO    	DPF_CHECK(DPF_MASK_IO)        DPF
#define DPF_I2C    	DPF_CHECK(DPF_MASK_I2C)       DPF
#define DPF_MALLOC  	DPF_CHECK(DPF_MASK_MALLOC)    DPF
#define DPF_MASTER	DPF_CHECK(DPF_MASK_MASTER)    DPF
#define DPF_MASTER2	DPF_CHECK(DPF_MASK_MASTER2)   DPF
#define DPF_OPEN    	DPF_CHECK(DPF_MASK_OPEN)      DPF
#define DPF_PULLDOWN    DPF_CHECK(DPF_MASK_PULLDOWN)  DPF
#define DPF_QUEUE	DPF_CHECK(DPF_MASK_QUEUE)     DPF
#define DPF_QUEUEAPI	DPF_CHECK(DPF_MASK_QUEUEAPI)  DPF
#define DPF_OFFSETS     DPF_CHECK(DPF_MASK_OFFSETS)   DPF
#define DPF_VTRSLAVE	DPF_CHECK(DPF_MASK_VTRSLAVE)  DPF
#define DPF_VTRMASTER	DPF_CHECK(DPF_MASK_VTRMASTER) DPF

#define DPF_MASK_ANY      0xffffffff

// ANC
#define DPF_ANC_SETUP     DPF_CHECK_ANC(DPF_ANCMASK_SETUP)  DPF
#define DPF_ANC_TABLE     DPF_CHECK_ANC(DPF_ANCMASK_TABLE)  DPF
#define DPF_ANC_INPUT     DPF_CHECK_ANC(DPF_ANCMASK_INPUT)  DPF
#define DPF_ANC_OUTPUT    DPF_CHECK_ANC(DPF_ANCMASK_OUTPUT) DPF

// Audio
#define DPF_AUDIO	  DPF_CHECK_AUDIO(DPF_AUDIOMASK_1)    DPF
#define DPF_AUDIO2   	  DPF_CHECK_AUDIO(DPF_AUDIOMASK_2)    DPF

// DMA
#define DPF_DMA_SETUP  	  DPF_CHECK_DMA(DPF_DMAMASK_SETUP)    DPF
#define DPF_DMA_VERBOSE   DPF_CHECK_DMA(DPF_DMAMASK_VERBOSE)  DPF
#define DPF_DMA_TIMING    DPF_CHECK_DMA(DPF_DMAMASK_TIMING)   DPF
#define DPF_DMA_DRIVER    DPF_CHECK_DMA(DPF_DMAMASK_DRIVER)   DPF
#define DPF_DMA_ALENLIST  DPF_CHECK_DMA(DPF_DMAMASK_ALENLIST) DPF

// Fifoapi
#define DPF_FIFO  	  DPF_CHECK_FIFO(DPF_FIFOMASK_1)      DPF
#define DPF_FIFO2	  DPF_CHECK_FIFO(DPF_FIFOMASK_2)      DPF

// IRQ
#define DPF_IRQ_AUDIO_DISPLAY	    DPF_CHECK_IRQ(DPF_IRQMASK_AUDIO_DISPLAY)  DPF
#define DPF_IRQ_AUDIO_RECORD        DPF_CHECK_IRQ(DPF_IRQMASK_AUDIO_RECORD)   DPF
#define DPF_IRQ_DMA                 DPF_CHECK_IRQ(DPF_IRQMASK_DMA)            DPF
#define DPF_IRQ_LTC                 DPF_CHECK_IRQ(DPF_IRQMASK_LTC)            DPF
#define DPF_IRQ_RENDER              DPF_CHECK_IRQ(DPF_IRQMASK_RENDER)         DPF
#define DPF_IRQ_VIDEO_DISPLAY	    DPF_CHECK_IRQ(DPF_IRQMASK_VIDEO_DISPLAY)  DPF
#define DPF_IRQ_VIDEO_RECORD        DPF_CHECK_IRQ(DPF_IRQMASK_VIDEO_RECORD)   DPF

// JP2K
#define DPF_JP2K   	            DPF_CHECK_JP2K(DPF_JP2KMASK_ANY)          DPF
#define DPF_JP2K_HEADER             DPF_CHECK_JP2K(DPF_JP2KMASK_HEADER)       DPF
#define DPF_JP2K_ISSUE              DPF_CHECK_JP2K(DPF_JP2KMASK_ISSUE)        DPF
#define DPF_JP2K_TIMING             DPF_CHECK_JP2K(DPF_JP2KMASK_TIMING)       DPF
#define DPF_JP2K_AES                DPF_CHECK_JP2K(DPF_JP2KMASK_AES)          DPF

// Render
#define DPF_RENDER                  DPF_CHECK_RENDER(DPF_MASK_ANY)            DPF
#define DPF_RENDER_STATE            DPF_CHECK_RENDER(DPF_RENDERMASK_STATE)    DPF
#define DPF_RENDER_MALLOC           DPF_CHECK_RENDER(DPF_RENDERMASK_MALLOC)   DPF
#define DPF_RENDER_ISSUE            DPF_CHECK_RENDER(DPF_RENDERMASK_ISSUE)    DPF

// Sched
#define DPF_SCHED 	            DPF_CHECK_SCHED(DPF_SCHEDMASK_MISC)       DPF
#define DPF_SCHED_VR                DPF_CHECK_SCHED(DPF_SCHEDMASK_VR)         DPF
#define DPF_SCHED_VO                DPF_CHECK_SCHED(DPF_SCHEDMASK_VO)         DPF
#define DPF_SCHED_VI                DPF_CHECK_SCHED(DPF_SCHEDMASK_VI)         DPF
#define DPF_SCHED_AO                DPF_CHECK_SCHED(DPF_SCHEDMASK_AO)         DPF
#define DPF_SCHED_AI                DPF_CHECK_SCHED(DPF_SCHEDMASK_AI)         DPF
#define DPF_SCHED_OUTDVI            DPF_CHECK_SCHED(DPF_SCHEDMASK_OUTDVI)     DPF
#define DPF_SCHED_OUTSD             DPF_CHECK_SCHED(DPF_SCHEDMASK_OUTSD)      DPF
#define DPF_SCHED_PULLDOWN          DPF_CHECK_SCHED(DPF_SCHEDMASK_PULLDOWN)   DPF
#define DPF_SCHED_PHASE             DPF_CHECK_SCHED(DPF_SCHEDMASK_PHASE)      DPF
#define DPF_SCHED_TRACE             DPF_CHECK_SCHED(DPF_SCHEDMASK_TRACE)      DPF
#define DPF_SCHED_RENDER            DPF_CHECK_SCHED(DPF_SCHEDMASK_RENDER)     DPF

// Timecode
#define DPF_TIMECODE_CLOSEDCAPTION  DPF_CHECK_TIMECODE(DPF_TCMASK_CLOSEDCAPTION)  DPF
#define DPF_TIMECODE_DVITC_INPUT    DPF_CHECK_TIMECODE(DPF_TCMASK_DVITC_INPUT)    DPF
#define DPF_TIMECODE_DVITC_OUTPUT   DPF_CHECK_TIMECODE(DPF_TCMASK_DVITC_OUTPUT)   DPF
#define DPF_TIMECODE_LTC_INPUT      DPF_CHECK_TIMECODE(DPF_TCMASK_LTC_INPUT)      DPF
#define DPF_TIMECODE_LTC_OUTPUT     DPF_CHECK_TIMECODE(DPF_TCMASK_LTC_OUTPUT)     DPF
#define DPF_TIMECODE_VITC_DECODE    DPF_CHECK_TIMECODE(DPF_TCMASK_VITC_DECODE)    DPF
#define DPF_TIMECODE_VITC_INPUT     DPF_CHECK_TIMECODE(DPF_TCMASK_VITC_INPUT)     DPF
#define DPF_TIMECODE_VITC_OUTPUT    DPF_CHECK_TIMECODE(DPF_TCMASK_VITC_OUTPUT)    DPF
#define DPF_TIMECODE_VTRTC          DPF_CHECK_TIMECODE(DPF_TCMASK_VTRTC)          DPF
#define DPF_TIMECODE_RECORD         DPF_CHECK_TIMECODE(DPF_TCMASK_RECORD)         DPF
#define DPF_TIMECODE_OPTIONAT       DPF_CHECK_TIMECODE(DPF_TCMASK_OPTIONAT)       DPF

// Video
#define DPF_VIDEO 	            DPF_CHECK_VIDEO(DPF_VIDEOMASK_VIDEO)      DPF
#define DPF_VIDEO2 	            DPF_CHECK_VIDEO(DPF_VIDEOMASK_VIDEO2)     DPF
#define DPF_VIDEO3 	            DPF_CHECK_VIDEO(DPF_VIDEOMASK_VIDEO3)     DPF
#define DPF_VIDEO_CLOCK             DPF_CHECK_VIDEO(DPF_VIDEOMASK_CLOCK)      DPF
#define DPF_VIDEO_DETECT            DPF_CHECK_VIDEO(DPF_VIDEOMASK_DETECT)     DPF

#define DPF_WARN  	DPF
#define DPF_ERROR	DPF
#define DPF_FATAL 	DPF

#ifdef _DEBUG
# define PE()		DPF_WARN("PE %s:%d\n", __FILE__, __LINE__);
#else
# define PE()		;
#endif

#define ASSERTINPUTJACK(jack) Assert(ps->jack[jack].binput)
#define ASSERTOUTPUTJACK(jack) Assert(!ps->jack[jack].binput)

#ifdef _DEBUG
# define Assert(a)	if(!(a)) DPF_printf("Assert '" #a "' @ %s:%d\r\n", __FILE__, __LINE__)
#else
# define Assert(a)
#endif

#endif	/* !_PS_DEBUG_H */
