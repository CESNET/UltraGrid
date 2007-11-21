/*
 * Define OS
 */
#undef HAVE_MACOSX

/*
 * Define this if you want IPv6 support.
 */
#undef HAVE_IPv6

/* 
 * Define this is you have a DVS HDstation card
 */
#undef HAVE_HDSTATION

/*
 * Define this if you want FastDXT support
 */
#undef HAVE_FASTDXT

/* Audio device related */
#undef HAVE_MACOSX_AUDIO
#undef HAVE_SPARC_AUDIO
#undef HAVE_SGI_AUDIO
#undef HAVE_PCA_AUDIO
#undef HAVE_LUIGI_AUDIO
#undef HAVE_NEWPCM_AUDIO
#undef HAVE_OSS_AUDIO
#undef HAVE_HP_AUDIO
#undef HAVE_NETBSD_AUDIO
#undef HAVE_OSPREY_AUDIO
#undef HAVE_MACHINE_PCAUDIOIO_H
#undef HAVE_ALSA_AUDIO
#undef HAVE_IXJ_AUDIO

/* GSM related */
#undef SASR
#undef FAST
#undef USE_FLOAT_MUL

/* FireWire and DV */
#undef HAVE_FIREWIRE_DV_FREEBSD
#undef HAVE_DV_CODEC

/*
 * If you don't have these types in <inttypes.h>, #define these to be
 * the types you do have.
 */
#undef int8_t
#undef int16_t
#undef int32_t
#undef int64_t
#undef uint8_t
#undef uint16_t
#undef uint32_t

/*
 * Debugging:
 * DEBUG: general debugging
 * DEBUG_MEM: debug memory allocation
 */
#undef DEBUG
#undef DEBUG_MEM

@BOTTOM@

#ifndef WORDS_BIGENDIAN
#define WORDS_SMALLENDIAN
#endif

