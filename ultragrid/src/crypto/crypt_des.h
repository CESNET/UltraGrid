/*****************************************************************************




Saleem N. Bhatti
February 1993
*****************************************************************************/

#if !defined(_qfDES_h_)
#define      _qfDES_h_

#if defined(__cplusplus)
extern "C" {
#endif

/* what */
typedef enum {qfDES_encrypt, qfDES_decrypt} QFDES_what;

/* mode */
typedef enum {qfDES_ecb, qfDES_cbc, qfDES_cfb, qfDES_ofb} QFDES_mode;

/* parity */
typedef enum {qfDES_even, qfDES_odd} QFDES_parity;

/* key/IV generation */
typedef enum {qfDES_key, qfDES_iv} QFDES_generate;


/* This does it all */
int qfDES (unsigned char *key, unsigned char *data, unsigned int size, const QFDES_what what, const QFDES_mode mode, unsigned char *initVec);

/* Handy macros */
#define qfDES_ECB_e(_key, _data, _size) qfDES(_key, _data, _size, qfDES_encrypt, qfDES_ecb, (unsigned char *) 0)
#define qfDES_ECB_d(_key, _data, _size) qfDES(_key, _data, _size, qfDES_decrypt, qfDES_ecb, (unsigned char *) 0)

#define qfDES_CBC_e(_key, _data, _size, _initVec) qfDES(_key, _data, _size, qfDES_encrypt, qfDES_cbc, _initVec)
#define qfDES_CBC_d(_key, _data, _size, _initVec) qfDES(_key, _data, _size, qfDES_decrypt, qfDES_cbc, _initVec)

#define qfDES_CFB_e(_key, _data, _size, _initVec) qfDES(_key, _data, _size, qfDES_encrypt, qfDES_cfb, _initVec)
#define qfDES_CFB_d(_key, _data, _size, _initVec) qfDES(_key, _data, _size, qfDES_decrypt, qfDES_cfb, _initVec)

#define qfDES_OFB_e(_key, _data, _size, _initVec) qfDES(_key, _data, _size, qfDES_encrypt, qfDES_ofb, _initVec)
#define qfDES_OFB_d(_key, _data, _size, _initVec) qfDES(_key, _data, _size, qfDES_decrypt, qfDES_ofb, _initVec)

/* Padded [m|re]alloc() */
unsigned char    qfDES_setPad (unsigned char pad);

#define qfDES_padSpace() qfDES_setPad((unsigned char) ' ')
#define qfDES_padZero() qfDES_setPad((unsigned char) '\0')

/* The size of text in a qfDES_malloc()ed block */
#define qfDES_plainTextSize(_ptr, _size) (unsigned int) ((_size) - (unsigned int) (_ptr)[(_size) - 1])

/* Keys */
void qfDES_setParity (unsigned char *ptr, unsigned int size, const QFDES_parity parity);
unsigned int qfDES_checkParity (unsigned char *ptr, unsigned int size, const QFDES_parity parity);

unsigned char *qfDES_generate (const QFDES_generate what); /* returns a pointer to static memory */

#define qfDES_generateKey() qfDES_generate(qfDES_key)
#define qfDES_generateIV() qfDES_generate(qfDES_iv)

int qfDES_checkWeakKeys (unsigned char *key);

#if defined(__cplusplus)
}
#endif

#endif /* !_qfDES_h_ */
