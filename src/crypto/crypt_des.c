/*****************************************************************************

Saleem N. Bhatti
February 1993

Patch for Intel/Linux courtesy of Mark Handley & George Pavlou
Added 2 August 1996, Saleem
*****************************************************************************/

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "crypto/crypt_des.h"
#include "crypto/random.h"

typedef unsigned int Word;

#define B00 0x80000000
#define B01 0x40000000
#define B02 0x20000000
#define B03 0x10000000
#define B04 0x08000000
#define B05 0x04000000
#define B06 0x02000000
#define B07 0x01000000
#define B08 0x00800000
#define B09 0x00400000
#define B10 0x00200000
#define B11 0x00100000
#define B12 0x00080000
#define B13 0x00040000
#define B14 0x00020000
#define B15 0x00010000
#define B16 0x00008000
#define B17 0x00004000
#define B18 0x00002000
#define B19 0x00001000
#define B20 0x00000800
#define B21 0x00000400
#define B22 0x00000200
#define B23 0x00000100
#define B24 0x00000080
#define B25 0x00000040
#define B26 0x00000020
#define B27 0x00000010
#define B28 0x00000008
#define B29 0x00000004
#define B30 0x00000002
#define B31 0x00000001

#define INITIAL_PERMUTATION_AUX(_i0, _i1, _o0, _o1) \
{ \
_o0 = _o1 = 0; \
_o0 |= (_i1 & B25) << 25; /* 58 */ \
_o0 |= (_i1 & B17) << 16; /* 50 */ \
_o0 |= (_i1 & B09) <<  7; /* 42 */ \
_o0 |= (_i1 & B01) >>  2; /* 34 */ \
_o0 |= (_i0 & B25) << 21; /* 26 */ \
_o0 |= (_i0 & B17) << 12; /* 18 */ \
_o0 |= (_i0 & B09) <<  3; /* 10 */ \
_o0 |= (_i0 & B01) >>  6; /*  2 */ \
_o0 |= (_i1 & B27) << 19; /* 60 */ \
_o0 |= (_i1 & B19) << 10; /* 52 */ \
_o0 |= (_i1 & B11) <<  1; /* 44 */ \
_o0 |= (_i1 & B03) >>  8; /* 36 */ \
_o0 |= (_i0 & B27) << 15; /* 28 */ \
_o0 |= (_i0 & B19) <<  6; /* 20 */ \
_o0 |= (_i0 & B11) >>  3; /* 12 */ \
_o0 |= (_i0 & B03) >> 12; /*  4 */ \
_o0 |= (_i1 & B29) << 13; /* 62 */ \
_o0 |= (_i1 & B21) <<  4; /* 54 */ \
_o0 |= (_i1 & B13) >>  5; /* 46 */ \
_o0 |= (_i1 & B05) >> 14; /* 38 */ \
_o0 |= (_i0 & B29) <<  9; /* 30 */ \
_o0 |= (_i0 & B21)      ; /* 22 */ \
_o0 |= (_i0 & B13) >>  9; /* 14 */ \
_o0 |= (_i0 & B05) >> 18; /*  6 */ \
_o0 |= (_i1 & B31) <<  7; /* 64 */ \
_o0 |= (_i1 & B23) >>  2; /* 56 */ \
_o0 |= (_i1 & B15) >> 11; /* 48 */ \
_o0 |= (_i1 & B07) >> 20; /* 40 */ \
_o0 |= (_i0 & B31) <<  3; /* 32 */ \
_o0 |= (_i0 & B23) >>  6; /* 24 */ \
_o0 |= (_i0 & B15) >> 15; /* 16 */ \
_o0 |= (_i0 & B07) >> 24; /*  8 */ \
_o1 |= (_i1 & B24) << 24; /* 57 */ \
_o1 |= (_i1 & B16) << 15; /* 49 */ \
_o1 |= (_i1 & B08) <<  6; /* 41 */ \
_o1 |= (_i1 & B00) >>  3; /* 33 */ \
_o1 |= (_i0 & B24) << 20; /* 25 */ \
_o1 |= (_i0 & B16) << 11; /* 17 */ \
_o1 |= (_i0 & B08) <<  2; /*  9 */ \
_o1 |= (_i0 & B00) >>  7; /*  1 */ \
_o1 |= (_i1 & B26) << 18; /* 59 */ \
_o1 |= (_i1 & B18) <<  9; /* 51 */ \
_o1 |= (_i1 & B10)      ; /* 43 */ \
_o1 |= (_i1 & B02) >>  9; /* 35 */ \
_o1 |= (_i0 & B26) << 14; /* 27 */ \
_o1 |= (_i0 & B18) <<  5; /* 19 */ \
_o1 |= (_i0 & B10) >>  4; /* 11 */ \
_o1 |= (_i0 & B02) >> 13; /*  3 */ \
_o1 |= (_i1 & B28) << 12; /* 61 */ \
_o1 |= (_i1 & B20) <<  3; /* 53 */ \
_o1 |= (_i1 & B12) >>  6; /* 45 */ \
_o1 |= (_i1 & B04) >> 15; /* 37 */ \
_o1 |= (_i0 & B28) <<  8; /* 29 */ \
_o1 |= (_i0 & B20) >>  1; /* 21 */ \
_o1 |= (_i0 & B12) >> 10; /* 13 */ \
_o1 |= (_i0 & B04) >> 19; /*  5 */ \
_o1 |= (_i1 & B30) <<  6; /* 63 */ \
_o1 |= (_i1 & B22) >>  3; /* 55 */ \
_o1 |= (_i1 & B14) >> 12; /* 47 */ \
_o1 |= (_i1 & B06) >> 21; /* 39 */ \
_o1 |= (_i0 & B30) <<  2; /* 31 */ \
_o1 |= (_i0 & B22) >>  7; /* 23 */ \
_o1 |= (_i0 & B14) >> 16; /* 15 */ \
_o1 |= (_i0 & B06) >> 25; /*  7 */ \
}

#define FINAL_PERMUTATION_AUX(_i0, _i1, _o0, _o1) \
{ \
_o0 = _o1 = 0; \
_o0 |= (_i1 & B07) <<  7; /* 40 */ \
_o0 |= (_i0 & B07) <<  6; /*  8 */ \
_o0 |= (_i1 & B15) << 13; /* 48 */ \
_o0 |= (_i0 & B15) << 12; /* 16 */ \
_o0 |= (_i1 & B23) << 19; /* 56 */ \
_o0 |= (_i0 & B23) << 18; /* 24 */ \
_o0 |= (_i1 & B31) << 25; /* 64 */ \
_o0 |= (_i0 & B31) << 24; /* 32 */ \
_o0 |= (_i1 & B06) >>  2; /* 39 */ \
_o0 |= (_i0 & B06) >>  3; /*  7 */ \
_o0 |= (_i1 & B14) <<  4; /* 47 */ \
_o0 |= (_i0 & B14) <<  3; /* 15 */ \
_o0 |= (_i1 & B22) << 10; /* 55 */ \
_o0 |= (_i0 & B22) <<  9; /* 23 */ \
_o0 |= (_i1 & B30) << 16; /* 63 */ \
_o0 |= (_i0 & B30) << 15; /* 31 */ \
_o0 |= (_i1 & B05) >> 11; /* 38 */ \
_o0 |= (_i0 & B05) >> 12; /*  6 */ \
_o0 |= (_i1 & B13) >>  5; /* 46 */ \
_o0 |= (_i0 & B13) >>  6; /* 14 */ \
_o0 |= (_i1 & B21) <<  1; /* 54 */ \
_o0 |= (_i0 & B21)      ; /* 22 */ \
_o0 |= (_i1 & B29) <<  7; /* 62 */ \
_o0 |= (_i0 & B29) <<  6; /* 30 */ \
_o0 |= (_i1 & B04) >> 20; /* 37 */ \
_o0 |= (_i0 & B04) >> 21; /*  5 */ \
_o0 |= (_i1 & B12) >> 14; /* 45 */ \
_o0 |= (_i0 & B12) >> 15; /* 13 */ \
_o0 |= (_i1 & B20) >>  8; /* 53 */ \
_o0 |= (_i0 & B20) >>  9; /* 21 */ \
_o0 |= (_i1 & B28) >>  2; /* 61 */ \
_o0 |= (_i0 & B28) >>  3; /* 29 */ \
_o1 |= (_i1 & B03) <<  3; /* 36 */ \
_o1 |= (_i0 & B03) <<  2; /*  4 */ \
_o1 |= (_i1 & B11) <<  9; /* 44 */ \
_o1 |= (_i0 & B11) <<  8; /* 12 */ \
_o1 |= (_i1 & B19) << 15; /* 52 */ \
_o1 |= (_i0 & B19) << 14; /* 20 */ \
_o1 |= (_i1 & B27) << 21; /* 60 */ \
_o1 |= (_i0 & B27) << 20; /* 28 */ \
_o1 |= (_i1 & B02) >>  6; /* 35 */ \
_o1 |= (_i0 & B02) >>  7; /*  3 */ \
_o1 |= (_i1 & B10)      ; /* 43 */ \
_o1 |= (_i0 & B10) >>  1; /* 11 */ \
_o1 |= (_i1 & B18) <<  6; /* 51 */ \
_o1 |= (_i0 & B18) <<  5; /* 19 */ \
_o1 |= (_i1 & B26) << 12; /* 59 */ \
_o1 |= (_i0 & B26) << 11; /* 27 */ \
_o1 |= (_i1 & B01) >> 15; /* 34 */ \
_o1 |= (_i0 & B01) >> 16; /*  2 */ \
_o1 |= (_i1 & B09) >>  9; /* 42 */ \
_o1 |= (_i0 & B09) >> 10; /* 10 */ \
_o1 |= (_i1 & B17) >>  3; /* 50 */ \
_o1 |= (_i0 & B17) >>  4; /* 18 */ \
_o1 |= (_i1 & B25) <<  3; /* 58 */ \
_o1 |= (_i0 & B25) <<  2; /* 26 */ \
_o1 |= (_i1 & B00) >> 24; /* 33 */ \
_o1 |= (_i0 & B00) >> 25; /*  1 */ \
_o1 |= (_i1 & B08) >> 18; /* 41 */ \
_o1 |= (_i0 & B08) >> 19; /*  9 */ \
_o1 |= (_i1 & B16) >> 12; /* 49 */ \
_o1 |= (_i0 & B16) >> 13; /* 17 */ \
_o1 |= (_i1 & B24) >>  6; /* 57 */ \
_o1 |= (_i0 & B24) >>  7; /* 25 */ \
}

/* 64b -> 2x28b */
#define PC1_AUX(_i0, _i1, _o0, _o1) \
{ \
_o0 = _o1 = 0; \
_o0 |= (_i1 & B24) << 24; /* 57 */ \
_o0 |= (_i1 & B16) << 15; /* 49 */ \
_o0 |= (_i1 & B08) <<  6; /* 41 */ \
_o0 |= (_i1 & B00) >>  3; /* 33 */ \
_o0 |= (_i0 & B24) << 20; /* 25 */ \
_o0 |= (_i0 & B16) << 11; /* 17 */ \
_o0 |= (_i0 & B08) <<  2; /*  9 */ \
_o0 |= (_i0 & B00) >>  7; /*  1 */ \
_o0 |= (_i1 & B25) << 17; /* 58 */ \
_o0 |= (_i1 & B17) <<  8; /* 50 */ \
_o0 |= (_i1 & B09) >>  1; /* 42 */ \
_o0 |= (_i1 & B01) >> 10; /* 34 */ \
_o0 |= (_i0 & B25) << 13; /* 26 */ \
_o0 |= (_i0 & B17) <<  4; /* 18 */ \
_o0 |= (_i0 & B09) >>  5; /* 10 */ \
_o0 |= (_i0 & B01) >> 14; /*  2 */ \
_o0 |= (_i1 & B26) << 10; /* 59 */ \
_o0 |= (_i1 & B18) <<  1; /* 51 */ \
_o0 |= (_i1 & B10) >>  8; /* 43 */ \
_o0 |= (_i1 & B02) >> 17; /* 35 */ \
_o0 |= (_i0 & B26) <<  6; /* 27 */ \
_o0 |= (_i0 & B18) >>  3; /* 19 */ \
_o0 |= (_i0 & B10) >> 12; /* 11 */ \
_o0 |= (_i0 & B02) >> 21; /*  3 */ \
_o0 |= (_i1 & B27) <<  3; /* 60 */ \
_o0 |= (_i1 & B19) >>  6; /* 52 */ \
_o0 |= (_i1 & B11) >> 15; /* 44 */ \
_o0 |= (_i1 & B03) >> 24; /* 36 */ \
_o1 |= (_i1 & B30) << 30; /* 63 */ \
_o1 |= (_i1 & B22) << 21; /* 55 */ \
_o1 |= (_i1 & B14) << 12; /* 47 */ \
_o1 |= (_i1 & B06) <<  3; /* 39 */ \
_o1 |= (_i0 & B30) << 26; /* 31 */ \
_o1 |= (_i0 & B22) << 17; /* 23 */ \
_o1 |= (_i0 & B14) <<  8; /* 15 */ \
_o1 |= (_i0 & B06) >>  1; /*  7 */ \
_o1 |= (_i1 & B29) << 21; /* 62 */ \
_o1 |= (_i1 & B21) << 12; /* 54 */ \
_o1 |= (_i1 & B13) <<  3; /* 46 */ \
_o1 |= (_i1 & B05) >>  6; /* 38 */ \
_o1 |= (_i0 & B29) << 17; /* 30 */ \
_o1 |= (_i0 & B21) <<  8; /* 22 */ \
_o1 |= (_i0 & B13) >>  1; /* 14 */ \
_o1 |= (_i0 & B05) >> 10; /*  6 */ \
_o1 |= (_i1 & B28) << 12; /* 61 */ \
_o1 |= (_i1 & B20) <<  3; /* 53 */ \
_o1 |= (_i1 & B12) >>  6; /* 45 */ \
_o1 |= (_i1 & B04) >> 15; /* 37 */ \
_o1 |= (_i0 & B28) <<  8; /* 29 */ \
_o1 |= (_i0 & B20) >>  1; /* 21 */ \
_o1 |= (_i0 & B12) >> 10; /* 13 */ \
_o1 |= (_i0 & B04) >> 19; /*  5 */ \
_o1 |= (_i0 & B27) <<  3; /* 28 */ \
_o1 |= (_i0 & B19) >>  6; /* 20 */ \
_o1 |= (_i0 & B11) >> 15; /* 12 */ \
_o1 |= (_i0 & B03) >> 24; /*  4 */ \
}

/* 2x28b -> 8x6b */
#define PC2_AUX(_i0, _i1, _o0, _o1) \
{ \
_o0 = _o1 = 0; \
_o0 |= (_i0 & B13) << 11; /* 14 */ \
_o0 |= (_i0 & B16) << 13; /* 17 */ \
_o0 |= (_i0 & B10) <<  6; /* 11 */ \
_o0 |= (_i0 & B23) << 18; /* 24 */ \
_o0 |= (_i0 & B00) >>  6; /*  1 */ \
_o0 |= (_i0 & B04) >>  3; /*  5 */ \
_o0 |= (_i0 & B02) >>  8; /*  3 */ \
_o0 |= (_i0 & B27) << 16; /* 28 */ \
_o0 |= (_i0 & B14) <<  2; /* 15 */ \
_o0 |= (_i0 & B05) >>  8; /*  6 */ \
_o0 |= (_i0 & B20) <<  6; /* 21 */ \
_o0 |= (_i0 & B09) >>  6; /* 10 */ \
_o0 |= (_i0 & B22) <<  4; /* 23 */ \
_o0 |= (_i0 & B18) >>  1; /* 19 */ \
_o0 |= (_i0 & B11) >>  9; /* 12 */ \
_o0 |= (_i0 & B03) >> 18; /*  4 */ \
_o0 |= (_i0 & B25) <<  3; /* 26 */ \
_o0 |= (_i0 & B07) >> 16; /*  8 */ \
_o0 |= (_i0 & B15) >> 11; /* 16 */ \
_o0 |= (_i0 & B06) >> 21; /*  7 */ \
_o0 |= (_i0 & B26) >>  2; /* 27 */ \
_o0 |= (_i0 & B19) >> 10; /* 20 */ \
_o0 |= (_i0 & B12) >> 18; /* 13 */ \
_o0 |= (_i0 & B01) >> 30; /*  2 */ \
_o1 |= (_i1 & B12) << 10; /* 41 */ \
_o1 |= (_i1 & B23) << 20; /* 52 */ \
_o1 |= (_i1 & B02) >>  2; /* 31 */ \
_o1 |= (_i1 & B08) <<  3; /* 37 */ \
_o1 |= (_i1 & B18) << 12; /* 47 */ \
_o1 |= (_i1 & B26) << 19; /* 55 */ \
_o1 |= (_i1 & B01) >>  9; /* 30 */ \
_o1 |= (_i1 & B11)      ; /* 40 */ \
_o1 |= (_i1 & B22) << 10; /* 51 */ \
_o1 |= (_i1 & B16) <<  3; /* 45 */ \
_o1 |= (_i1 & B04) >> 10; /* 33 */ \
_o1 |= (_i1 & B19) <<  4; /* 48 */ \
_o1 |= (_i1 & B15) >>  3; /* 44 */ \
_o1 |= (_i1 & B20) <<  1; /* 49 */ \
_o1 |= (_i1 & B10) >> 10; /* 39 */ \
_o1 |= (_i1 & B27) <<  6; /* 56 */ \
_o1 |= (_i1 & B05) >> 17; /* 34 */ \
_o1 |= (_i1 & B24) <<  1; /* 53 */ \
_o1 |= (_i1 & B17) >>  9; /* 46 */ \
_o1 |= (_i1 & B13) >> 14; /* 42 */ \
_o1 |= (_i1 & B21) >>  7; /* 50 */ \
_o1 |= (_i1 & B07) >> 22; /* 36 */ \
_o1 |= (_i1 & B00) >> 30; /* 29 */ \
_o1 |= (_i1 & B03) >> 28; /* 32 */ \
}

static
Word s_p0[64] = {               /* Combined S-Box1 and permutation P */
        0x00808200, 0x00000000, 0x00008000, 0x00808202,
        0x00808002, 0x00008202, 0x00000002, 0x00008000,
        0x00000200, 0x00808200, 0x00808202, 0x00000200,
        0x00800202, 0x00808002, 0x00800000, 0x00000002,
        0x00000202, 0x00800200, 0x00800200, 0x00008200,
        0x00008200, 0x00808000, 0x00808000, 0x00800202,
        0x00008002, 0x00800002, 0x00800002, 0x00008002,
        0x00000000, 0x00000202, 0x00008202, 0x00800000,
        0x00008000, 0x00808202, 0x00000002, 0x00808000,
        0x00808200, 0x00800000, 0x00800000, 0x00000200,
        0x00808002, 0x00008000, 0x00008200, 0x00800002,
        0x00000200, 0x00000002, 0x00800202, 0x00008202,
        0x00808202, 0x00008002, 0x00808000, 0x00800202,
        0x00800002, 0x00000202, 0x00008202, 0x00808200,
        0x00000202, 0x00800200, 0x00800200, 0x00000000,
        0x00008002, 0x00008200, 0x00000000, 0x00808002
};

static
Word s_p1[64] = {               /* Combined S-Box2 and permutation P */
        0x40084010, 0x40004000, 0x00004000, 0x00084010,
        0x00080000, 0x00000010, 0x40080010, 0x40004010,
        0x40000010, 0x40084010, 0x40084000, 0x40000000,
        0x40004000, 0x00080000, 0x00000010, 0x40080010,
        0x00084000, 0x00080010, 0x40004010, 0x00000000,
        0x40000000, 0x00004000, 0x00084010, 0x40080000,
        0x00080010, 0x40000010, 0x00000000, 0x00084000,
        0x00004010, 0x40084000, 0x40080000, 0x00004010,
        0x00000000, 0x00084010, 0x40080010, 0x00080000,
        0x40004010, 0x40080000, 0x40084000, 0x00004000,
        0x40080000, 0x40004000, 0x00000010, 0x40084010,
        0x00084010, 0x00000010, 0x00004000, 0x40000000,
        0x00004010, 0x40084000, 0x00080000, 0x40000010,
        0x00080010, 0x40004010, 0x40000010, 0x00080010,
        0x00084000, 0x00000000, 0x40004000, 0x00004010,
        0x40000000, 0x40080010, 0x40084010, 0x00084000
};

static
Word s_p2[64] = {               /* Combined S-Box3 and permutation P */
        0x00000104, 0x04010100, 0x00000000, 0x04010004,
        0x04000100, 0x00000000, 0x00010104, 0x04000100,
        0x00010004, 0x04000004, 0x04000004, 0x00010000,
        0x04010104, 0x00010004, 0x04010000, 0x00000104,
        0x04000000, 0x00000004, 0x04010100, 0x00000100,
        0x00010100, 0x04010000, 0x04010004, 0x00010104,
        0x04000104, 0x00010100, 0x00010000, 0x04000104,
        0x00000004, 0x04010104, 0x00000100, 0x04000000,
        0x04010100, 0x04000000, 0x00010004, 0x00000104,
        0x00010000, 0x04010100, 0x04000100, 0x00000000,
        0x00000100, 0x00010004, 0x04010104, 0x04000100,
        0x04000004, 0x00000100, 0x00000000, 0x04010004,
        0x04000104, 0x00010000, 0x04000000, 0x04010104,
        0x00000004, 0x00010104, 0x00010100, 0x04000004,
        0x04010000, 0x04000104, 0x00000104, 0x04010000,
        0x00010104, 0x00000004, 0x04010004, 0x00010100
};

static
Word s_p3[64] = {               /* Combined S-Box4 and permutation P */
        0x80401000, 0x80001040, 0x80001040, 0x00000040,
        0x00401040, 0x80400040, 0x80400000, 0x80001000,
        0x00000000, 0x00401000, 0x00401000, 0x80401040,
        0x80000040, 0x00000000, 0x00400040, 0x80400000,
        0x80000000, 0x00001000, 0x00400000, 0x80401000,
        0x00000040, 0x00400000, 0x80001000, 0x00001040,
        0x80400040, 0x80000000, 0x00001040, 0x00400040,
        0x00001000, 0x00401040, 0x80401040, 0x80000040,
        0x00400040, 0x80400000, 0x00401000, 0x80401040,
        0x80000040, 0x00000000, 0x00000000, 0x00401000,
        0x00001040, 0x00400040, 0x80400040, 0x80000000,
        0x80401000, 0x80001040, 0x80001040, 0x00000040,
        0x80401040, 0x80000040, 0x80000000, 0x00001000,
        0x80400000, 0x80001000, 0x00401040, 0x80400040,
        0x80001000, 0x00001040, 0x00400000, 0x80401000,
        0x00000040, 0x00400000, 0x00001000, 0x00401040
};

static
Word s_p4[64] = {               /* Combined S-Box5 and permutation P */
        0x00000080, 0x01040080, 0x01040000, 0x21000080,
        0x00040000, 0x00000080, 0x20000000, 0x01040000,
        0x20040080, 0x00040000, 0x01000080, 0x20040080,
        0x21000080, 0x21040000, 0x00040080, 0x20000000,
        0x01000000, 0x20040000, 0x20040000, 0x00000000,
        0x20000080, 0x21040080, 0x21040080, 0x01000080,
        0x21040000, 0x20000080, 0x00000000, 0x21000000,
        0x01040080, 0x01000000, 0x21000000, 0x00040080,
        0x00040000, 0x21000080, 0x00000080, 0x01000000,
        0x20000000, 0x01040000, 0x21000080, 0x20040080,
        0x01000080, 0x20000000, 0x21040000, 0x01040080,
        0x20040080, 0x00000080, 0x01000000, 0x21040000,
        0x21040080, 0x00040080, 0x21000000, 0x21040080,
        0x01040000, 0x00000000, 0x20040000, 0x21000000,
        0x00040080, 0x01000080, 0x20000080, 0x00040000,
        0x00000000, 0x20040000, 0x01040080, 0x20000080
};

static
Word s_p5[64] = {               /* Combined S-Box6 and permutation P */
        0x10000008, 0x10200000, 0x00002000, 0x10202008,
        0x10200000, 0x00000008, 0x10202008, 0x00200000,
        0x10002000, 0x00202008, 0x00200000, 0x10000008,
        0x00200008, 0x10002000, 0x10000000, 0x00002008,
        0x00000000, 0x00200008, 0x10002008, 0x00002000,
        0x00202000, 0x10002008, 0x00000008, 0x10200008,
        0x10200008, 0x00000000, 0x00202008, 0x10202000,
        0x00002008, 0x00202000, 0x10202000, 0x10000000,
        0x10002000, 0x00000008, 0x10200008, 0x00202000,
        0x10202008, 0x00200000, 0x00002008, 0x10000008,
        0x00200000, 0x10002000, 0x10000000, 0x00002008,
        0x10000008, 0x10202008, 0x00202000, 0x10200000,
        0x00202008, 0x10202000, 0x00000000, 0x10200008,
        0x00000008, 0x00002000, 0x10200000, 0x00202008,
        0x00002000, 0x00200008, 0x10002008, 0x00000000,
        0x10202000, 0x10000000, 0x00200008, 0x10002008
};

static
Word s_p6[64] = {               /* Combined S-Box7 and permutation P */
        0x00100000, 0x02100001, 0x02000401, 0x00000000,
        0x00000400, 0x02000401, 0x00100401, 0x02100400,
        0x02100401, 0x00100000, 0x00000000, 0x02000001,
        0x00000001, 0x02000000, 0x02100001, 0x00000401,
        0x02000400, 0x00100401, 0x00100001, 0x02000400,
        0x02000001, 0x02100000, 0x02100400, 0x00100001,
        0x02100000, 0x00000400, 0x00000401, 0x02100401,
        0x00100400, 0x00000001, 0x02000000, 0x00100400,
        0x02000000, 0x00100400, 0x00100000, 0x02000401,
        0x02000401, 0x02100001, 0x02100001, 0x00000001,
        0x00100001, 0x02000000, 0x02000400, 0x00100000,
        0x02100400, 0x00000401, 0x00100401, 0x02100400,
        0x00000401, 0x02000001, 0x02100401, 0x02100000,
        0x00100400, 0x00000000, 0x00000001, 0x02100401,
        0x00000000, 0x00100401, 0x02100000, 0x00000400,
        0x02000001, 0x02000400, 0x00000400, 0x00100001
};

static
Word s_p7[64] = {               /* Combined S-Box8 and permutation P */
        0x08000820, 0x00000800, 0x00020000, 0x08020820,
        0x08000000, 0x08000820, 0x00000020, 0x08000000,
        0x00020020, 0x08020000, 0x08020820, 0x00020800,
        0x08020800, 0x00020820, 0x00000800, 0x00000020,
        0x08020000, 0x08000020, 0x08000800, 0x00000820,
        0x00020800, 0x00020020, 0x08020020, 0x08020800,
        0x00000820, 0x00000000, 0x00000000, 0x08020020,
        0x08000020, 0x08000800, 0x00020820, 0x00020000,
        0x00020820, 0x00020000, 0x08020800, 0x00000800,
        0x00000020, 0x08020020, 0x00000800, 0x00020820,
        0x08000800, 0x00000020, 0x08000020, 0x08020000,
        0x08020020, 0x08000000, 0x00020000, 0x08000820,
        0x00000000, 0x08020820, 0x00020020, 0x08000020,
        0x08020000, 0x08000800, 0x08000820, 0x00000000,
        0x08020820, 0x00020800, 0x00020800, 0x00000820,
        0x00000820, 0x00020020, 0x08000000, 0x08020800
};

#define INITIAL_PERMUTATION(t, regL, regR) \
        INITIAL_PERMUTATION_AUX(t[0], t[1], regL, regR)

#define FINAL_PERMUTATION(regR, regL, t) \
        FINAL_PERMUTATION_AUX(regR, regL, t[0], t[1])

#define PC1(k, regC, regD) \
        PC1_AUX(k[0], k[1], regC, regD)

#define PC2(regC, regD, k) \
        PC2_AUX(regC, regD, k[0], k[1])

unsigned char G_padChar = (char)0;      /* Default pad charcater */

static Word ROTATE_LEFT(Word x)
{
        Word a;
        a = (x & 0x80000000) >> 27;
        return (x << 1) | a;
}

static Word ROTATE_RIGHT(Word x)
{
        Word a;
        a = x & 0x00000010;
        return (x >> 1) | (a << 27);
}

/*
** The S Box transformations and the permutation P are combined in the vectors
** s_p0 - s_p7. Permutation E and the MOD 2 addition with the intermediate key
** are then done "inline" on each round. The intermediate key is already in a
** a 8x6bit form because of the modified permuation PC2 used.
*/

#if !defined(WORDS_BIGENDIAN)

#define DES(t, ik) \
{ \
    register Word l, r, reg32, round; \
    register unsigned char *bb; \
    INITIAL_PERMUTATION(t, l, r); \
    for(bb = (unsigned char *) ik, round = 0x8000; round; bb += 8, round >>= 1) { \
        register Word w = (r << 1) | (r >> 31); \
        reg32  = s_p7[( w        & 0x3f) ^ bb[4]]; \
        reg32 |= s_p6[((w >>= 4) & 0x3f) ^ bb[5]]; \
        reg32 |= s_p5[((w >>= 4) & 0x3f) ^ bb[6]]; \
        reg32 |= s_p4[((w >>= 4) & 0x3f) ^ bb[7]]; \
        reg32 |= s_p3[((w >>= 4) & 0x3f) ^ bb[0]]; \
        reg32 |= s_p2[((w >>= 4) & 0x3f) ^ bb[1]]; \
        reg32 |= s_p1[((w >>  4) & 0x3f) ^ bb[2]]; \
        reg32 |= s_p0[(((r & 0x1) << 5) | ((r & 0xf8000000) >> 27)) ^ bb[3]]; \
        reg32 ^= l; \
        l = r; \
        r = reg32; \
    } \
    FINAL_PERMUTATION(r, l, t); \
}

#define MAKE_LITTLE_ENDIAN(t, s) \
{ \
    register unsigned int z, l = s/4; \
    register Word *tp = (Word *) t; \
    for(z = 0; z < l; ++z) tp[z] = htonl(tp[z]); \
}

#else                           /* WORDS_BIGENDIAN */

#define DES(t, ik) \
{ \
    register Word l, r, reg32, round; \
    register unsigned char *bb; \
    INITIAL_PERMUTATION(t, l, r); \
    for(bb = (unsigned char *) ik, round = 0x8000; round; bb += 8, round >>= 1) { \
        register Word w = (r << 1) | (r >> 31); \
        reg32  = s_p7[( w        & 0x3f) ^ bb[7]]; \
        reg32 |= s_p6[((w >>= 4) & 0x3f) ^ bb[6]]; \
        reg32 |= s_p5[((w >>= 4) & 0x3f) ^ bb[5]]; \
        reg32 |= s_p4[((w >>= 4) & 0x3f) ^ bb[4]]; \
        reg32 |= s_p3[((w >>= 4) & 0x3f) ^ bb[3]]; \
        reg32 |= s_p2[((w >>= 4) & 0x3f) ^ bb[2]]; \
        reg32 |= s_p1[((w >>  4) & 0x3f) ^ bb[1]]; \
        reg32 |= s_p0[(((r & 0x1) << 5) | ((r & 0xf8000000) >> 27)) ^ bb[0]]; \
        reg32 ^= l; \
        l = r; \
        r = reg32; \
    } \
    FINAL_PERMUTATION(r, l, t); \
}

#endif                          /* WORDS_BIGENDIAN */

int
qfDES(unsigned char *key,
      unsigned char *data,
      unsigned int size,
      const QFDES_what what, const QFDES_mode mode, unsigned char *initVec)
{
        /* Store some info to optimise for multiple calls ... */
        static unsigned char desKey[8], desKeys[128];
        static Word *oldKey = (Word *) desKey, *keys = (Word *) desKeys;
        static QFDES_what oldWhat;
        static QFDES_mode oldMode;
        unsigned char b0[8], b1[8];     /* feedback blocks */
        Word *newKey = (Word *) key,    /* key from user */
            *text,              /* text to be [en|de]crypted */
            *cb = (Word *) b0,  /* the chained block in CBC mode */
            *cb1 = (Word *) b1; /* a copy for use when decrypting */

#if !defined(WORDS_BIGENDIAN)
        unsigned int origSize = size;
        MAKE_LITTLE_ENDIAN(key, 8);
        MAKE_LITTLE_ENDIAN(data, origSize);
#endif

        /*
         ** Check new key against old key
         ** and set up intermediate keys.
         */
        if (newKey[0] != oldKey[0] || newKey[1] != oldKey[1]) {
                Word c, d;      /* C and D registers */

                oldKey[0] = newKey[0];
                oldKey[1] = newKey[1];
                oldWhat = what;
                oldMode = mode;

                PC1(newKey, c, d);

                if ((what == qfDES_encrypt) || (mode == qfDES_cfb)
                    || (mode == qfDES_ofb)) {
                        int z;
                        Word r;
                        Word *k = keys;
                        Word rol[16] =
                            { 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1 };

                        for (z = 0; z < 16; z++, k += 2) {
                                for (r = 0; r < rol[z]; r++) {
                                        c = ROTATE_LEFT(c);
                                        d = ROTATE_LEFT(d);
                                }
                                PC2(c, d, k);
                        }
                } else {
                        int z;
                        Word r;
                        Word *k = keys;
                        Word ror[16] =
                            { 0, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1 };

                        for (z = 0; z < 16; z++, k += 2) {
                                r = 0;
                                while (ror[z] > r) {
                                        r++;
                                        c = ROTATE_RIGHT(c);
                                        d = ROTATE_RIGHT(d);
                                }
                                PC2(c, d, k);
                        }
                }
        } else if ((what != oldWhat)
                   && ((mode == qfDES_ecb) || (mode == qfDES_cbc))) {
                /*
                 ** Same key but different direction.
                 ** Reverse intermediate key sequence (ECB and CBC).
                 */
                Word *ik1, *ik2, ik3[2];

                for (ik1 = keys, ik2 = &(keys[30]); ik1 < ik2;
                     ik1 += 2, ik2 -= 2) {
                        ik3[0] = ik1[0];
                        ik3[1] = ik1[1];
                        ik1[0] = ik2[0];
                        ik1[1] = ik2[1];
                        ik2[0] = ik3[0];
                        ik2[1] = ik3[1];
                }

                oldWhat = what;
        }

        /* Set up initilaisation vector */
        if (mode != qfDES_ecb) {
                if (initVec) {
                        {
                                cb[0] = ((Word *) initVec)[0];
                                cb[1] = ((Word *) initVec)[1];
                        }
                } else {
                        cb[0] = 0;
                        cb[1] = 0;
                }
#if !defined(WORDS_BIGENDIAN)
                MAKE_LITTLE_ENDIAN(cb, 8);
#endif
        }

        /*
         ** Lots of gotos and code duplication follow (arrgh) but it speeds
         ** it up a wee bit!
         ** What would be more useful is looking more carefully at the DES
         ** permutations to produce more efficient versions of the macros
         ** of the "auto-generated" versions used in qfDES-aux.c.
         */

        size >>= 3;             /* this is always a multpile of 8 */

        if (what == qfDES_encrypt) {
                switch ((int)mode) {
                case ((int)qfDES_ecb):
                        goto _ECB_;
                case ((int)qfDES_cbc):
                        goto _CBC_encrypt_;
                case ((int)qfDES_cfb):
                        goto _CFB_encrypt_;
                case ((int)qfDES_ofb):
                        goto _OFB_;
                }
        } else {
                switch ((int)mode) {
                case ((int)qfDES_ecb):
                        goto _ECB_;
                case ((int)qfDES_cbc):
                        goto _CBC_decrypt_;
                case ((int)qfDES_cfb):
                        goto _CFB_decrypt_;
                case ((int)qfDES_ofb):
                        goto _OFB_;
                }
        }

 _ECB_:

        /* ECB */
        for (text = (Word *) data; size; --size, text += 2) {
                DES(text, keys);
        }

        goto _exit_qfDES_;

 _CBC_encrypt_:

        /* CBC Encryption */
        for (text = (Word *) data; size; --size, text += 2) {

                /* chaining block */
                text[0] ^= cb[0];
                text[1] ^= cb[1];

                DES(text, keys);

                /* set up chaining block for next round */
                cb[0] = text[0];
                cb[1] = text[1];
        }

        goto _initVec_;

 _CBC_decrypt_:

        /* CBC Decryption */
        for (text = (Word *) data; size; --size, text += 2) {

                /* set up chaining block */
                /*
                 ** The decryption is done in place so I need
                 ** to copy this text block for the next round.
                 */
                cb1[0] = text[0];
                cb1[1] = text[1];

                DES(text, keys);

                /* chaining block for next round */
                text[0] ^= cb[0];
                text[1] ^= cb[1];

                /*
                 ** Copy back the saved encrypted text - this makes
                 ** CBC decryption slower than CBC encryption.
                 */
                cb[0] = cb1[0];
                cb[1] = cb1[1];
        }

        goto _initVec_;

 _CFB_encrypt_:

        /* CFB Encryption */
        for (text = (Word *) data; size; --size, text += 2) {

                /* use cb as the feedback block */
                DES(cb, keys);

                text[0] ^= cb[0];
                text[1] ^= cb[1];

                /* set up feedback block for next round */
                cb[0] = text[0];
                cb[1] = text[1];
        }

        goto _initVec_;

 _CFB_decrypt_:

        /* CFB Decryption */
        for (text = (Word *) data; size; --size, text += 2) {

                /* set up feedback block */
                /*
                 ** The decryption is done in place so I need
                 ** to copy this text block for the next round.
                 */
                cb1[0] = text[0];
                cb1[1] = text[1];

                /* use cb as the feedback block */
                DES(cb, keys);

                text[0] ^= cb[0];
                text[1] ^= cb[1];

                /* set up feedback block for next round */
                cb[0] = cb1[0];
                cb[1] = cb1[1];
        }

        goto _initVec_;

 _OFB_:

        /* OFB */
        for (text = (Word *) data; size; --size, text += 2) {

                /* use cb as the feed back block */
                DES(cb, keys);

                text[0] ^= cb[0];
                text[1] ^= cb[1];
        }

 _initVec_:

        /*
         ** Copy the final chained block back to initVec (CBC, CFB and OFB).
         ** This allows the [en|de]cryption of large amounts of data in small
         ** chunks.
         */
        if (initVec) {
                ((Word *) initVec)[0] = cb[0];
                ((Word *) initVec)[1] = cb[1];

#if !defined(WORDS_BIGENDIAN)
                MAKE_LITTLE_ENDIAN(initVec, 8);
#endif
        }

 _exit_qfDES_:

#if !defined(WORDS_BIGENDIAN)
        MAKE_LITTLE_ENDIAN(key, 8);
        MAKE_LITTLE_ENDIAN(data, origSize);
#endif

        return 0;
}

/*
** This function sets bit 8 of each byte to odd or even parity as requested.
** It is assumed that the right-most bit of each byte is the parity bit.
** Although this is really only used by the two key generation functions below,
** it may be useful to someone.
*/

void qfDES_setParity(unsigned char *ptr, unsigned int size,
                     const QFDES_parity parity)
{
        unsigned int i, mask, bits;

        for (i = 0; i < size; ++i, ++ptr) {
                for (mask = 0x80, bits = 0; mask > 0x01; mask >>= 1)
                        if (((unsigned int)*ptr) & mask)
                                ++bits;

                *ptr |= bits % 2 == (unsigned int)parity ? 0x00 : 0x01;
        }
}

unsigned int qfDES_checkParity(unsigned char *ptr, unsigned int size,
                               const QFDES_parity parity)
{
        unsigned int i, mask, bits, parityBit, parityErrors = 0;

        for (i = 0; i < size; ++i, ++ptr) {
                for (mask = 0x80, bits = 0; mask > 0x01; mask >>= 1)
                        if (((unsigned int)*ptr) & mask)
                                ++bits;

                parityBit = bits % 2 == (unsigned int)parity ? 0 : 1;

                if ((((unsigned int)*ptr) & 0x1) != parityBit)
                        ++parityErrors;
        }

        return parityErrors;
}

static
unsigned char weakKeys[18][8] =
    { {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11},
{0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01},
{0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe},
{0x1f, 0x1f, 0x1f, 0x1f, 0x0e, 0x0e, 0x0e, 0x0e},
{0xe0, 0xe0, 0xe0, 0xe0, 0xf1, 0xf1, 0xf1, 0xf1},
{0x01, 0xfe, 0x01, 0xfe, 0x01, 0xfe, 0x01, 0xfe},
{0xfe, 0x01, 0xfe, 0x01, 0xfe, 0x01, 0xfe, 0x01},
{0x1f, 0xe0, 0x1f, 0xe0, 0x0e, 0xf1, 0x0e, 0xf1},
{0xe0, 0x1f, 0xe0, 0x1f, 0xf1, 0x0e, 0xf1, 0x0e},
{0x01, 0xe0, 0x01, 0xe0, 0x01, 0xf1, 0x01, 0xf1},
{0xe0, 0x01, 0xe0, 0x01, 0xf1, 0x01, 0xf1, 0x01},
{0x1f, 0xfe, 0x1f, 0xfe, 0x0e, 0xfe, 0x0e, 0xfe},
{0xfe, 0x1f, 0xfe, 0x1f, 0xfe, 0x0e, 0xfe, 0x0e},
{0x01, 0x1f, 0x01, 0x1f, 0x01, 0x0e, 0x01, 0x0e},
{0x1f, 0x01, 0x1f, 0x01, 0x0e, 0x01, 0x0e, 0x01},
{0xe0, 0xfe, 0xe0, 0xfe, 0xf1, 0xfe, 0xf1, 0xfe},
{0xfe, 0xe0, 0xfe, 0xe0, 0xfe, 0xf1, 0xfe, 0xf1}
};

/*
** Although this is really only used by the key generation function below,
** it may be handy to someone.
*/

int qfDES_checkWeakKeys(unsigned char *key)
{
        unsigned char *bp;
        int i;

        for (bp = weakKeys[i = 0]; i < 18; bp = weakKeys[++i])
                if (memcmp((void *)key, (void *)bp, 8) == 0)
                        return -1;

        return 0;
}

/*
** The following function attempts to genreate a random key or IV.
** It relies on the randomness of the  of the random(3) function. Although
** it is probably not particularly fast, keys and IV will most probably be
** generated off-line so it does not matter too much.
*/

unsigned char *qfDES_generate(const QFDES_generate what)
{
        static
        unsigned char buffer[8];
        static
        int flag = 0;

        unsigned char *bp;
        int mask = what == qfDES_key ? 0xfe : 0xff;

        /* Set up a seed - 42 is the answer ... */
        if (!flag) {
                lbl_srandom((int)(getpid() * 42) ^ (int)time((time_t *) 0));
                flag = 1;
        }
        do {

                for (bp = buffer; bp <= &(buffer[7]);
                     *bp++ = (unsigned char)(lbl_random() & mask)) ;

                if (what == qfDES_key)
                        qfDES_setParity(buffer, 8, qfDES_odd);

        } while (what == qfDES_key ? qfDES_checkWeakKeys(buffer) : 0);

        return buffer;
}

unsigned char qfDES_setPad(unsigned char pad)
{
        unsigned char b = G_padChar;
        G_padChar = pad;
        return b;
}
