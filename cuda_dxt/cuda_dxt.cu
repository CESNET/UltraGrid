///
///  @file    cuda_dxt.cu   
///  @author  Martin Jirman <jirman@cesnet.cz>
///  @brief   CUDA implementation of DXT compression
///

#include <stdio.h>
#include "cuda_dxt.h"


typedef unsigned int u32;
typedef u32 UINT;

struct vec3;
struct uvec2;

struct uvec3 {
    u32 r, g, b;
    __device__ uvec3(u32 r, u32 g, u32 b) : r(r), g(g), b(b) {}
    __device__ uvec3(u32 n = 0) : r(n), g(n), b(n) {}
    __device__ uvec3(const vec3 & v);
};

struct vec2 {
    float r, g;
    __device__ vec2(float x, float y) : r(x), g(y) {}
    __device__ vec2(float n = 0.0f) : r(n), g(n) {}
    __device__ vec2(const vec3 & v);
    __device__ vec2(const uvec2 & v);
    __device__ vec2 operator+(const vec2 & o) const {
        return vec2(r + o.r, g + o.g);
    }
    __device__ vec2 operator*(const vec2 & v) const {
        return vec2(r * v.r, g * v.g);
    }
    __device__ vec2 operator*(const float n) const {
        return *this * vec2(n);
    }
    __device__ vec2 operator-(const vec2 & o) const {
        return *this + (o * -1.0f);
    }
    __device__ vec2 operator/(const float n) const {
        return *this * (1.0f / n);
    }
};

struct uvec2 {
    u32 r, g;
    __device__ uvec2(u32 x, u32 y) : r(x), g(y) {}
    __device__ uvec2(u32 n = 0) : r(n), g(n) {}
    __device__ uvec2(const vec2 & v);
};

struct vec3 : public vec2 {
    float b;
    __device__ vec3(float x, float y, float z) : vec2(x, y), b(z) {}
    __device__ vec3(float n = 0.0f) : vec2(n), b(n) {}
    __device__ vec3(const vec2 & v) : vec2(v), b(0.0f) {}
    __device__ vec3(const uvec3 & v) : vec2(v.r, v.g), b(v.b) {}
    __device__ vec3 operator+(const vec3 & o) const {
        return vec3(r + o.r, g + o.g, b + o.b);
    }
    __device__ vec3 operator*(const vec3 & v) const {
        return vec3(r * v.r, g * v.g, b * v.b);
    }
    __device__ vec3 operator*(const float n) const {
        return *this * vec3(n);
    }
    __device__ vec3 operator-(const vec3 & o) const {
        return *this + (o * -1.0f);
    }
    __device__ vec3 operator/(const float n) const {
        return *this * (1.0f / n);
    }
//     __device__ vec2 & yz() {
//         return *this;
//     }
    __device__ vec2 gb() const {
        return vec2(g, b);
    }
};

uvec3::uvec3(const vec3 & v) : r(v.r), g(v.g), b(v.b) {}
uvec2::uvec2(const vec2 & v) : r(v.r), g(v.g) {}
vec2::vec2(const vec3 & v) : r(v.r), g(v.g) {}
vec2::vec2(const uvec2 & v) : r(v.r), g(v.g) {}


__device__ static vec3 min(const vec3 & a, const vec3 & b) {
    return vec3(min(a.r, b.r), min(a.g, b.g), min(a.b, b.b));
}

__device__ static vec3 max(const vec3 & a, const vec3 & b) {
    return vec3(max(a.r, b.r), max(a.g, b.g), max(a.b, b.b));
}

__device__ static vec2 min(const vec2 & a, const vec2 & b) {
    return vec2(min(a.r, b.r), min(a.g, b.g));
}

__device__ static vec2 max(const vec2 & a, const vec2 & b) {
    return vec2(max(a.r, b.r), max(a.g, b.g));
}

__device__ static float dot(const vec3 & a, const vec3 & b) {
    return a.r * b.r + a.g * b.g + a.b * b.b;
}

__device__ static vec2 clamp(const vec2 & v, float min_val, float max_val) {
    return min(vec2(max_val), max(vec2(min_val), v));
}

__device__ static float clamp(const float & v, float min_val, float max_val) {
    return min(max_val, max(min_val, v));
}

__device__ static vec2 abs(const vec2 & v) {
    return vec2(fabsf(v.r), fabsf(v.g));
}

__device__ static vec3 round(const vec3 & v) {
    return vec3(round(v.r), round(v.g), round(v.b));
}

__device__ static vec3 lerp(const vec3 & a, const vec3 & b, const float q) {
    return a * (1.0f - q) + b * q;
}



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


__device__ static const float offset = 128.0 / 255.0;

__device__ static vec3 ConvertRGBToYCoCg(vec3 color)
{
    float Y = (color.r + 2.0 * color.g + color.b) * 0.25;
    float Co = ( ( 2.0 * color.r - 2.0 * color.b      ) * 0.25 + offset );
    float Cg = ( (    -color.r + 2.0 * color.g - color.b) * 0.25 + offset );

    return vec3(Y, Co, Cg);
}

// __device__ static float colorDistance(vec3 c0, vec3 c1)
// {
//     return dot(c0-c1, c0-c1);
// }
__device__ static float colorDistance(vec2 c0, vec2 c1)
{
    return dot(c0-c1, c0-c1);
}

__device__ static void FindMinMaxColorsBox(vec3 block[16], vec3 & mincol, vec3 & maxcol)
{
    mincol = block[0];
    maxcol = block[0];
    
    for ( int i = 1; i < 16; i++ ) {
        mincol = min(mincol, block[i]);
        maxcol = max(maxcol, block[i]);
    }
}

// __device__ static void InsetBBox(vec3 & mincol, vec3 & maxcol)
// {
//     vec3 inset = (maxcol - mincol) / 16.0 - (8.0 / 255.0) / 16.0;
//     mincol = clamp(mincol + inset, 0.0, 1.0);
//     maxcol = clamp(maxcol - inset, 0.0, 1.0);
// }
__device__ static void InsetYBBox(float & mincol, float & maxcol)
{
    float inset = (maxcol - mincol) / 32.0 - (16.0 / 255.0) / 32.0;
    mincol = clamp(mincol + inset, 0.0, 1.0);
    maxcol = clamp(maxcol - inset, 0.0, 1.0);
}
__device__ static void InsetCoCgBBox(vec2 & mincol, vec2 & maxcol)
{
    vec2 inset = (maxcol - mincol) / 16.0 - (8.0 / 255.0) / 16.0;
    mincol = clamp(mincol + inset, 0.0, 1.0);
    maxcol = clamp(maxcol - inset, 0.0, 1.0);
}

// __device__ static void SelectDiagonal(vec3 block[16], vec3 & mincol, vec3 & maxcol)
// {
//     vec3 center = (mincol + maxcol) * 0.5;
// 
//     vec2 cov = vec2(0, 0);
//     for (int i = 0; i < 16; i++) {
//         vec3 t = block[i] - center;
//         cov.r += t.r * t.b;
//         cov.g += t.g * t.b;
//     }
// 
//     if (cov.r < 0.0) {
//         float temp = maxcol.r;
//         maxcol.r = mincol.r;
//         mincol.r = temp;
//     }
//     if (cov.g < 0.0) {
//         float temp = maxcol.g;
//         maxcol.g = mincol.g;
//         mincol.g = temp;
//     }
// }

// __device__ static vec3 RoundAndExpand(vec3 v, UINT & w)
// {
//     uvec3 c = uvec3(round(v * vec3(31, 63, 31)));
//     w = (c.r << 11u) | (c.g << 5u) | c.b;
// 
//     c.r = (c.r << 3u) | (c.r >> 2u);
//     c.b = (c.b << 3u) | (c.b >> 2u);
//     c.g = (c.g << 2u) | (c.g >> 4u);
// 
//     return vec3(c) * (1.0 / 255.0);
// }

// __device__ static UINT EmitEndPointsDXT1(vec3 & mincol, vec3 & maxcol)
// {
//     uvec2 outp;
//     maxcol = RoundAndExpand(maxcol, outp.r);
//     mincol = RoundAndExpand(mincol, outp.g);
// 
//     // We have to do this in case we select an alternate diagonal.
//     if (outp.r < outp.g) {
//         vec3 tmp = mincol;
//         mincol = maxcol;
//         maxcol = tmp;
//         return outp.g | (outp.r << 16u);
//     }
// 
//     return outp.r | (outp.g << 16u);
// }

__device__ static UINT ScaleYCoCg(vec2 minColor, vec2 maxColor)
{
    vec2 m0 = abs(minColor - offset);
    vec2 m1 = abs(maxColor - offset);

    float m = max(max(m0.r, m0.g), max(m1.r, m1.g));

    const float s0 = 64.0 / 255.0;
    const float s1 = 32.0 / 255.0;

    UINT scale = 1u;
    if ( m < s0 )
        scale = 2u;
    if ( m < s1 )
        scale = 4u;

    return scale;
}

__device__ static bool SelectYCoCgDiagonal(const vec3 block[16], vec2 minColor, vec2 maxColor)
{
    vec2 mid = (maxColor + minColor) * 0.5;

    float cov = 0.0;
    for ( int i = 0; i < 16; i++ ) {
        vec2 t = block[i].gb() - mid;
        cov += t.r * t.g;
    }
    return cov < 0.0;
}

__device__ static UINT EmitEndPointsYCoCgDXT5(float & mincol_r, float & mincol_g,
                                              float & maxcol_r, float & maxcol_g,
                                              UINT scale)
{
    vec2 mincol = vec2(mincol_r, mincol_g);
    vec2 maxcol = vec2(maxcol_r, maxcol_g);
    
    maxcol = (maxcol - offset) * float(scale) + float(offset);
    mincol = (mincol - offset) * float(scale) + float(offset);

    InsetCoCgBBox(mincol, maxcol);

    maxcol = round(maxcol * vec2(31, 63));
    mincol = round(mincol * vec2(31, 63));

    uvec2 imaxcol = uvec2(maxcol);
    uvec2 imincol = uvec2(mincol);

    uvec2 outp;
    outp.r = (imaxcol.r << 11u) | (imaxcol.g << 5u) | (scale - UINT(1));
    outp.g = (imincol.r << 11u) | (imincol.g << 5u) | (scale - UINT(1));

    imaxcol.r = (imaxcol.r << 3u) | (imaxcol.r >> 2u);
    imaxcol.g = (imaxcol.g << 2u) | (imaxcol.g >> 4u);
    imincol.r = (imincol.r << 3u) | (imincol.r >> 2u);
    imincol.g = (imincol.g << 2u) | (imincol.g >> 4u);

    maxcol = vec2(imaxcol) * (1.0 / 255.0);
    mincol = vec2(imincol) * (1.0 / 255.0);

    // Undo rescale.
    maxcol = (maxcol - offset) / float(scale) + float(offset);
    mincol = (mincol - offset) / float(scale) + float(offset);
    
    // distribute back
    mincol_r = mincol.r;
    mincol_g = mincol.g;
    maxcol_r = maxcol_r;
    maxcol_g = maxcol.g;

    return outp.r | (outp.g << 16u);
}

__device__ static UINT EmitIndicesYCoCgDXT5(vec3 block[16], vec2 mincol, vec2 maxcol)
{
    // Compute palette
    vec2 c[4];
    c[0] = maxcol;
    c[1] = mincol;
    c[2] = lerp(c[0], c[1], 1.0/3.0);
    c[3] = lerp(c[0], c[1], 2.0/3.0);

    // Compute indices
    UINT indices = 0u;
    for ( int i = 0; i < 16; i++ )
    {
        // find index of closest color
        float4 dist;
        dist.x = colorDistance(block[i].gb(), c[0]);
        dist.y = colorDistance(block[i].gb(), c[1]);
        dist.z = colorDistance(block[i].gb(), c[2]);
        dist.w = colorDistance(block[i].gb(), c[3]);

        uint4 b;
        b.x = dist.x > dist.w ? 1u : 0u;
        b.y = dist.y > dist.z ? 1u : 0u;
        b.z = dist.x > dist.z ? 1u : 0u;
        b.w = dist.y > dist.w ? 1u : 0u;
        UINT b4 = dist.z > dist.w ? 1u : 0u;

        UINT index = (b.x & b4) | (((b.y & b.z) | (b.x & b.w)) << 1u);
        indices |= index << (UINT(i) * 2u);
    }

    // Output indices
    return indices;
}

__device__ static UINT EmitAlphaEndPointsYCoCgDXT5(float mincol, float maxcol)
{
    uvec2 tmp = uvec2(round(mincol * 255.0), round(maxcol * 255.0));
    UINT c0 = tmp.r;
    UINT c1 = tmp.g;

    return (c0 << 8u) | c1;
}

// Version shown in the YCoCg-DXT article.
__device__ static uvec2 EmitAlphaIndicesYCoCgDXT5(vec3 block[16], float minAlpha, float maxAlpha)
{
    const float ALPHA_RANGE = 7.0;

    float mid = (maxAlpha - minAlpha) / (2.0 * ALPHA_RANGE);

    float ab1 = minAlpha + mid;
    float ab2 = (6.0 * maxAlpha + 1.0 * minAlpha) * (1.0 / ALPHA_RANGE) + mid;
    float ab3 = (5.0 * maxAlpha + 2.0 * minAlpha) * (1.0 / ALPHA_RANGE) + mid;
    float ab4 = (4.0 * maxAlpha + 3.0 * minAlpha) * (1.0 / ALPHA_RANGE) + mid;
    float ab5 = (3.0 * maxAlpha + 4.0 * minAlpha) * (1.0 / ALPHA_RANGE) + mid;
    float ab6 = (2.0 * maxAlpha + 5.0 * minAlpha) * (1.0 / ALPHA_RANGE) + mid;
    float ab7 = (1.0 * maxAlpha + 6.0 * minAlpha) * (1.0 / ALPHA_RANGE) + mid;

    uvec2 indices = uvec2(0, 0);

    UINT index = 1u;
    for ( int i = 0; i < 6; i++ ) {
        float a = block[i].r;
        index = 1u;
        index += (a <= ab1) ? 1u : 0u;
        index += (a <= ab2) ? 1u : 0u;
        index += (a <= ab3) ? 1u : 0u;
        index += (a <= ab4) ? 1u : 0u;
        index += (a <= ab5) ? 1u : 0u;
        index += (a <= ab6) ? 1u : 0u;
        index += (a <= ab7) ? 1u : 0u;
        index &= 7u;
        index ^= (2u > index) ? 1u : 0u;
        indices.r |= index << (3u * UINT(i) + 16u);
    }

    indices.g = index >> 1u;

    for ( int i = 6; i < 16; i++ ) {
        float a = block[i].r;
        index = 1u;
        index += (a <= ab1) ? 1u : 0u;
        index += (a <= ab2) ? 1u : 0u;
        index += (a <= ab3) ? 1u : 0u;
        index += (a <= ab4) ? 1u : 0u;
        index += (a <= ab5) ? 1u : 0u;
        index += (a <= ab6) ? 1u : 0u;
        index += (a <= ab7) ? 1u : 0u;
        index &= 7u;
        index ^= (2u > index) ? 1u : 0u;
        indices.g |= index << (3u * UINT(i) - 16u);
    }

    return indices;
}



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////




/// Encodes color palette endpoint into 565 code and adjusts input values.
__device__ static u32 encode_endpoint(float & r, float & g, float & b) {
    // clamp to range [0,1] and use full output range for each component
    r = rintf(__saturatef(r) * 31.0f);
    g = rintf(__saturatef(g) * 63.0f);  // 6 bits for green sample
    b = rintf(__saturatef(b) * 31.0f);
    
    // compose output 16bit code representing the endpoint color
    const u32 code = ((u32)r << 11) + ((u32)g << 5) + (u32)b;

    // convert all 3 endpoint component samples back to unit range
    r *= 0.0322580645161f;  // divide by 31
    g *= 0.015873015873f;   // divide by 63
    b *= 0.0322580645161f;  // divide by 31
    
    // return output 16bit code for the endpoint
    return code;
}


/// Transform YUV to RGB.
__device__ static void yuv_to_rgb(float & r, float & g, float & b) {
    const float y = 1.1643f * (r - 0.0625f);  // TODO: convert to FFMA
    const float u = g - 0.5f;
    const float v = b - 0.5f;
    r = y + 1.7926f * v;
    g = y - 0.2132f * u - 0.5328f * v;
    b = y + 2.1124f * u;
}


/// Swaps two referenced values.
template <typename T>
__device__ static void swap(T & a, T & b) {
    const T temp = a;
    a = b;
    b = temp;
}


/// Encodes and saves the block.
template <int DXT_TYPE>
__device__ static void dxt_encode(void * out, const int block_idx,
                                  float r[16], float g[16], float b[16]);


/// Encodes the block into DXT6 format (DXT5-YcOcG) and saves it into output 
/// buffer
template <>
__device__ static void dxt_encode<6>(void * out, const int block_idx,
                                     float r[16], float g[16], float b[16]) {
     // Read block of data
    vec3 block[16];
    
    for(int i = 0; i < 16; i++) {
        block[i] = ConvertRGBToYCoCg(vec3(r[i], g[i], b[i]));
    }

    // Find min and max colors
    vec3 mincol, maxcol;
    FindMinMaxColorsBox(block, mincol, maxcol);

    if(SelectYCoCgDiagonal(block, mincol.gb(), maxcol.gb()))  {
        float tmp = maxcol.b;
        maxcol.b = mincol.b;
        mincol.b = tmp;
    }

    u32 scale = ScaleYCoCg(mincol.gb(), maxcol.gb());
//     printf("Scale: %u.\n", scale);
    // Output CoCg in DXT1 block.
    uint4 outp;
    outp.z = EmitEndPointsYCoCgDXT5(mincol.g, mincol.b, maxcol.g, maxcol.b, scale);
    outp.w = EmitIndicesYCoCgDXT5(block, mincol.gb(), maxcol.gb());

    InsetYBBox(mincol.r, maxcol.r);

    // Output Y in DXT5 alpha block.
    outp.x = EmitAlphaEndPointsYCoCgDXT5(mincol.r, maxcol.r);

    uvec2 indices = EmitAlphaIndicesYCoCgDXT5(block, mincol.r, maxcol.r);
    outp.x |= indices.r;
    outp.y = indices.g;
    
    ((uint4*)out)[block_idx] = outp;
    
}

/// Encodes the block into DXT1 format and saves it into output buffer
template <>
__device__ static void dxt_encode<1>(void * out, const int block_idx,
                                     float r[16], float g[16], float b[16]) {
    // find min and max sample values for each component
    float mincol_r = r[0];
    float mincol_g = g[0];
    float mincol_b = b[0];
    float maxcol_r = r[0];
    float maxcol_g = g[0];
    float maxcol_b = b[0];
    for(int i = 1; i < 16; i++) {
        mincol_r = min(mincol_r, r[i]);
        mincol_g = min(mincol_g, g[i]);
        mincol_b = min(mincol_b, b[i]);
        maxcol_r = max(maxcol_r, r[i]);
        maxcol_g = max(maxcol_g, g[i]);
        maxcol_b = max(maxcol_b, b[i]);
    }

    // inset the bounding box
    const float inset_r = (maxcol_r - mincol_r) * 0.0625f;
    const float inset_g = (maxcol_g - mincol_g) * 0.0625f;
    const float inset_b = (maxcol_b - mincol_b) * 0.0625f;
    mincol_r += inset_r;
    mincol_g += inset_g;
    mincol_b += inset_b;
    maxcol_r -= inset_r;
    maxcol_g -= inset_g;
    maxcol_b -= inset_b;

    // select diagonal
    const float center_r = (mincol_r + maxcol_r) * 0.5f;
    const float center_g = (mincol_g + maxcol_g) * 0.5f;
    const float center_b = (mincol_b + maxcol_b) * 0.5f;
    float cov_x = 0.0f;
    float cov_y = 0.0f;
    for(int i = 0; i < 16; i++) {
        const float dir_r = r[i] - center_r;
        const float dir_g = g[i] - center_g;
        const float dir_b = b[i] - center_b;
        cov_x += dir_r * dir_b;
        cov_y += dir_g * dir_b;
    }
    if(cov_x < 0.0f) {
        swap(maxcol_r, mincol_r);
    }
    if(cov_y < 0.0f) {
        swap(maxcol_g, mincol_g);
    }

    // encode both endpoints into 565 color format
    const u32 max_code = encode_endpoint(maxcol_r, maxcol_g, maxcol_b);
    const u32 min_code = encode_endpoint(mincol_r, mincol_g, mincol_b);

    // swap palette end colors if 'max' code is less than 'min' color code
    // (Palette color #3 would otherwise be interpreted as 'transparent'.)
    const bool swap_end_colors = max_code < min_code;
    
    // encode the palette into 32 bits (Only 2 end colors are stored.)
    const u32 palette_code = swap_end_colors ?
            min_code + (max_code << 16): max_code + (min_code << 16);
    
    // pack palette color indices (if both endpoint colors are not equal)
    u32 indices = 0;
    if(max_code != min_code) {
        // project each color to line maxcol-mincol, represent it as
        // "mincol + t * (maxcol - mincol)" and then use 't' to find closest 
        // palette color index.
        const float dir_r = mincol_r - maxcol_r;
        const float dir_g = mincol_g - maxcol_g;
        const float dir_b = mincol_b - maxcol_b;
        const float dir_sqr_len = dir_r * dir_r + dir_g * dir_g + dir_b * dir_b;
        const float dir_inv_sqr_len = __fdividef(1.0f, dir_sqr_len);
        const float t_r = dir_r * dir_inv_sqr_len;
        const float t_g = dir_g * dir_inv_sqr_len;
        const float t_b = dir_b * dir_inv_sqr_len;
        const float t_bias = t_r * maxcol_r + t_g * maxcol_g + t_b * maxcol_b;
        
        // for each pixel color:
        for(int i = 0; i < 16; i++) {
            // get 't' for the color
            const float col_t = r[i] * t_r + g[i] * t_g + b[i] * t_b - t_bias;
            
            // scale the range of the 't' to [0..3] and convert to integer
            // to get the index of palette color
            const u32 col_idx = (u32)(3.0f * __saturatef(col_t) + 0.5f);
            
            // pack the color palette index with others
            indices += col_idx << (i * 2);
        }
    }
    
    // possibly invert indices if end colors must be swapped
    if(swap_end_colors) {
        indices = ~indices;
    }
    
    // substitute all packed indices (each index is packed into two bits)
    // 00 -> 00, 01 -> 10, 10 -> 11 and 11 -> 01
    const u32 lsbs = indices & 0x55555555;
    const u32 msbs = indices & 0xaaaaaaaa;
    indices = msbs ^ (2 * lsbs + (msbs >> 1));

    // compose and save output
    ((uint2*)out)[block_idx] = make_uint2(palette_code, indices);
}


/// DXT compression - each thread compresses one 4x4 DXT block.
/// Alpha-color palette mode is not used (always emmits 4color palette code).
template <bool YUV_TO_RGB, bool VERTICAL_MIRRORING, int DXT_TYPE>
__global__ static void dxt_kernel(const void * src, void * out, int size_x, int size_y) {
    // coordinates of this thread's 4x4 block
    const int block_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    
    // coordinates of block's top-left pixel
    const int block_x = block_idx_x * 4;
    const int block_y = block_idx_y * 4;
    
    // raster order index of the block
    const int block_idx = block_idx_x + (size_x >> 2) * block_idx_y;
    
    // skip if out of bounds
    if(block_y >= size_y || block_x >= size_x) {
        return;
    }

    // samples of 16 pixels
    float r[16];
    float g[16];
    float b[16];

    // load RGB samples for all 16 input pixels
    const int src_stride = (size_x >> 2) * 3;
    for(int y = 0; y < 4; y++) {
        // offset of loaded pixels in the buffer
        const int load_offset = y * 4;
        
        // pointer to source of this input row
        int row_idx = block_y + y;
        if(VERTICAL_MIRRORING) {
            row_idx = size_y - 1 - row_idx;
        }
        const uchar4 * const row_src = (uchar4*)src
                                     + src_stride * row_idx
                                     + block_idx_x * 3;
        
        // load all 4 3component pixels of the row
        const uchar4 p0 = row_src[0];
        const uchar4 p1 = row_src[1];
        const uchar4 p2 = row_src[2];
        
        // pixel #0
        r[load_offset + 0] = p0.x * 0.00392156862745f;
        g[load_offset + 0] = p0.y * 0.00392156862745f;
        b[load_offset + 0] = p0.z * 0.00392156862745f;
        
        // pixel #1
        r[load_offset + 1] = p0.w * 0.00392156862745f;
        g[load_offset + 1] = p1.x * 0.00392156862745f;
        b[load_offset + 1] = p1.y * 0.00392156862745f;
        
        // pixel #2
        r[load_offset + 2] = p1.z * 0.00392156862745f;
        g[load_offset + 2] = p1.w * 0.00392156862745f;
        b[load_offset + 2] = p2.x * 0.00392156862745f;
        
        // pixel #3
        r[load_offset + 3] = p2.y * 0.00392156862745f;
        g[load_offset + 3] = p2.z * 0.00392156862745f;
        b[load_offset + 3] = p2.w * 0.00392156862745f;
    }
    
    // transform colors from YUV to RGB if required
    if(YUV_TO_RGB) {
        for(int i = 0; i < 16; i++) {
            yuv_to_rgb(r[i], g[i], b[i]);
        }
    }
    
    // Select the right DXT type transform
    dxt_encode<DXT_TYPE>(out, block_idx, r, g, b);
}


/// Compute grid size and launch DXT kernel.
template <bool YUV_TO_RGB, int DXT_TYPE>
static int dxt_launch(const void * src, void * out, int sx, int sy, cudaStream_t str) {
    // vertical mirroring?
    bool mirrored = false;
    if(sy < 0) {
        mirrored = true;
        sy = -sy;
    }
    
    // check image size and alignment
    if((sx & 3) || (sy & 3) || (15 & (size_t)src) || (7 & (size_t)out)) {
        return -1;
    }
    
    // grid and threadblock sizes
    const dim3 tsiz(16, 16);
    const dim3 gsiz((sx + tsiz.x - 1) / tsiz.x, (sy + tsiz.y - 1) / tsiz.y);
    
    // launch kernel, sync and check the result
    if(mirrored) {
        dxt_kernel<YUV_TO_RGB, true, DXT_TYPE><<<gsiz, tsiz, 0, str>>>(src, out, sx, sy);
    } else {
        dxt_kernel<YUV_TO_RGB, false, DXT_TYPE><<<gsiz, tsiz, 0, str>>>(src, out, sx, sy);
    }
    return cudaSuccess != cudaStreamSynchronize(str) ? -3 : 0;
}


/// CUDA DXT1 compression (only RGB without alpha).
/// @param src  Pointer to top-left source pixel in device-memory buffer. 
///             8bit RGB samples are expected (no alpha and no padding).
///             (Pointer must be aligned to multiples of 16 bytes.)
/// @param out  Pointer to output buffer in device memory.
///             (Must be aligned to multiples of 8 bytes.)
/// @param size_x  Width of the input image (must be divisible by 4).
/// @param size_y  Height of the input image (must be divisible by 4).
/// @param stream  CUDA stream to run in, or 0 for default stream.
/// @return 0 if OK, nonzero if failed.
int cuda_rgb_to_dxt1(const void * src, void * out, int size_x, int size_y, cudaStream_t stream) {
    return dxt_launch<false, 1>(src, out, size_x, size_y, stream);
}


/// CUDA DXT1 compression (only RGB without alpha).
/// Converts input from YUV to RGB color space.
/// @param src  Pointer to top-left source pixel in device-memory buffer. 
///             8bit RGB samples are expected (no alpha and no padding).
///             (Pointer must be aligned to multiples of 16 bytes.)
/// @param out  Pointer to output buffer in device memory.
///             (Must be aligned to multiples of 8 bytes.)
/// @param size_x  Width of the input image (must be divisible by 4).
/// @param size_y  Height of the input image (must be divisible by 4).
/// @param stream  CUDA stream to run in, or 0 for default stream.
/// @return 0 if OK, nonzero if failed.
int cuda_yuv_to_dxt1(const void * src, void * out, int size_x, int size_y, cudaStream_t stream) {
    return dxt_launch<true, 1>(src, out, size_x, size_y, stream);
}


/// CUDA DXT6 (DXT5-YcOcG) compression (only RGB without alpha).
/// @param src  Pointer to top-left source pixel in device-memory buffer. 
///             8bit RGB samples are expected (no alpha and no padding).
///             (Pointer must be aligned to multiples of 16 bytes.)
/// @param out  Pointer to output buffer in device memory.
///             (Must be aligned to multiples of 8 bytes.)
/// @param size_x  Width of the input image (must be divisible by 4).
/// @param size_y  Height of the input image (must be divisible by 4).
///                (Input is read bottom up if negative)
/// @param stream  CUDA stream to run in, or 0 for default stream.
/// @return 0 if OK, nonzero if failed.
int cuda_rgb_to_dxt6(const void * src, void * out, int size_x, int size_y, cudaStream_t stream) {
    return dxt_launch<false, 6>(src, out, size_x, size_y, stream);
}
