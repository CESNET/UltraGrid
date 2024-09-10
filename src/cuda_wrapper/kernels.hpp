/**
 * @file   cuda_wrapper/kernels.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2024 CESNET
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CUDA_WRAPPER_KERNELS_HPP_1A3F7B57_EE91_4363_8D50_9CDDC60CB74F
#define CUDA_WRAPPER_KERNELS_HPP_1A3F7B57_EE91_4363_8D50_9CDDC60CB74F

#include <cstddef>

struct cmpto_j2k_dec_comp_format;

int postprocess_rg48_to_r12l(
    void * postprocessor,
    void * img_custom_data,
    size_t img_custom_data_size,
    int size_x,
    int size_y,
    struct cmpto_j2k_dec_comp_format * comp_formats,
    int comp_count,
    void * input_samples,
    size_t input_samples_size,
    void * temp_buffer,
    size_t temp_buffer_size,
    void * output_buffer,
    size_t output_buffer_size,
    void * stream
);

void preprocess_r12l_to_rg48(int width, int height, void *src, void *dst);

#endif // defined CUDA_WRAPPER_KERNELS_HPP_1A3F7B57_EE91_4363_8D50_9CDDC60CB74F
