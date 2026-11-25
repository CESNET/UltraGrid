/**
 * @file   test/test_rapidcheck_main.cpp
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Runner for the property-based test suites.
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
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

#include <iostream>

extern bool test_alpha_blend_properties();
extern bool test_overlay_layout_properties();
extern bool test_soft_edge_properties();
extern bool test_overlay_watch_properties();

/* color_space.c calls get_commandline_param("color-601"); register_param
 * runs at static-init time. Linking host.cpp into the test binary would
 * drag in most of UltraGrid, so stub these two — the tests never set
 * either flag, and a NULL return from get_commandline_param falls back
 * to the codebase's BT.709 default. */
extern "C" const char *get_commandline_param(const char *) { return nullptr; }
extern "C" void register_param(const char *, const char *) {}

int main()
{
        std::cout << "UltraGrid property-based tests (RapidCheck)\n";

        bool ok = true;
        std::cout << "\n=== Alpha blend ===\n";
        ok &= test_alpha_blend_properties();
        std::cout << "\n=== Overlay layout ===\n";
        ok &= test_overlay_layout_properties();
        std::cout << "\n=== Soft edges ===\n";
        ok &= test_soft_edge_properties();
        std::cout << "\n=== Overlay watch ===\n";
        ok &= test_overlay_watch_properties();

        std::cout << (ok ? "\nAll properties passed.\n"
                         : "\nFAILURES — see output above.\n");
        return ok ? 0 : 1;
}
