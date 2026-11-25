/**
 * @file   test/test_file_monitoring_rapidcheck.cpp
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Property-based tests for utils/overlay_watch.c.
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

#include <rapidcheck.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" {
#include "utils/overlay_watch.h"
}

namespace {

/* Local tempfile RAII. The project's get_temp_file() in utils/fs.h is
 * the canonical helper (and is used by the unit-test sibling
 * test_overlay_watch.c), but linking fs.o pulls in transitive deps
 * (log_msg, strrpbrk, get_ug_data_path) that would balloon the test
 * binary. Test runs on POSIX dev machines only, so mkstemp suffices. */
struct tempfile {
        std::string path;
        tempfile()
            : path("/tmp/ug-overlay-watch-XXXXXX")
        {
                std::vector<char> buf(path.begin(), path.end());
                buf.push_back('\0');
                int fd = mkstemp(buf.data());
                if (fd < 0) {
                        path.clear();
                        return;
                }
                close(fd);
                path = buf.data();
        }
        ~tempfile() { if (!path.empty()) unlink(path.c_str()); }
        tempfile(const tempfile &) = delete;
        tempfile &operator=(const tempfile &) = delete;

        void write_bytes(const std::vector<uint8_t> &bytes) const
        {
                FILE *f = fopen(path.c_str(), "wb");
                RC_ASSERT(f != nullptr);
                const size_t n = fwrite(bytes.data(), 1, bytes.size(), f);
                fclose(f);
                RC_ASSERT(n == bytes.size());
        }
};

std::vector<uint8_t> gen_bytes(size_t min_n = 1, size_t max_n = 64)
{
        const size_t n = static_cast<size_t>(*rc::gen::inRange(
                static_cast<int>(min_n), static_cast<int>(max_n + 1)));
        return *rc::gen::container<std::vector<uint8_t>>(
                n, rc::gen::arbitrary<uint8_t>());
}

} // namespace

bool test_overlay_watch_properties()
{
        bool ok = true;

        /* Init + immediate change-check on the same file: nothing changed,
         * so changed() must be false. */
        ok &= rc::check("watch: init then changed() == false", []() {
                tempfile f;
                RC_PRE(!f.path.empty());
                const auto bytes = gen_bytes(1, 32);
                f.write_bytes(bytes);

                struct overlay_watch w;
                overlay_watch_init(&w, f.path.c_str());
                RC_ASSERT(!overlay_watch_changed(&w, f.path.c_str()));
        });

        /* Size change is detected. We rewrite with a different number of
         * bytes; mtime may or may not tick (filesystem resolution), but
         * size always differs. */
        ok &= rc::check("watch: size change detected", []() {
                tempfile f;
                RC_PRE(!f.path.empty());
                const auto a = gen_bytes(1, 16);
                const auto b = gen_bytes(17, 64);
                f.write_bytes(a);

                struct overlay_watch w;
                overlay_watch_init(&w, f.path.c_str());
                f.write_bytes(b);
                RC_ASSERT(overlay_watch_changed(&w, f.path.c_str()));
        });

        /* ack() commits the new fingerprint, so a subsequent changed()
         * with no further edit returns false. */
        ok &= rc::check("watch: ack clears the change", []() {
                tempfile f;
                RC_PRE(!f.path.empty());
                f.write_bytes(gen_bytes(1, 8));

                struct overlay_watch w;
                overlay_watch_init(&w, f.path.c_str());
                f.write_bytes(gen_bytes(9, 32));
                RC_ASSERT(overlay_watch_changed(&w, f.path.c_str()));
                overlay_watch_ack(&w, f.path.c_str());
                RC_ASSERT(!overlay_watch_changed(&w, f.path.c_str()));
        });

        /* Missing file at init: watch is invalid; the first time the file
         * appears, changed() returns true. */
        ok &= rc::check("watch: missing-then-appears triggers change", []() {
                tempfile f;
                RC_PRE(!f.path.empty());
                /* Delete it before init so the watch starts invalid. */
                unlink(f.path.c_str());

                struct overlay_watch w;
                overlay_watch_init(&w, f.path.c_str());
                /* No file: changed() returns false (transient, per docs). */
                RC_ASSERT(!overlay_watch_changed(&w, f.path.c_str()));
                f.write_bytes(gen_bytes(1, 16));
                RC_ASSERT(overlay_watch_changed(&w, f.path.c_str()));
        });

        /* fingerprint() gives a stable answer for an unchanged file. */
        ok &= rc::check("watch: fingerprint stable when file unchanged", []() {
                tempfile f;
                RC_PRE(!f.path.empty());
                f.write_bytes(gen_bytes(1, 16));

                int64_t m1 = 0, s1 = 0, m2 = 0, s2 = 0;
                RC_ASSERT(overlay_watch_fingerprint(f.path.c_str(), &m1, &s1));
                RC_ASSERT(overlay_watch_fingerprint(f.path.c_str(), &m2, &s2));
                RC_ASSERT(m1 == m2);
                RC_ASSERT(s1 == s2);
        });

        return ok;
}
