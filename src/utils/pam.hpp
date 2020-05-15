/**
 * @file   utils/pam.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file is a part of UltraGrid.
 */
/*
 * Copyright (c) 2013-2020 CESNET z.s.p.o.
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

#ifndef PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F
#define PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F

#include <exception>
#include <fstream>
#include <iostream>
#include <string>

[[maybe_unused]] static bool pam_read(const char *filename, unsigned int *width, unsigned int *height, int *depth, unsigned char **data, void *(*allocator)(size_t) = malloc) {
        try {
                std::string line;
                std::ifstream file(filename, std::ifstream::in | std::ifstream::binary);

                file.exceptions(std::ifstream::failbit | std::ifstream::badbit );

                getline(file, line);
                if (!file.good() || line != "P7") {
                        throw std::string("Only logo in PAM format is currently supported.");
                }
                getline(file, line);
                *width = 0, *height = 0, *depth = 0;
                while (!file.eof()) {
                        if (line.compare(0, std::string("WIDTH ").length(), "WIDTH") == 0) {
                                *width = atoi(line.c_str() + std::string("WIDTH ").length());
                        } else if (line.compare(0, std::string("HEIGHT ").length(), "HEIGHT") == 0) {
                                *height = atoi(line.c_str() + std::string("HEIGHT ").length());
                        } else if (line.compare(0, std::string("DEPTH ").length(), "DEPTH") == 0) {
                                *depth = atoi(line.c_str() + std::string("DEPTH ").length());
                        } else if (line.compare(0, std::string("MAXVAL ").length(), "MAXVAL") == 0) {
                                if (atoi(line.c_str() + std::string("MAXVAL ").length()) != 255) {
                                        throw std::string("Only supported maxval is 255.");
                                }
                        } else if (line.compare(0, std::string("TUPLETYPE").length(), "TUPLETYPE") == 0) {
                                // ignored - assuming MAXVAL == 255, value of DEPTH is sufficient
                                // to determine pixel format
                        } else if (line.compare(0, std::string("ENDHDR").length(), "ENDHDR") == 0) {
                                break;
                        }
                        getline(file, line);
                }
                if (*width * *height == 0) {
                        throw std::string("Unspecified size header field!");
                }
                if (*depth == 0) {
                        throw std::string("Unspecified depth header field!");
                }
                if (data != nullptr && allocator != nullptr) {
                        int datalen = *depth * *width * *height;
                        *data = (unsigned char *) allocator(datalen);
                        if (!*data) {
                                throw std::string("Unable to allocate data.");
                        }
                        file.read((char *) *data, datalen);
                        if (file.eof()) {
                                throw std::string("Unable to load PAM data from file.");
                        }
                }
                file.close();
        } catch (std::string const & s) {
                std::cerr << s << std::endl;
                return false;
        } catch (std::exception const & e) {
                std::cerr << e.what() << std::endl;
                return false;
        } catch (...) {
                return false;
        }
        return true;
}

[[maybe_unused]] static bool pam_write(const char *filename, unsigned int width, unsigned int height, int depth, const unsigned char *data) {
        try {
                std::ofstream file(filename, std::ifstream::out | std::ifstream::binary);

                file.exceptions(std::ifstream::failbit | std::ifstream::badbit );
                file << "P7\n";
                file << "WIDTH " << width << "\n";
                file << "HEIGHT " << height << "\n";
                file << "DEPTH " << depth << "\n";
                file << "MAXVAL 255\n";
                std::string tuple_type;
                switch (depth) {
                        case 4: tuple_type = "RGB_ALPHA"; break;
                        case 3: tuple_type = "RGB"; break;
                        case 2: tuple_type = "GRAYSCALE_ALPHA"; break;
                        case 1: tuple_type = "GRAYSCALE"; break;
                        default: std::cerr << "Wrong depth: " << depth << "\n";
                }
                file << "TUPLTYPE " << tuple_type << "\n";
                file << "ENDHDR\n";
                file.write((const char *) data, width * height * depth);
                file.close();
        } catch (std::string const & s) {
                std::cerr << s << std::endl;
                return false;
        } catch (std::exception const & e) {
                std::cerr << e.what() << std::endl;
                return false;
        } catch (...) {
                return false;
        }
        return true;
}

#endif // defined PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F
