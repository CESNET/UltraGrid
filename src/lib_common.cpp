/**
 * @file   lib_common.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2015 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#ifdef BUILD_LIBRARIES
#include <dlfcn.h>
#include <glob.h>
#include <libgen.h>
#endif

#include <iostream>
#include <map>

#include "debug.h"
#include "host.h"

#include "lib_common.h"
#include "rang.hpp"

using namespace std;

struct library_class_info_t {
        const char *class_name;
        const char *file_prefix;
};

const map<enum library_class, library_class_info_t> library_class_info = {
        { LIBRARY_CLASS_UNDEFINED, { "General module", "" }},
        { LIBRARY_CLASS_CAPTURE_FILTER, { "Video capture filter" , "vcapfilter" }},
        { LIBRARY_CLASS_AUDIO_CAPTURE, { "Audio capture device", "acap" }},
        { LIBRARY_CLASS_AUDIO_PLAYBACK, { "Audio playback device", "aplay" }},
        { LIBRARY_CLASS_VIDEO_CAPTURE, { "Video capture device", "vidcap" }},
        { LIBRARY_CLASS_VIDEO_DISPLAY, { "Video display device", "display" }},
        { LIBRARY_CLASS_AUDIO_COMPRESS, { "Audio compression", "acompress" }},
        { LIBRARY_CLASS_VIDEO_DECOMPRESS, { "Video decompression", "vdecompress" }},
        { LIBRARY_CLASS_VIDEO_COMPRESS, { "Video compression", "vcompress" }},
        { LIBRARY_CLASS_VIDEO_POSTPROCESS, { "Video postprocess", "vo_pp" }},
        { LIBRARY_CLASS_VIDEO_RXTX, { "Video RXTX", "video_rxtx" }},
};

static map<string, string> lib_errors;

#ifdef BUILD_LIBRARIES
static void push_basename_entry(char ***binarynames, const char *bnc, size_t * templates) {
	char * alt_v0 = strdup(bnc);
	assert(alt_v0 != NULL);
	char * bn_v0 = basename(alt_v0);
	for (size_t i = 0; i < *templates && bn_v0 != NULL; ++i) {
		if (strcmp((*binarynames)[i], bn_v0) == 0)
			bn_v0 = NULL;
	}
	if (bn_v0 != NULL) {
		*binarynames = (char **)realloc(*binarynames, (*templates + 2)*sizeof(**binarynames));
		assert(*binarynames != NULL);
		(*binarynames)[(*templates)++] = strdup(bn_v0);
		(*binarynames)[*templates] = NULL;
	}
	free(alt_v0);
}

static int running_from_path(char * uv_argv[]) {
	const char *env_path = getenv("PATH");
	if (env_path == NULL) {
		// LOG(LOG_LEVEL_WARNING) << "Failed to determine environment PATH variable, plugins shall be searched relative to binary" << "\n";
		return 0;
	}

	char *self_path = realpath("/proc/self/exe", NULL);
	if (self_path == NULL) {
		// LOG(LOG_LEVEL_WARNING) << "Failed to determine path to running process" << "\n";
		return 0;
	}

	// considered binary names, currently only uv, but reserve other as well
	const char * binarynames_tpl[] = {
		"uv",
		"ultragrid",
		"uv-nightly",
		"ultragrid-nightly"
	};

	size_t templates = sizeof(binarynames_tpl)/sizeof(binarynames_tpl[0]);
	char **binarynames = (char **)calloc(templates + 1 + 3, sizeof(*binarynames)); // pre-allocate 3 dirnames
	assert(binarynames != NULL);
	for (size_t i = 0; i < templates; ++i)
		binarynames[i] = strdup(binarynames_tpl[i]);
	
	push_basename_entry(&binarynames, uv_argv[0], &templates);
	if (strchr(uv_argv[0], '/') != NULL) {
		char * alt_v0 = realpath(uv_argv[0], NULL);
		if (alt_v0 != NULL) {
			push_basename_entry(&binarynames, alt_v0, &templates);
			free(alt_v0);
		}
	}
	push_basename_entry(&binarynames, self_path, &templates);

	char *pathdup = strdup(env_path);
	char *tokencont = pathdup;
	int rval = 0;

	// if path contains /, make it the first and only candidate 

	char *next_token = NULL;
	while (rval == 0) {
		// loop through paths
		const char *pathelem = strtok_r(tokencont, ":", &next_token);
		tokencont = NULL;
		if (pathelem == NULL)
			break;
		
		for (size_t i = 0; binarynames[i] != NULL && rval == 0; ++i) {
			char * candidate = (char *)calloc(1, strlen(pathelem) + 1 + strlen(binarynames[i]) + 1);
			sprintf(candidate, "%s/%s", pathelem, binarynames[i]);

			char * real_candidate = realpath(candidate, NULL);
			if (real_candidate != NULL) {
				rval = strcmp(real_candidate, self_path) == 0;
				free(real_candidate);
			}
			free(candidate);
		}	
	};
	
	for (size_t i = 0; binarynames[i] != NULL; ++i)
		free(binarynames[i]);
	free(binarynames);

	free(pathdup);
	free(self_path);
	return rval;
}
#endif

void open_all(const char *pattern) {
#ifdef BUILD_LIBRARIES
        char path[512];
        glob_t glob_buf;

        /* binary not from $PATH */
        if (!running_from_path(uv_argv)) {
                char *tmp = strdup(uv_argv[0]);
                char *dir = dirname(tmp);
                snprintf(path, sizeof(path), "%s/../lib/ultragrid/%s", dir, pattern);
                free(tmp);
        } else {
                snprintf(path, sizeof(path), LIB_DIR "/ultragrid/%s", pattern);
        }

        glob(path, 0, NULL, &glob_buf);

        for(unsigned int i = 0; i < glob_buf.gl_pathc; ++i) {
                if (!dlopen(glob_buf.gl_pathv[i], RTLD_NOW|RTLD_GLOBAL)) {
                        char *error = dlerror();
                        verbose_msg("Library %s opening warning: %s \n", glob_buf.gl_pathv[i],
                                        error);
                        char *filename = basename(glob_buf.gl_pathv[i]);
                        if (filename && error) {
                                lib_errors.emplace(filename, error);
                        }
                }
        }

        globfree(&glob_buf);
#else
        UNUSED(pattern);
#endif
}

struct lib_info {
        const void *data;
        int abi_version;
};

// http://stackoverflow.com/questions/1801892/making-mapfind-operation-case-insensitive
/************************************************************************/
/* Comparator for case-insensitive comparison in STL assos. containers  */
/************************************************************************/
struct ci_less : std::binary_function<std::string, std::string, bool>
{
        // case-independent (ci) compare_less binary function
        struct nocase_compare : public std::binary_function<unsigned char,unsigned char,bool>
        {
                bool operator() (const unsigned char& c1, const unsigned char& c2) const {
                        return tolower (c1) < tolower (c2);
                }
        };
        bool operator() (const std::string & s1, const std::string & s2) const {
                return std::lexicographical_compare
                        (s1.begin (), s1.end (),   // source range
                         s2.begin (), s2.end (),   // dest range
                         nocase_compare ());  // comparison
        }
};

static map<enum library_class, map<string, lib_info, ci_less>> *libraries = nullptr;

/**
 * The purpose of this initializor instead of ordinary static initialization is that register_video_capture_filter()
 * may be called before static members are initialized (it is __attribute__((constructor)))
 */
struct init_libraries {
        init_libraries() {
                if (libraries == nullptr) {
                        libraries = new remove_pointer<decltype(libraries)>::type();
                }
        }
};

static struct init_libraries loader;

void register_library(const char *name, const void *data, enum library_class cls, int abi_version)
{
        struct init_libraries loader;
        (*libraries)[cls][name] = {data, abi_version};
}

const void *load_library(const char *name, enum library_class cls, int abi_version)
{
        if (libraries->find(cls) != libraries->end()) {
                auto it_cls = libraries->find(cls)->second;
                auto it_module = it_cls.find(name);
                if (it_module != it_cls.end()) {
                        auto mod_pair = it_cls.find(name)->second;
                        if (mod_pair.abi_version == abi_version) {
                                return mod_pair.data;
                        } else {
                                LOG(LOG_LEVEL_WARNING) << "Module " << name << " ABI version mismatch (required " <<
                                        abi_version << ", have " << mod_pair.abi_version << ")\n";
                        }
                }
        }

        // Library was not found or was not loaded due to unsatisfied
        // dependencies. If the latter one, display reason why dlopen() failed.
        if (library_class_info.find(cls) != library_class_info.end()) {
                string filename = "ultragrid_";
                if (strlen(library_class_info.at(cls).file_prefix) > 0) {
                        filename += library_class_info.at(cls).file_prefix;
                        filename += "_";
                }

                filename += name + string(".so");

                if (lib_errors.find(filename) != lib_errors.end()) {
                        LOG(LOG_LEVEL_WARNING) << filename << ": " << lib_errors.find(filename)->second << "\n";
                }
        }

        return NULL;
}

void list_modules(enum library_class cls, int abi_version) {
        const auto & class_set = get_libraries_for_class(cls, abi_version);
        for (auto && item : class_set) {
                cout << rang::style::bold << "\t" << item.first.c_str() << "\n" << rang::style::reset;
        }
}

void list_all_modules() {
        for (auto cls_it = library_class_info.begin(); cls_it != library_class_info.end();
                        ++cls_it) {
                cout << cls_it->second.class_name << "\n";
                auto it = libraries->find(cls_it->first);
                if (it != libraries->end()) {
                        for (auto && item : it->second) {
                                cout << rang::style::bold << "\t" << item.first << "\n" << rang::style::reset;
                        }
                }
                cout << "\n";
        }

        if (!lib_errors.empty()) {
                cout << rang::style::bold << rang::fg::red << "Errors:\n" << rang::fg::reset << rang::style::reset;
                for (auto && item : lib_errors) {
                        cout << "\t" << rang::fg::red << item.first << rang::fg::reset << "\n\t\t" << item.second << "\n";
                }
                cout << "\n";
        }
}

map<string, const void *> get_libraries_for_class(enum library_class cls, int abi_version)
{
        map<string, const void *> ret;
        auto it = libraries->find(cls);
        if (it != libraries->end()) {
                for (auto && item : it->second) {
                        if (abi_version == item.second.abi_version) {
                                ret[item.first] = item.second.data;
                        } else {
                                LOG(LOG_LEVEL_WARNING) << "Module " << item.first << " ABI version mismatch (required " <<
                                        abi_version << ", have " << item.second.abi_version << ")\n";

                        }
                }
        }

        return ret;
}

