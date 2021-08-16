/**
 * @file   main.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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
#include <charconv>
#include "nat-helper.hpp"

namespace {
	void printUsage(std::string_view name){
		std::cout << "Usage: "
			<< name << " [-h/--help] [-p/--port <port>]\n";
	}
}

int main(int argc, char **argv){
	int port = 12558;

	for(int i = 1; i < argc; i++){
		std::string_view arg(argv[i]);
		if(arg == "-h" || arg == "--help"){
			printUsage(argv[0]);
			return 0;
		} else if(arg == "-p" || arg == "--port"){
			if(i + 1 >= argc){
				std::cerr << "Expected port number\n";
				printUsage(argv[0]);
				return 1;
			}
			std::string_view portStr(argv[++i]);
			auto res = std::from_chars(portStr.data(), portStr.data() + portStr.size(), port);
			if(res.ec != std::errc() || res.ptr == portStr.data()){
				std::cerr << "Failed to parse port number\n";
				printUsage(argv[0]);
				return 1;
			}
		} else {
			std::cerr << "Unknown argument " << arg << std::endl;
			printUsage(argv[0]);
			return 1;
		}
	}


	NatHelper server(port);

	server.run();

	for(std::string in; std::getline(std::cin, in);){
		if(in == "exit"){
			break;
		}
	}

	server.stop();

	return 0;
}
