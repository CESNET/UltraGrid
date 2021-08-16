/**
 * @file   message.cpp
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
#include <charconv>
#include <cassert>
#include <cstring>
#include "message.hpp"

std::string_view Message::getStr() const{
	return std::string_view(data + headerSize, size);
}

void Message::async_readMsg(asio::ip::tcp::socket& socket,
		std::function<void(bool)> onComplete)
{
	using namespace std::placeholders;
	asio::async_read(socket,
			asio::buffer(data, headerSize),
			std::bind(&Message::async_readBody,
				this, std::ref(socket), onComplete, _1, _2));
}

void Message::setStr(std::string_view msg){
	assert(!sendingNow);
	size = std::min(bodySize, msg.size());

	memset(data, ' ', headerSize);
	std::to_chars(data, data + headerSize, size);

	memcpy(data + headerSize, msg.data(), size);
}

void Message::async_sendMsg(asio::ip::tcp::socket& socket,
		std::function<void(bool)> onComplete)
{
	using namespace std::placeholders;
	sendingNow = true;
	asio::async_write(socket,
			asio::buffer(data, headerSize + size),
			std::bind(&Message::async_sendComplete,
				this, onComplete, _1, _2));
}

void Message::async_readBody(asio::ip::tcp::socket& socket,
		std::function<void(bool)> onComplete,
		const std::error_code& ec, [[maybe_unused]] size_t readLen)
{
	if(ec){
		onComplete(false);
		return;
	}

	assert(readLen == headerSize);
	size_t expectedSize = 0;

	auto res = std::from_chars(data, data + headerSize, expectedSize);
	if(res.ec != std::errc()){
		onComplete(false);
	}

	using namespace std::placeholders;
	asio::async_read(socket,
			asio::buffer(data + headerSize, expectedSize),
			std::bind(&Message::async_readBodyComplete,
				this, onComplete, _1, _2));

}

void Message::async_readBodyComplete(std::function<void(bool)> onComplete,
		const std::error_code& ec, size_t readLen)
{
	if(ec){
		onComplete(false);
		return;
	}

	size = readLen;

	onComplete(true);
}

void Message::async_sendComplete(std::function<void(bool)> onComplete,
		const std::error_code& ec, [[maybe_unused]] size_t sentLen)
{
	sendingNow = false;
	onComplete(!ec);
}
