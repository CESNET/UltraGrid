/**
 * @file   message.hpp
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
#ifndef UG_HOLE_PUNCH_MESSAGE_HPP
#define UG_HOLE_PUNCH_MESSAGE_HPP

#include <string_view>
#include <functional>
#include <asio.hpp>

class Message{
public:
	Message() = default;

	void async_readMsg(asio::ip::tcp::socket& socket,
			std::function<void(bool)> onComplete);

	std::string_view getStr() const;

	void setStr(std::string_view msg);
	void async_sendMsg(asio::ip::tcp::socket& socket,
			std::function<void(bool)> onComplete);

	size_t getSize() const { return size; }
	bool empty() const { return size == 0; }
	bool isSendingNow() const { return sendingNow; }
	void clear();

private:
	void async_readBody(asio::ip::tcp::socket& socket,
			std::function<void(bool)> onComplete,
			const std::error_code& ec, size_t readLen);

	void async_readBodyComplete(std::function<void(bool)> onComplete,
			const std::error_code& ec, size_t readLen);

	void async_sendComplete(std::function<void(bool)> onComplete,
			const std::error_code& ec, size_t sentLen);

	size_t size = 0;

	bool sendingNow = false;

	static constexpr size_t headerSize = 5;
	static constexpr size_t bodySize = 2048;
	char data[headerSize + bodySize];
};

#endif
