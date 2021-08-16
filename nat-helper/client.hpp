/**
 * @file   client.hpp
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
#ifndef UG_HOLE_PUNCH_CLIENT_HPP
#define UG_HOLE_PUNCH_CLIENT_HPP

#include <string>
#include <vector>
#include <queue>
#include <functional>
#include <memory>
#include <asio.hpp>
#include "message.hpp"

class Client : public std::enable_shared_from_this<Client>{
public:
	Client(asio::ip::tcp::socket&& socket);
	Client(const Client&) = delete;
	Client(Client&&) = delete;

	Client& operator=(const Client&) = delete;
	Client& operator=(Client&&) = delete;

	void readDescription(std::function<void(Client&, bool)> onComplete);
	std::string getClientName() { return clientName; }
	std::string getRoomName() { return roomName; }
	std::string getSdpDesc() { return sdpDesc; }

	void sendMsg(std::string_view msg);

	void readCandidate(std::function<void(Client&, bool)> onComplete);
	const std::vector<std::string>& getCandidates() { return candidates; }

	bool isSendCallbackPending() const;

private:
	void readNameComplete(asio::ip::tcp::socket& socket,
			std::function<void(Client&, bool)> onComplete, bool success);

	void readRoomComplete(asio::ip::tcp::socket& socket,
			std::function<void(Client&, bool)> onComplete, bool success);

	void readDescComplete(
			std::function<void(Client&, bool)> onComplete, bool success);

	void readCandidateComplete(
			std::function<void(Client&, bool)> onComplete, bool success);

	void onMsgSent(bool success);

	std::string clientName;
	std::string roomName;
	std::string sdpDesc;
	std::vector<std::string> candidates;

	std::queue<std::string> sendQueue;

	Message inMsg;
	Message outMsg;
	asio::ip::tcp::socket socket;
};


#endif
