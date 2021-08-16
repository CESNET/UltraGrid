/**
 * @file   client.cpp
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
#include "client.hpp"

Client::Client(asio::ip::tcp::socket&& socket) : socket(std::move(socket)) {  }

void Client::readDescription(std::function<void(Client&, bool)> onComplete){
	using namespace std::placeholders;
	inMsg.async_readMsg(socket,
			std::bind(&Client::readNameComplete,
				shared_from_this(), std::ref(socket), onComplete, _1));
}

void Client::readNameComplete(asio::ip::tcp::socket& socket,
		std::function<void(Client&, bool)> onComplete,
		bool success)
{
	if(!success){
		onComplete(*this, false);
		return;
	}

	clientName = std::string(inMsg.getStr());
	using namespace std::placeholders;
	inMsg.async_readMsg(socket,
			std::bind(&Client::readRoomComplete,
				shared_from_this(), std::ref(socket), onComplete, _1));
}

void Client::readRoomComplete(asio::ip::tcp::socket& socket,
		std::function<void(Client&, bool)> onComplete,
		bool success)
{
	if(!success){
		onComplete(*this, false);
		return;
	}

	roomName = std::string(inMsg.getStr());
	using namespace std::placeholders;
	inMsg.async_readMsg(socket,
			std::bind(&Client::readDescComplete, shared_from_this(), onComplete, _1));
}

void Client::readDescComplete(
		std::function<void(Client&, bool)> onComplete,
		bool success)
{
	if(!success){
		onComplete(*this, false);
		return;
	}

	sdpDesc = std::string(inMsg.getStr());
	onComplete(*this, true);
}

void Client::readCandidate(std::function<void(Client&, bool)> onComplete){
	using namespace std::placeholders;
	inMsg.async_readMsg(socket,
			std::bind(&Client::readCandidateComplete,
				shared_from_this(), onComplete, _1));
}

void Client::readCandidateComplete(
		std::function<void(Client&, bool)> onComplete,
		bool success)
{
	if(!success){
		onComplete(*this, false);
		return;
	}

	candidates.emplace_back(inMsg.getStr());
	onComplete(*this, true);
}

bool Client::isSendCallbackPending() const{
	return !sendQueue.empty() || outMsg.isSendingNow();
}


void Client::sendMsg(std::string_view msg){
	if(isSendCallbackPending()){
		sendQueue.emplace(msg);
		return;
	}

	outMsg.setStr(msg);
	using namespace std::placeholders;
	outMsg.async_sendMsg(socket, std::bind(&Client::onMsgSent, shared_from_this(), _1));
}

void Client::onMsgSent(bool success){
	if(!success){
		//TODO
	}

	if(sendQueue.empty())
		return;

	using namespace std::placeholders;
	outMsg.setStr(sendQueue.front());
	sendQueue.pop();
	outMsg.async_sendMsg(socket, std::bind(&Client::onMsgSent, shared_from_this(), _1));
}
