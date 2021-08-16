/**
 * @file   room.cpp
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
#include <functional>
#include <iostream>
#include "room.hpp"

Room::Room(std::string name) : name(std::move(name)) {  }

void Room::addClient(std::shared_ptr<Client>&& client){
	assert(!isFull());

	Client *clientPtr = client.get();

	for(auto& [c, _] : clients){
		c->sendMsg(clientPtr->getClientName());
		c->sendMsg(clientPtr->getSdpDesc());

		clientPtr->sendMsg(c->getClientName());
		clientPtr->sendMsg(c->getSdpDesc());
		for(const auto& candidate : c->getCandidates()){
			clientPtr->sendMsg(candidate);
		}
	}

	clients.insert({clientPtr, std::move(client)});

	using namespace std::placeholders;
	clientPtr->readCandidate(
			std::bind(&Room::onClientCandidate, shared_from_this(), _1, _2));

}

void Room::onClientCandidate(Client& client, bool success){
	if(!success){
		std::cerr << "Error reading candidate, removing client "
			<< client.getClientName() << std::endl;

		clients.erase(&client);
		return;
	}

	std::cout << "Client candidate recieved" << std::endl;

	for(auto& [c, cu] : clients){
		if(&client == c)
			continue;

		c->sendMsg(client.getCandidates().back());
	}

	using namespace std::placeholders;
	client.readCandidate(
			std::bind(&Room::onClientCandidate, shared_from_this(), _1, _2));
}
