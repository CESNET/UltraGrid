/**
 * @file   nat-helper.cpp
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
#include <functional>
#include "nat-helper.hpp"

NatHelper::NatHelper(int port) : port(port),
	io_service(),
	//work_guard(io_service.get_executor()),
	acceptor(io_service),
	pendingSocket(io_service)
{

}

NatHelper::NatHelper() : NatHelper(12558)
{

}


void NatHelper::worker(){
	std::cout << "Running" << std::endl;
	io_service.run();
	std::cout << "End" << std::endl;
}

void NatHelper::run(){
	using namespace std::placeholders;

	asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(), port);
	acceptor.open(endpoint.protocol());
	acceptor.bind(endpoint);
	acceptor.listen(asio::socket_base::max_connections);
	acceptor.set_option(asio::ip::tcp::acceptor::reuse_address(false));
	acceptor.async_accept(pendingSocket,
			std::bind(&NatHelper::onConnectionAccepted, this, _1));

	std::cout << "Starting thread" << std::endl;
	worker_thread = std::thread(&NatHelper::worker, this);
}

void NatHelper::stop(){
	std::cout << "Stopping..." << std::endl;
	io_service.stop();
	worker_thread.join();
}

void NatHelper::onConnectionAccepted(const std::error_code& ec){
	if(ec){
		std::cerr << "Error accepting client connection: " << ec.message() << std::endl;
		return;
	}

	using namespace std::placeholders;

	auto client = std::make_shared<Client>(std::move(pendingSocket));
	client->readDescription(
			std::bind(&NatHelper::onClientDesc, this, _1, _2));
	incomingClients.insert({client.get(), std::move(client)});

	acceptor.async_accept(pendingSocket,
			std::bind(&NatHelper::onConnectionAccepted, this, _1));
}

void NatHelper::cleanEmptyRooms(){
	for(auto it = rooms.begin(); it != rooms.end(); ){
		if(it->second->isEmpty()){
			std::cout << "Removing empty room " << it->first << "\n";
			it = rooms.erase(it);
		} else {
			it++;
		}
	}
}

void NatHelper::onClientDesc(Client& client, bool success){
	if(!success){
		std::cerr << "Error reading client description" << std::endl;
		incomingClients.erase(&client);
		return;
	}

	const auto& roomName = client.getRoomName();
	std::cout << "Moving client " << client.getClientName()
		<< " to room " << roomName
		<< std::endl;

	auto roomIt = rooms.find(roomName);
	if(roomIt == rooms.end()){
		cleanEmptyRooms();
		std::cout << "Creating room " << roomName << std::endl;
		roomIt = rooms.insert({roomName, std::make_shared<Room>(roomName)}).first;
	}

	auto clientIt = incomingClients.find(&client);
	auto& room = roomIt->second;
	if(room->isFull()){
		std::cerr << "Room " << roomName << " is full, dropping client" << std::endl;
	} else {
		roomIt->second->addClient(std::move(clientIt->second));
	}
	incomingClients.erase(clientIt);
}
