#include <stdio.h>
#include <iostream>
#include <QHostAddress>

#include "control_port.hpp"

ControlPort::ControlPort(){
	QObject::connect(&socket, &QTcpSocket::readyRead, this, &ControlPort::readReady);

	QObject::connect(&socket, &QTcpSocket::connected, this, &ControlPort::connected);
	QObject::connect(&socket, &QTcpSocket::disconnected, this, &ControlPort::disconnected);
}

void ControlPort::connect(int local_port){
	socket.connectToHost(QHostAddress::LocalHost, local_port);
}

void ControlPort::parseLine(std::string_view line){
	constexpr std::string_view prefix = "Control socket listening on port";
	if(line.substr(0, prefix.length()) != prefix)
		return;

	auto tmp = std::string(line.substr(prefix.length()));
	int port = 0;
	sscanf(tmp.c_str(), " %d", &port);

	connect(port);
}

QAbstractSocket::SocketState ControlPort::getState() const{
	return socket.state();
}

void ControlPort::writeLine(std::string_view line){
	int toWrite = line.length();
	const char *data = line.data();
	while(toWrite > 0){
		auto ret = socket.write(data, toWrite);
		if(ret < 0){
			break;
		}
		data += ret;
		toWrite -= ret;
	}
}

void ControlPort::readReady(){
	char buf[512];

	while(true){
		int ret = socket.read(buf, sizeof(buf));
		if(ret == 0 || ret == -1)
			break;

		lineBuf.write(std::string_view(buf, ret));

		while(lineBuf.hasLine()){
			for(auto& c : lineCallbacks){
				c(lineBuf.peek());
			}
			lineBuf.pop();
		}
	}
}

