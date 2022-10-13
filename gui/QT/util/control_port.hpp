#ifndef CONTROL_PORT_7a66c3f43b5f
#define CONTROL_PORT_7a66c3f43b5f

#include <vector>
#include <thread>
#include <string_view>
#include <functional>
#include <QTcpSocket>
#include "line_buffer.hpp"

class Socket;

class ControlPort : public QObject{
	Q_OBJECT
public:
	ControlPort();
	~ControlPort() = default;

	void connect(int local_port);
	void parseLine(std::string_view line);

	void writeLine(std::string_view line);

	QAbstractSocket::SocketState getState() const;

	using LineCallback = std::function<void(std::string_view)>;
	void addLineCallback(LineCallback callback) { lineCallbacks.push_back(callback); }

signals:
	void connected();
	void disconnected();

private slots:
	void readReady();

private:
	QTcpSocket socket;
	LineBuffer lineBuf;
	std::vector<LineCallback> lineCallbacks;
};

#endif
