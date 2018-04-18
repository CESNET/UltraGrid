#ifndef VUMETERWIDGET_HPP
#define VUMETERWIDGET_HPP

#include <QTimer>
#include <QProgressBar>
#include <memory>

#include <chrono>

#include "astat.h"

class VuMeterWidget : public QProgressBar{
	Q_OBJECT
public:
	VuMeterWidget(QWidget *parent) :
		QProgressBar(parent),
		connection(nullptr, ug_control_connection_done),
		port(8888),
		barLevel(0.0),
		updatesPerSecond(24)
   	{
		connect(&timer, SIGNAL(timeout()), this, SLOT(updateVal()));
		timer.start(1000/updatesPerSecond);
		setValue(50);
		connect_ug();
  	}

private:
	QTimer timer;
	std::unique_ptr<ug_connection, void(*)(ug_connection *)> connection;

	int port;

	double peak[2];
	double rms[2];
	int count;

	double barLevel;
	int updatesPerSecond;

	void updateVolumes();
	void connect_ug();
	void disconnect_ug();

	std::chrono::system_clock::time_point last_connect;

public slots:
	void updateVal();

};

#endif
