#ifndef VUMETERWIDGET_HPP
#define VUMETERWIDGET_HPP

#include <QTimer>
#include <QWidget>
#include <QPaintEvent>
#include <memory>

#include <chrono>
#include <future>
#include <thread>

#include "astat.h"

class VuMeterWidget : public QWidget{
	Q_OBJECT
public:
	VuMeterWidget(QWidget *parent) :
		QWidget(parent),
		connection(nullptr, ug_control_connection_done),
		port(8888),
		peak {0.0},
		rms {0.0},
		barLevel {0.0},
		rmsLevel {0.0},
		updatesPerSecond(24)
   	{
		connect(&timer, SIGNAL(timeout()), this, SLOT(updateVal()));
		timer.start(1000/updatesPerSecond);
		//setValue(50);
		ug_control_init();
		connect_ug();
  	}

	~VuMeterWidget(){
		ug_control_cleanup();
	}

protected:
	void paintEvent(QPaintEvent *paintEvent);

private:
	QTimer timer;
	std::unique_ptr<ug_connection, void(*)(ug_connection *)> connection;
	std::future<ug_connection *> future_connection;

	int port;

	static const int num_channels = 2;

	double peak[num_channels];
	double rms[num_channels];

	double barLevel[num_channels];
	double rmsLevel[num_channels];
	int updatesPerSecond;

	bool connected = false;

	void updateVolumes();
	void connect_ug();
	void disconnect_ug();

	std::chrono::system_clock::time_point last_connect;

	static const int meterVerticalPad = 5;
	static const int meterBarPad = 2;
	static constexpr double zeroLevel = -40.0;

	void paintMeter(QPainter&, int x, int y, int width, int height, double peak, double rms);
	void paintScale(QPainter&, int x, int y, int width, int height);

public slots:
	void updateVal();

};

#endif
