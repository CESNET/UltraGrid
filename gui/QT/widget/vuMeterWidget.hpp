#ifndef VUMETERWIDGET_HPP
#define VUMETERWIDGET_HPP

#include <QTimer>
#include <QWidget>
#include <QPaintEvent>
#include <memory>
#include <atomic>
#include <chrono>
#include <string_view>

class ControlPort;

class VuMeterWidget : public QWidget{
	Q_OBJECT
public:
	VuMeterWidget(QWidget *parent);

	void parseLine(std::string_view line);
	void setControlPort(ControlPort *controlPort);
	void setParsePrefix(std::string_view prefix);

protected:
	void paintEvent(QPaintEvent *paintEvent);

private:
	static const int max_channels = 8;
	std::string parsePrefix = "stats ARECV";

	int channels = 0;
	double peak[max_channels];
	double rms[max_channels];

	double barLevel[max_channels];
	double rmsLevel[max_channels];

	QTimer timer;
	int updatesPerSecond;
	using clock = std::chrono::steady_clock;
	clock::time_point lastUpdate;

	ControlPort *controlPort = nullptr;
	bool connected = false;

	void paintMeter(QPainter&, int x, int y, int width, int height, double peak, double rms);
	void paintScale(QPainter&, int x, int y, int width, int height);

public slots:
	void updateVal();

private slots:
	void onControlPortConnect();
};

#endif
