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

	void setOnRightSide(bool side) { onRightSide = side; }

protected:
	void paintEvent(QPaintEvent *paintEvent);

private:
	using clock = std::chrono::steady_clock;
	struct PeakMaximum {
		double val = 0;
		clock::time_point ts;
	};

	static const int max_channels = 8;
	std::string parsePrefix = "stats ARECV";

	int channels = 0;
	double peak[max_channels];
	double rms[max_channels];

	double barLevel[max_channels];
	double rmsLevel[max_channels];
	PeakMaximum maximumLevel[max_channels];

	bool onRightSide = false;

	QTimer timer;
	int updatesPerSecond;
	clock::time_point lastUpdate;

	ControlPort *controlPort = nullptr;
	bool connected = false;

	void paintMeter(QPainter&, int x, int y, int width, int height, double peak, double rms, double maxPeak);
	void paintScale(QPainter&, int x, int y, int width, int height, bool leftTicks, bool rightTicks);

public slots:
	void updateVal();

private slots:
	void onControlPortConnect();
};

#endif
