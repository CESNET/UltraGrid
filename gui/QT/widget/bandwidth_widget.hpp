#ifndef BANDWIDTH_WIDGET_c035f927c51f
#define BANDWIDTH_WIDGET_c035f927c51f

#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QElapsedTimer>

#include <string_view>

#include "ssrc_container.hpp"

class BandwidthWidget : public QLabel{
	Q_OBJECT
public:
	BandwidthWidget(QWidget *parent = nullptr);
	~BandwidthWidget() = default;

	void updateVal();
	void parseLine(std::string_view line);
	void reset();

private slots:
	void timeout();

private:
	void report(long long bytes_per_second);

	QTimer timer;
	QElapsedTimer elapsedTimer;

	SSRC_container<long long, decltype(elapsedTimer.elapsed())> reports;

	const int timeout_msec = 2500;
};

#endif //BANDWIDTH_WIDGET_c035f927c51f
