#ifndef RECV_REPORT_WIDGET_96127bfd005a
#define RECV_REPORT_WIDGET_96127bfd005a

#include <QWidget>
#include <QProgressBar>
#include <QTimer>
#include <QElapsedTimer>

#include <string_view>

#include "ssrc_container.hpp"

class RecvReportWidget : public QProgressBar{
	Q_OBJECT
public:
	RecvReportWidget(QWidget *parent = nullptr);
	~RecvReportWidget() = default;

	void updateVal();
	void parseLine(std::string_view line);
	void reset();

private slots:
	void timeout();

private:
	QTimer timer;
	QElapsedTimer elapsedTimer;

	struct RecvReport{
		int lost;
		int total;
		int rtt_usec;
	};
	SSRC_container<RecvReport, decltype(elapsedTimer.elapsed())> reports;

	const int timeout_msec = 15000;
};

#endif
