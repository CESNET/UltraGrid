#ifndef RECV_LOSS_WIDGET_fcb5ae52ae4c
#define RECV_LOSS_WIDGET_fcb5ae52ae4c

#include <QWidget>
#include <QProgressBar>
#include <QTimer>

#include <QElapsedTimer>
#include <string_view>

#include "ssrc_container.hpp"

class RecvLossWidget : public QProgressBar{
Q_OBJECT
public:
	RecvLossWidget(QWidget *parent = nullptr);
	~RecvLossWidget() = default;

	void parseLine(std::string_view);
	void reset();

private slots:
	void timeout();

private:
	static const int timeout_msec = 15000;
	QTimer timer;
	QElapsedTimer elapsedTimer;

	struct SSRC_report{
		int received = 0;
		int total = 0;
	};
	SSRC_container<SSRC_report, decltype(elapsedTimer.elapsed())> reports;

	void addReport(unsigned ssrc, int received, int total);
	void updateVal();
};

#endif //RECV_LOSS_WIDGET_fcb5ae52ae4c
