#ifndef RECV_REPORT_WIDGET_96127bfd005a
#define RECV_REPORT_WIDGET_96127bfd005a

#include <QWidget>
#include <QProgressBar>
#include <QTimer>

class RecvReportWidget : public QProgressBar{
	Q_OBJECT
public:
	RecvReportWidget(QWidget *parent = nullptr);
	~RecvReportWidget() = default;

	void report(int rtt_usec, float loss);
	void reset();

private slots:
	void timeout();

private:
	QTimer timer;

	const int timeout_msec = 15000;
};

#endif
