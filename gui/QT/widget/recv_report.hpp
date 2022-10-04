#ifndef RECV_REPORT_WIDGET_96127bfd005a
#define RECV_REPORT_WIDGET_96127bfd005a

#include <QWidget>
#include <QProgressBar>

class RecvReportWidget : public QProgressBar{
	Q_OBJECT
public:
	RecvReportWidget(QWidget *parent = nullptr);
	~RecvReportWidget() = default;

	void report(int rtt_usec, float loss);
	void reset();

private:
};

#endif
