#ifndef RECV_LOSS_WIDGET_fcb5ae52ae4c
#define RECV_LOSS_WIDGET_fcb5ae52ae4c

#include <QWidget>
#include <QProgressBar>
#include <QTimer>

#include <vector>
#include <QElapsedTimer>
#include <string_view>

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
		int ssrc = 0;
		int received = 0;
		int total = 0;

		qint64 lastReportMs = 0;
	};
	std::vector<SSRC_report> reports;

	void addReport(int ssrc, int received, int total);
	void updateVal();
};

#endif //RECV_LOSS_WIDGET_fcb5ae52ae4c
