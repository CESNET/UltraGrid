#ifndef RECV_LOSS_WIDGET_fcb5ae52ae4c
#define RECV_LOSS_WIDGET_fcb5ae52ae4c

#include <QWidget>
#include <QProgressBar>
#include <QTimer>

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
	QTimer timer;

	const int timeout_msec = 15000;
};

#endif //RECV_LOSS_WIDGET_fcb5ae52ae4c
