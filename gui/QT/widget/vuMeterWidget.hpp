#ifndef VUMETERWIDGET_HPP
#define VUMETERWIDGET_HPP

#include <QTimer>
#include <QWidget>
#include <QPaintEvent>
#include <memory>
#include <atomic>

#include <future>

struct ug_connection;
struct ug_connection_deleter{ void operator()(ug_connection *c); };
using unique_ug_connection = std::unique_ptr<ug_connection, ug_connection_deleter>;

class VuMeterWidget : public QWidget{
	Q_OBJECT
public:
	VuMeterWidget(QWidget *parent);
	~VuMeterWidget();

	void setPort(int port);

protected:
	void paintEvent(QPaintEvent *paintEvent);

private:
	QTimer timer;
	unique_ug_connection connection;
	std::future<ug_connection *> future_connection;
	std::atomic<bool> cancelConnect = false;

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

	void paintMeter(QPainter&, int x, int y, int width, int height, double peak, double rms);
	void paintScale(QPainter&, int x, int y, int width, int height);

public slots:
	void updateVal();

};

#endif
