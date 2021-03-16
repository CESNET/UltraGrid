#include <algorithm>
#include <cmath>
#include <QPainter>
#include <QBrush>
#include <QLinearGradient>
#include <QRect>
#include <QString>
#include <QStyle>
#include "vuMeterWidget.hpp"

constexpr double VuMeterWidget::zeroLevel;

void VuMeterWidget::updateVal(){
	updateVolumes();

	const double fallSpeed = 200.0;

	for(int i = 0; i < num_channels; i++){

		barLevel[i] = std::max(barLevel[i] - fallSpeed / updatesPerSecond, 0.0);
		rmsLevel[i] = std::max(rmsLevel[i] - fallSpeed / updatesPerSecond, 0.0);

#if 1
		double newPeakHeight = 100 - std::max(peak[i], zeroLevel) * (100 / zeroLevel);
		double newRmsHeight = 100 - std::max(rms[i], zeroLevel) * (100 / zeroLevel);
#else
		double newHeight = pow(10.0, peak[i] / 20.0) * 100;
#endif
		barLevel[i] = std::max(barLevel[i], newPeakHeight);
		rmsLevel[i] = std::max(rmsLevel[i], newRmsHeight);
	}


	update();
}

void VuMeterWidget::updateVolumes(){
	ug_connection *ug_c = connection.get();
	if(ug_c == nullptr){
		connect_ug();
		return;
	}

	int count; 
	bool ret = ug_control_get_volumes(ug_c, peak, rms, &count);

	connected = ret;
	setToolTip(connected ? "" : "Unable to read volume info from UG");

	if(!ret){
		last_connect = std::chrono::system_clock::now();
		connect_ug();
	}
}

void VuMeterWidget::paintMeter(QPainter& painter,
		int x,
		int y,
		int width,
		int height,
		double peak,
		double rms)
{
	const int in_width = width - meterBarPad * 2;
	const int in_full_height = height - meterBarPad * 2;

	int barHeight = in_full_height * (peak / 100);
	int rmsHeight = in_full_height * (rms / 100);

	QLinearGradient gradient(0, 0, in_width, in_full_height);
	gradient.setColorAt(0, Qt::red);
	gradient.setColorAt(0.25, Qt::yellow);
	gradient.setColorAt(0.5, Qt::green);
	gradient.setColorAt(1, Qt::green);
	QBrush brush(gradient);

	painter.fillRect(x, y, width, height, connected ? Qt::black : Qt::gray);
	if(connected){
		painter.fillRect(x + meterBarPad,
				y + meterBarPad + (in_full_height - barHeight),
				in_width, barHeight,
				brush);
	}


	int pos = y + meterBarPad + (in_full_height - rmsHeight);
	painter.setPen(Qt::red);
	painter.drawLine(x, pos, x + width, pos);
}

void VuMeterWidget::paintScale(QPainter& painter,
		int x,
		int y,
		int width,
		int height)
{
	const int barHeight = height - meterVerticalPad * 2 - meterBarPad * 2;	
	const int barStart = y + meterVerticalPad + meterBarPad;
	const float stepPx = (float) barHeight / std::fabs(zeroLevel);

	painter.setPen(connected ? Qt::black : Qt::gray);
	int i = 0;
	for(float y = barStart; y <= barStart + barHeight + 0.1; y += stepPx){
		static const int lineLenghtSmall = 2;
		static const int lineLenghtMid = 4;
		static const int lineLenghtLarge = 6;
		static const int drawThreshold = 4;
		if(i % 10 == 0){
			painter.drawLine(x, y, x + lineLenghtLarge, y);
			painter.drawLine(x + width - lineLenghtLarge, y, x + width, y);
			painter.drawText(QRect(x + width / 2 - 8, y - 6, 16, 12), Qt::AlignCenter,QString::number(i));
		} else if(i % 5 == 0){
			painter.drawLine(x, y, x + lineLenghtMid, y);
			painter.drawLine(x + width - lineLenghtMid, y, x + width, y);
		} else if(stepPx >= drawThreshold){
			painter.drawLine(x, y, x + lineLenghtSmall, y);
			painter.drawLine(x + width - lineLenghtSmall, y, x + width, y);
		}

		i++;
	}

	if(!connected){
		int iconDim = 32;
		QIcon icon = style()->standardIcon(QStyle::SP_MessageBoxWarning);
		QPixmap pixmap = icon.pixmap(iconDim,iconDim);
		painter.eraseRect(x+width/2 - (iconDim+4)/2, y + height / 2 - (iconDim+4)/2,
				iconDim+4, iconDim+4);
		painter.drawPixmap(x+width/2 - iconDim/2, y + height / 2 - iconDim/2,
				iconDim, iconDim,
				pixmap,
				0, 0,
				iconDim, iconDim);
	}
}

void VuMeterWidget::paintEvent(QPaintEvent * /*paintEvent*/){
	const int meter_width = 10;

	QPainter painter(this);

	paintMeter(painter,
			0,
			meterVerticalPad,
			meter_width,
			height() - meterVerticalPad * 2,
			barLevel[0],
			rmsLevel[0]);
	paintMeter(painter,
			width() - meter_width,
			meterVerticalPad,
			meter_width,
			height() - meterVerticalPad * 2,
			barLevel[1],
			rmsLevel[1]);

	paintScale(painter,
			meter_width,
			0,
			width() - meter_width * 2,
			height());

}

ug_connection *connectLoop(int port){
	const int max_tries = 2;
	int tries = 0;

	ug_connection *connection = nullptr;
	connection = ug_control_connection_init(port);
	while(!connection && tries < max_tries){
		std::this_thread::sleep_for (std::chrono::seconds(2));
		connection = ug_control_connection_init(port);
		tries++;
	}

	return connection;
}

void VuMeterWidget::connect_ug(){
	disconnect_ug();

	ug_connection *c = nullptr;

	if(future_connection.valid()){
		std::future_status status;
		status = future_connection.wait_for(std::chrono::seconds(0));
		if(status == std::future_status::ready){
			c = future_connection.get();
		}
	} else {
		future_connection = std::async(std::launch::async, connectLoop, port);
	}

	if(!c)
		return;

	connection = std::unique_ptr<ug_connection, void(*)(ug_connection *)>(c, ug_control_connection_done);
}

void VuMeterWidget::disconnect_ug(){
	connection = std::unique_ptr<ug_connection, void(*)(ug_connection *)>(nullptr, ug_control_connection_done);
}
