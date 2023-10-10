#include <algorithm>
#include <cassert>
#include <cmath>
#include <thread>
#include <functional>
#include <QPainter>
#include <QBrush>
#include <QLinearGradient>
#include <QRect>
#include <QString>
#include <QStyle>
#include "vuMeterWidget.hpp"
#include "control_port.hpp"
#include "utils/string_view_utils.hpp"
#include "astat.h"

namespace{

	static constexpr int meterVerticalPad = 5;
	static constexpr int meterBarPad = 2;
	static constexpr double zeroLevel = -40.0;

}//anon namespace

VuMeterWidget::VuMeterWidget(QWidget *parent) :
	QWidget(parent),
	peak {0.0},
	rms {0.0},
	barLevel {0.0},
	rmsLevel {0.0},
	updatesPerSecond(24)
{
	connect(&timer, SIGNAL(timeout()), this, SLOT(updateVal()));
	timer.start(1000/updatesPerSecond);
}

void VuMeterWidget::parseLine(std::string_view line){
	if(!sv_is_prefix(line, parsePrefix))
		return;

	line.remove_prefix(parsePrefix.size());

	for(channels = 0; !line.empty() && channels < max_channels; channels++){
		auto rmsTag = tokenize(line, ' ');
		if(!sv_is_prefix(rmsTag, "volrms"))
			return;
		auto rms_sv = tokenize(line, ' ');

		auto peakTag = tokenize(line, ' ');
		if(!sv_is_prefix(peakTag, "volpeak"))
			return;
		auto peak_sv = tokenize(line, ' ');

		double ch_rms = 0;
		double ch_peak = 0;
		if(!parse_num(rms_sv, ch_rms)
				|| !parse_num(peak_sv, ch_peak))
		{
			return;
		}

		peak[channels] = ch_peak;
		rms[channels] = ch_rms;
	}

	lastUpdate = clock::now();
}

void VuMeterWidget::setControlPort(ControlPort *controlPort){
	this->controlPort = controlPort;

	using namespace std::placeholders;
	controlPort->addLineCallback(std::bind(&VuMeterWidget::parseLine, this, _1));
	connect(controlPort, &ControlPort::connected, this, &VuMeterWidget::onControlPortConnect);
}

void VuMeterWidget::setParsePrefix(std::string_view prefix){
	parsePrefix = prefix;
}

void VuMeterWidget::updateVal(){
	const double fallSpeed = 200.0;

	connected = controlPort->getState() == QAbstractSocket::ConnectedState;

	for(int i = 0; i < channels; i++){

		barLevel[i] = std::max(barLevel[i] - fallSpeed / updatesPerSecond, 0.0);
		rmsLevel[i] = std::max(rmsLevel[i] - fallSpeed / updatesPerSecond, 0.0);

		double newPeakHeight = 100 - std::max(peak[i], zeroLevel) * (100 / zeroLevel);
		double newRmsHeight = 100 - std::max(rms[i], zeroLevel) * (100 / zeroLevel);

		barLevel[i] = std::max(barLevel[i], newPeakHeight);
		rmsLevel[i] = std::max(rmsLevel[i], newRmsHeight);
	}

	update();
}

void VuMeterWidget::onControlPortConnect(){
	assert(controlPort);
	controlPort->writeLine("stats on\r\n");
	lastUpdate = clock::time_point();
}

void VuMeterWidget::paintMeter(QPainter& painter,
		int x,
		int y,
		int width,
		int height,
		double peak,
		double rms)
{
	const int meterBarPadHorizontal = 1;
	const int in_width = width - meterBarPadHorizontal * 2;
	const int in_full_height = height - meterBarPad * 2;

	int barHeight = in_full_height * (peak / 100);
	int rmsHeight = in_full_height * (rms / 100);

	QLinearGradient gradient(0, 0, in_width, in_full_height);
	gradient.setColorAt(0, Qt::red);
	gradient.setColorAt(0.25, Qt::yellow);
	gradient.setColorAt(0.5, Qt::green);
	gradient.setColorAt(1, Qt::green);
	QBrush brush(gradient);

	using namespace std::chrono_literals;
	bool enabled = connected && (clock::now() - lastUpdate <= 2s);

	painter.fillRect(x, y + 1, width, height - 2, enabled ? Qt::black : Qt::gray);
	if(!enabled)
		return;

	painter.fillRect(x + meterBarPadHorizontal,
			y + meterBarPad + (in_full_height - barHeight),
			in_width, barHeight,
			brush);
	int pos = y + meterBarPad + (in_full_height - rmsHeight);
	QPen rmsPen(Qt::blue);
	rmsPen.setWidth(4);
	painter.setPen(rmsPen);
	painter.drawLine(x + meterBarPadHorizontal, pos, x + in_width, pos);
}

void VuMeterWidget::paintScale(QPainter& painter,
		int x,
		int y,
		int width,
		int height,
		bool leftTicks,
		bool rightTicks)
{
	const int barHeight = height - meterVerticalPad * 2 - meterBarPad * 2;	
	const int barStart = y + meterVerticalPad + meterBarPad;
	const float stepPx = (float) barHeight / std::fabs(zeroLevel);

	painter.setPen(connected ? Qt::black : Qt::gray);
	int i = 0;

	int textPosX;
	if(leftTicks && rightTicks){
		textPosX = x + width / 2 - 8;
	} else if(leftTicks){
		textPosX = x + width - 16;
	} else {
		textPosX = x;
	}

	for(float y = barStart; y <= barStart + barHeight + 0.1; y += stepPx){
		static const int lineLenghtSmall = 2;
		static const int lineLenghtMid = 4;
		static const int lineLenghtLarge = 6;
		static const int drawThreshold = 4;
		if(i % 10 == 0){
			if(leftTicks) painter.drawLine(x, y, x + lineLenghtLarge, y);
			if(rightTicks) painter.drawLine(x + width - lineLenghtLarge, y, x + width, y);
			painter.drawText(QRect(textPosX, y - 6, 16, 12), Qt::AlignCenter,QString::number(i));
		} else if(i % 5 == 0){
			if(leftTicks) painter.drawLine(x, y, x + lineLenghtMid, y);
			if(rightTicks) painter.drawLine(x + width - lineLenghtMid, y, x + width, y);
		} else if(stepPx >= drawThreshold){
			if(leftTicks) painter.drawLine(x, y, x + lineLenghtSmall, y);
			if(rightTicks) painter.drawLine(x + width - lineLenghtSmall, y, x + width, y);
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
	const int scaleWidth = 30;

	const int meterCount = std::max(1, channels);

	const int meterWidth = (width() - scaleWidth) / meterCount;

	QPainter painter(this);

	if(meterCount % 2 != 0){
		int scaleX = onRightSide ? 0 : meterCount * meterWidth;
		int metersStartX = onRightSide ? scaleWidth : 0;

		for(int i = 0; i < meterCount; i++){
			paintMeter(painter,
					metersStartX + i * meterWidth,
					meterVerticalPad,
					meterWidth,
					height() - meterVerticalPad * 2,
					barLevel[i],
					rmsLevel[i]);
		}

		paintScale(painter,
				scaleX,
				0,
				scaleWidth,
				height(),
				!onRightSide,
				onRightSide);
	} else {
		int i = 0;
		for(; i < meterCount / 2; i++){
			paintMeter(painter,
					i * meterWidth,
					meterVerticalPad,
					meterWidth,
					height() - meterVerticalPad * 2,
					barLevel[i],
					rmsLevel[i]);
		}

		paintScale(painter,
				i * meterWidth,
				0,
				scaleWidth,
				height(),
				true,
				true);

		for(; i < meterCount; i++){
			paintMeter(painter,
					i * meterWidth + scaleWidth,
					meterVerticalPad,
					meterWidth,
					height() - meterVerticalPad * 2,
					barLevel[i],
					rmsLevel[i]);
		}

	}

}
