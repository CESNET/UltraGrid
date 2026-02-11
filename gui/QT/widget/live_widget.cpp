#include "live_widget.hpp"

#include <QPainter>

constexpr auto label = "LIVE";

LiveWidget::LiveWidget(QWidget *parent) :
	QWidget(parent),
	liveText(label)
{
	QFontMetrics fm = fontMetrics();
	boundingRect = fm.boundingRect(liveText);
	boundingRect.setWidth(boundingRect.width() + boundingRect.height()); //Add space for dot

	setMinimumSize(boundingRect.width(), boundingRect.height());
}

void LiveWidget::setLive(bool live){
	this->live = live;
	update();
}

void LiveWidget::paintEvent(QPaintEvent *event){
	QPainter painter(this);


	if(live){
		painter.setBrush(Qt::red);
		painter.setPen(Qt::red);
	} else{
		painter.setPen(Qt::gray);
		auto font = painter.font();
		font.setStrikeOut(true);
		painter.setFont(font);
	}
	painter.drawEllipse(QPointF(boundingRect.height() / 2.f, boundingRect.height() / 2.f), boundingRect.height() / 4.f, boundingRect.height() / 4.f);
	painter.drawText(QRect(0, 0, boundingRect.width(), boundingRect.height()), Qt::AlignRight, liveText);
}
