#include <QString>
#include "recv_report.hpp"

RecvReportWidget::RecvReportWidget(QWidget *parent) : QProgressBar(parent){
	reset();
	QObject::connect(&timer, &QTimer::timeout, this, &RecvReportWidget::timeout);
	timer.setInterval(timeout_msec);
}

void RecvReportWidget::reset(){
	timer.stop();
	setRange(0, 100);
	setValue(0);
	setFormat("No receiver report");
	setTextVisible(true);
	setToolTip("No RTCP receiver report. Note that this does not necessarily mean"
			" that the receiver isn't receiving.");
}

void RecvReportWidget::report(int rtt_usec, float loss){
	setFormat("Receiver reports %p% received");
	setValue(100.f - loss);
	setToolTip("RTT = " + QString::number(rtt_usec) + " usec");
	timer.start();
}

void RecvReportWidget::timeout(){
	setValue(0);
	setFormat("Receiver report timed out");
	setToolTip("Receiver reports stopped arriving");
}
