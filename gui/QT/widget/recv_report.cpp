#include <QString>
#include "recv_report.hpp"

RecvReportWidget::RecvReportWidget(QWidget *parent) : QProgressBar(parent){
	reset();
	timer.callOnTimeout(this, &RecvReportWidget::timeout);
	timer.setInterval(timeout_msec);
}

void RecvReportWidget::reset(){
	timer.stop();
	setRange(0, 100);
	setValue(0);
	setFormat("No reciever report");
	setTextVisible(true);
	setToolTip("No RTCP reciever report. Note that this does not necessarily mean"
			" that the reciever isn't recieving.");
}

void RecvReportWidget::report(int rtt_usec, float loss){
	setFormat("Receiver reports %p% received");
	setValue(100.f - loss);
	setToolTip("RTT = " + QString::number(rtt_usec) + " usec");
	timer.start();
}

