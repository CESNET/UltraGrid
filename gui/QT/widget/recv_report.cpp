#include <QString>
#include "recv_report.hpp"
#include "utils/string_view_utils.hpp"

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

void RecvReportWidget::parseLine(std::string_view line){
	if(!sv_is_prefix(line, "Receiver of 0x"))
		return;

	auto tmp = std::string(line);
	int ssrc = 0;
	int rtt = 0;
	float loss = 0;
	sscanf(tmp.c_str(), "Receiver of 0x%08x reports RTT=%d usec, loss %f%%", &ssrc, &rtt, &loss);

	report(rtt, loss);
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
