#include <QString>
#include "recv_report.hpp"
#include "utils/string_view_utils.hpp"

RecvReportWidget::RecvReportWidget(QWidget *parent) : QProgressBar(parent){
	elapsedTimer.start();
	reset();
	QObject::connect(&timer, &QTimer::timeout, this, &RecvReportWidget::timeout);
	timer.setInterval(timeout_msec);
}

void RecvReportWidget::reset(){
	reports.clear();
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
	int packets;
	sscanf(tmp.c_str(), "Receiver of 0x%08x reports RTT=%d usec, loss %f%% (out of %d)", &ssrc, &rtt, &loss, &packets);

	RecvReport report;
	report.total = packets;
	report.lost = packets * (loss / 100.f);
	report.rtt_usec = rtt;
	
	reports.insert(ssrc, report, elapsedTimer.elapsed());
	updateVal();
}

void RecvReportWidget::updateVal(){
	reports.remove_timed_out(timeout_msec, elapsedTimer.elapsed());

	int total = 0;
	int lost = 0;

	QString tooltip;
	for(const auto& r : reports.get()){
		total += r.item.total;
		lost += r.item.lost;
		tooltip += QString::number(r.ssrc, 16) +
			": RTT=" + QString::number(r.item.rtt_usec) + "us\n";
	}

	if(total < 1) total = 1;

	float loss = (static_cast<float>(lost) / total) * 100.f;

	setFormat("Receiver reports %p% received");
	setValue(100.f - loss);
	setToolTip(tooltip);
	timer.start();
}

void RecvReportWidget::timeout(){
	setValue(0);
	setFormat("Receiver report timed out");
	setToolTip("Receiver reports stopped arriving");
}
