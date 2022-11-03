#include "recv_loss.hpp"
#include "utils/string_view_utils.hpp"

RecvLossWidget::RecvLossWidget(QWidget *parent) : QProgressBar(parent){
	elapsedTimer.start();
	reset();
	QObject::connect(&timer, &QTimer::timeout, this, &RecvLossWidget::timeout);
	timer.setInterval(timeout_msec);
}

void RecvLossWidget::reset(){
	timer.stop();
	setRange(0, 100);
	setValue(0);
	setFormat("Nothing received");
	setTextVisible(true);
	setToolTip("Nothing received");

	reports.clear();
}

void RecvLossWidget::parseLine(std::string_view line){
	if(!sv_is_prefix(line, "SSRC "))
		return;

	auto tmp = line;
	tokenize(tmp, ' '); //"SSRC"
	auto ssrc_sv = tokenize(tmp, ' '); //actual ssrc
	if(ssrc_sv.substr(0, 2) == "0x")
		ssrc_sv.remove_prefix(2);
	auto packet_counts = tokenize(tmp, ' '); //received/total

	auto received_sv = tokenize(packet_counts, '/');
	auto total_sv = tokenize(packet_counts, '/');

	int ssrc;
	int received;
	int total;

	if(!parse_num(received_sv, received)
			|| !parse_num(total_sv, total)
			|| !parse_num(ssrc_sv, ssrc, 16))
		return;

	addReport(ssrc, received, total);
	updateVal();
}

void RecvLossWidget::addReport(int ssrc, int received, int total){
	SSRC_report *rep = nullptr;
	for(auto& r : reports){
		if(r.ssrc == ssrc){
			rep = &r;
		}
	}

	if(!rep){
		SSRC_report newRep;
		newRep.ssrc = ssrc;
		reports.push_back(newRep);
		rep = &reports.back();
	}

	rep->received = received;
	rep->total = total;
	rep->lastReportMs = elapsedTimer.elapsed();
}

void RecvLossWidget::updateVal(){
	int received = 0;
	int total = 0;

	auto now = elapsedTimer.elapsed();
	auto endIt = std::remove_if(reports.begin(), reports.end(),
			[now](const SSRC_report& r){ return now - r.lastReportMs > timeout_msec; });

	reports.erase(endIt, reports.end());

	for(auto& r : reports){
		received += r.received;
		total += r.total;
	}

	float ratio = ((float) received / total) * 100.f;
	setFormat("Received %p% of packets");
	setValue(ratio);
	//setToolTip(QString::fromUtf8(line.data(), line.length()));
	timer.start();
}

void RecvLossWidget::timeout(){
	setValue(0);
	setFormat("Nothing received");
	setToolTip("Nothing received");
}
