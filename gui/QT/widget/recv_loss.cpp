#include "recv_loss.hpp"
#include "utils/string_view_utils.hpp"

RecvLossWidget::RecvLossWidget(QWidget *parent) : QProgressBar(parent){
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
}

void RecvLossWidget::parseLine(std::string_view line){
	if(!sv_is_prefix(line, "SSRC "))
	    return;

	auto tmp = line;
	tokenize(tmp, ' '); //"SSRC"
	auto ssrc_sv = tokenize(tmp, ' '); //actual ssrc
	auto packet_counts = tokenize(tmp, ' '); //received/total

	auto received_sv = tokenize(packet_counts, '/');
	auto total_sv = tokenize(packet_counts, '/');

	int received;
	int total;

	if(!parse_num(received_sv, received) || !parse_num(total_sv, total))
	    return;

	float ratio = ((float) received / total) * 100.f;
	setFormat("Received %p% of packets");
	setValue(ratio);
	setToolTip(QString::fromUtf8(line.data(), line.length()));
	timer.start();
}

void RecvLossWidget::timeout(){
	setValue(0);
	setFormat("Nothing received");
	setToolTip("Nothing received");
}
