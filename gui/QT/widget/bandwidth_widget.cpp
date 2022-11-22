#include <stdio.h>
#include "bandwidth_widget.hpp"
#include "utils/string_view_utils.hpp"


BandwidthWidget::BandwidthWidget(QWidget *parent) : QLabel(parent){
	elapsedTimer.start();
	reset();
	QObject::connect(&timer, &QTimer::timeout, this, &BandwidthWidget::timeout);
	timer.setInterval(timeout_msec);
}

void BandwidthWidget::updateVal(){
	reports.remove_timed_out(timeout_msec, elapsedTimer.elapsed());

	long long total = 0;
	for(const auto& i : reports.get()){
		total += i.item;
	}

	report(total);

	timer.start();
}

void BandwidthWidget::parseLine(std::string_view line){
	static constexpr std::string_view prefix = "stats tx_send";
	if(!sv_is_prefix(line, prefix))
		return;

	line.remove_prefix(prefix.size());

	auto ssrc_sv = tokenize(line, ' ');
	if(ssrc_sv.substr(0, 2) == "0x")
		ssrc_sv.remove_prefix(2);
	auto type = tokenize(line, ' ');
	auto bytes_sv = tokenize(line, ' ');

	long long bytes = 0;
	uint32_t ssrc = 0;
	if(!parse_num(bytes_sv, bytes)
			|| !parse_num(ssrc_sv, ssrc, 16))
	{
		return;
	}

	long long num = 0;

	for(const auto& r : reports.get()){
		if(r.ssrc != ssrc)
			continue;

		num = bytes * 1000 / (elapsedTimer.elapsed() - r.timestamp);
	}

	reports.insert(ssrc, num, elapsedTimer.elapsed());
	updateVal();
}

void BandwidthWidget::reset(){
	reports.clear();
	report(0);
	timer.stop();
}

void BandwidthWidget::timeout(){
	reset();
}

void BandwidthWidget::report(long long bytes_per_second){
	static constexpr std::string_view units = "kMG";

	float num = bytes_per_second;
	char unit = ' ';

	for(unsigned i = 0; num > 1000 && i < units.size(); i++){
		unit = units[i];
		num /= 1000;
	}

	setText(QString::number(num, 'f', 2) + " " + unit + "Bps");
}
