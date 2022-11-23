#include <stdio.h>
#include "bandwidth_widget.hpp"
#include "utils/string_view_utils.hpp"

namespace {
QString getBitrateStr(long long bitsPerSecond){
	static constexpr std::string_view units = "kMG";

	float num = bitsPerSecond;
	char unit = ' ';

	for(unsigned i = 0; num > 1000 && i < units.size(); i++){
		unit = units[i];
		num /= 1000;
	}

	return QString::number(num, 'f', 2) + " " + unit + "bps";
}
}

BandwidthWidget::BandwidthWidget(QWidget *parent) : QLabel(parent){
	elapsedTimer.start();
	reset();
	QObject::connect(&timer, &QTimer::timeout, this, &BandwidthWidget::timeout);
	timer.setInterval(timeout_msec);
}

void BandwidthWidget::updateVal(){
	reports.remove_timed_out(timeout_msec, elapsedTimer.elapsed());

	if(!isEnabled()){
		setText("-- bps");
		setToolTip("");
		return;
	}

	long long total = 0;
	QString tooltip;
	for(const auto& i : reports.get()){
		total += i.item.bitsPerSecond;
		tooltip += QString::number(i.ssrc, 16)
			+ " (" + QString::fromStdString(i.item.type) + "): "
			+ getBitrateStr(i.item.bitsPerSecond) + "\n";
	}

	if(tooltip.endsWith('\n'))
		tooltip.chop(1);

	setToolTip(tooltip);

	setText(getBitrateStr(total));

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

	BandwidthReport report = {};
	report.bitsPerSecond = 0;
	report.type = type;

	for(const auto& r : reports.get()){
		if(r.ssrc != ssrc)
			continue;

		auto duration = std::max<qint64>(elapsedTimer.elapsed() - r.timestamp, 1);
		report.bitsPerSecond = bytes * 8 * 1000 / duration;
	}

	reports.insert(ssrc, report, elapsedTimer.elapsed());
	updateVal();
}

void BandwidthWidget::reset(){
	reports.clear();
	updateVal();
	timer.stop();
}

void BandwidthWidget::timeout(){
	reset();
}
