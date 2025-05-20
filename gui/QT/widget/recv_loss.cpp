#include <QtGlobal>
#if defined(Q_OS_WIN) || defined(Q_OS_MACOS)
#include <QStyleFactory>
#endif
#include "recv_loss.hpp"
#include "utils/string_view_utils.hpp"

RecvLossWidget::RecvLossWidget(QWidget *parent) : QProgressBar(parent){
	elapsedTimer.start();
	reset();
	QObject::connect(&timer, &QTimer::timeout, this, &RecvLossWidget::timeout);
	timer.setInterval(timeout_msec);
#if defined(Q_OS_WIN) || defined(Q_OS_MACOS)
	setStyle(QStyleFactory::create("Fusion")); //mac style progressbars have no text
#endif
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
	static constexpr std::string_view prefix = "[Pbuf] ";
	if(!sv_is_prefix(line, prefix))
		return;

        line.remove_prefix(prefix.size());

	auto tmp = line;
	auto ident = tokenize(tmp, ' '); //"[<identifier>]"
	auto packet_counts = tokenize(tmp, ' '); //received/total

	auto received_sv = tokenize(packet_counts, '/');
	auto total_sv = tokenize(packet_counts, '/');

	int received;
	int total;

	if(!parse_num(received_sv, received)
			|| !parse_num(total_sv, total))
		return;

	addReport(std::string(ident), received, total);
	updateVal();
}

void RecvLossWidget::addReport(std::string ident, int received, int total){
	SSRC_report rep;
	rep.received = received;
	rep.total = total;
	reports.insert(ident, rep, elapsedTimer.elapsed());
}

void RecvLossWidget::updateVal(){
	int received = 0;
	int total = 0;

	reports.remove_timed_out(timeout_msec, elapsedTimer.elapsed());

	QString tooltip;
	for(auto& r : reports.get()){
		received += r.item.received;
		total += r.item.total;
		tooltip += QString::fromStdString(r.key) + ": "
			+ QString::number(r.item.received) + '/' + QString::number(r.item.total) + '\n';
	}

	if(tooltip.endsWith('\n'))
		tooltip.chop(1);

	if(total < 1)
		total = 1;

	float ratio = ((float) received / total) * 100.f;
	setFormat("Received %p% of packets");
	setValue(ratio);
	setToolTip(tooltip);
	timer.start();
}

void RecvLossWidget::timeout(){
	setValue(0);
	setFormat("Nothing received");
	setToolTip("Nothing received");
}
