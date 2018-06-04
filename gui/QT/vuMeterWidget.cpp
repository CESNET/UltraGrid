#include "vuMeterWidget.hpp"
#include <algorithm>
#include <cmath>

void VuMeterWidget::updateVal(){
	updateVolumes();

	const double zeroLevel = 30.0;
	const double fallSpeed = 200.0;

	barLevel = std::max(barLevel - fallSpeed / updatesPerSecond, 0.0);
	double newHeight = 100 + std::max(peak[1], -zeroLevel) * (100 / zeroLevel);

	barLevel = std::max(barLevel, newHeight);

	setValue(std::round(barLevel));
}

void VuMeterWidget::updateVolumes(){
	ug_connection *ug_c = connection.get();
	if(ug_c == nullptr){
		connect_ug();
		return;
	}

	bool ret = ug_control_get_volumes(ug_c, peak, rms, &count);

	if(!ret){
		last_connect = std::chrono::system_clock::now();
		connect_ug();
	}
}

void VuMeterWidget::connect_ug(){
	disconnect_ug();

	const std::chrono::duration<double> timeout = std::chrono::seconds(1);

	if(std::chrono::system_clock::now() - last_connect < timeout){
		return;
	}

	ug_connection *c = ug_control_connection_init(port);
	if(!c)
		return;

	connection = std::unique_ptr<ug_connection, void(*)(ug_connection *)>(c, ug_control_connection_done);
	last_connect = std::chrono::system_clock::now();
}

void VuMeterWidget::disconnect_ug(){
	connection = std::unique_ptr<ug_connection, void(*)(ug_connection *)>(nullptr, ug_control_connection_done);
}
