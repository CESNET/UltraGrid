#include <stdio.h>
#include <iostream>
#include <QMetaType>
#include "settings_ui.hpp"

void SettingsUi::init(Settings *settings, AvailableSettings *availableSettings){
	this->settings = settings;
	this->availableSettings = availableSettings;
}

void SettingsUi::addControl(WidgetUi *widget){
	uiControls.emplace_back(widget);

	connect(widget, &WidgetUi::changed, this, &SettingsUi::changed);
}

void SettingsUi::initMainWin(Ui::UltragridWindow *ui){
	mainWin = ui;

	refreshAll();
 
	addCallbacks();
#ifdef DEBUG
	connect(mainWin->actionTest, SIGNAL(triggered()), this, SLOT(test()));
#endif

	addControl(new CheckboxUi(ui->fECCheckBox, settings, "network.fec.enabled"));
	addControl(new CheckboxUi(ui->previewCheckBox, settings, "preview"));
	addControl(
			new LineEditUi(ui->networkDestinationEdit, settings, "network.destination")
			);
	addControl(new ActionCheckableUi(ui->actionAdvanced, settings, "advanced"));
	addControl(new ActionCheckableUi(ui->actionVuMeter, settings, "vuMeter"));
	addControl(new ActionCheckableUi(ui->actionUse_hw_acceleration,
				settings,
				"decode.hwaccel"));

	addControl(
			new ComboBoxUi(ui->videoSourceComboBox,
				settings,
				"video.source",
				std::bind(getVideoSrc, availableSettings))
			);

	LineEditUi *videoBitrate = new LineEditUi(ui->videoBitrateEdit,
			settings,
			"");
	addControl(videoBitrate);
	using namespace std::placeholders;
	settings->getOption("video.compress").addOnChangeCallback(
			std::bind(videoCompressBitrateCallback, videoBitrate, _1, _2)
			);

	addControl(
			new ComboBoxUi(ui->videoCompressionComboBox,
				settings,
				"video.compress",
				std::bind(getVideoCompress, availableSettings))
			);

	addControl(
			new ComboBoxUi(ui->videoDisplayComboBox,
				settings,
				"video.display",
				std::bind(getVideoDisplay, availableSettings))
			);

	ComboBoxUi *videoModeCombo = new ComboBoxUi(ui->videoModeComboBox,
				settings,
				"",
				std::bind(getVideoModes, availableSettings));
	videoModeCombo->registerCallback("video.source");
	addControl(videoModeCombo);

	addControl(
			new ComboBoxUi(ui->audioCompressionComboBox,
				settings,
				"audio.compress",
				std::bind(getAudioCompress, availableSettings))
			);

	ComboBoxUi *audioSrc = new ComboBoxUi(ui->audioSourceComboBox,
			settings,
			"audio.source",
			std::bind(getAudioSrc, availableSettings));
	audioSrc->registerCallback("video.source");
	addControl(audioSrc);

	ComboBoxUi *audioPlayback = new ComboBoxUi(ui->audioPlaybackComboBox,
			settings,
			"audio.playback",
			std::bind(getAudioPlayback, availableSettings));
	audioPlayback->registerCallback("video.display");
	addControl(audioPlayback);

	addControl(new LineEditUi(ui->audioBitrateEdit,
				settings,
				"audio.compress.bitrate"));

	addControl(new SpinBoxUi(ui->audioChannelsSpinBox,
				settings,
				"audio.source.channels"));

	addControl(new LineEditUi(ui->portLineEdit, settings, "network.port"));
}

void SettingsUi::refreshAll(){
	for(auto &i : uiControls){
		i->refresh();
	}
}

void vuMeterCallback(Ui::UltragridWindow *win, Option &opt, bool suboption){
	win->vuMeter->setVisible(opt.isEnabled());
}


void SettingsUi::addCallbacks(){
	using namespace std::placeholders;

#define CALLBACK(opt, fun) { opt, std::bind(fun, this, _1, _2) }
	const static struct{
		const char *opt;
		std::function<void(Option &, bool)> callback;
	} callbacks[] = {
		CALLBACK("video.compress", &SettingsUi::jpegLabelCallback),
		{"audio.compress", std::bind(audioCompressionCallback, mainWin, _1, _2)},
		{"advanced", std::bind(&SettingsUi::refreshAll, this)},
		{"vuMeter", std::bind(vuMeterCallback, mainWin, _1, _2)},
	};
#undef CALLBACK

	for(const auto & call : callbacks){
		settings->getOption(call.opt).addOnChangeCallback(call.callback);
	}
}

void SettingsUi::test(){
	printf("%s\n", settings->getLaunchParams().c_str());
}

void SettingsUi::jpegLabelCallback(Option &opt, bool suboption){
	if(suboption)
		return;

	mainWin->videoBitrateLabel->setEnabled(true);
	mainWin->videoBitrateEdit->setEnabled(true);

	const std::string &val = opt.getValue();
	if(val == "jpeg"){
		mainWin->videoBitrateLabel->setText(QString("Jpeg quality"));
	} else {
		mainWin->videoBitrateLabel->setText(QString("Bitrate"));
		if(val == ""){
			mainWin->videoBitrateLabel->setEnabled(false);
			mainWin->videoBitrateEdit->setEnabled(false);
		}
	}
}

void SettingsUi::fecCallback(Option &opt, bool){
	mainWin->fECCheckBox->setChecked(opt.isEnabled());
	settingsWin->fecGroupBox->setChecked(opt.isEnabled());
}

void SettingsUi::initSettingsWin(Ui::Settings *ui){
	settingsWin = ui;

	addControl(new LineEditUi(ui->basePort, settings, "network.port"));
	addControl(new LineEditUi(ui->controlPort, settings, "network.control_port"));
	addControl(new CheckboxUi(ui->fecAutoCheck, settings, "network.fec.auto"));
	addControl(new GroupBoxUi(ui->fecGroupBox, settings, "network.fec.enabled"));
	addControl(new SpinBoxUi(ui->multSpin, settings, "network.fec.mult.factor"));
	addControl(new SpinBoxUi(ui->ldgmK, settings, "network.fec.ldgm.k"));
	addControl(new SpinBoxUi(ui->ldgmM, settings, "network.fec.ldgm.m"));
	addControl(new SpinBoxUi(ui->ldgmC, settings, "network.fec.ldgm.c"));
	addControl(new SpinBoxUi(ui->rsK, settings, "network.fec.rs.k"));
	addControl(new SpinBoxUi(ui->rsN, settings, "network.fec.rs.n"));
	addControl(new RadioButtonUi(ui->fecNoneRadio, "", settings, "network.fec.type"));
	addControl(new RadioButtonUi(ui->fecMultRadio, "mult", settings, "network.fec.type"));
	addControl(new RadioButtonUi(ui->fecLdgmRadio, "ldgm", settings, "network.fec.type"));
	addControl(new RadioButtonUi(ui->fecRsRadio, "rs", settings, "network.fec.type"));
	addControl(new CheckboxUi(ui->glCursor, settings, "video.display.gl.cursor"));
	addControl(new CheckboxUi(ui->glDeinterlace, settings, "video.display.gl.deinterlace"));
	addControl(new CheckboxUi(ui->glFullscreen, settings, "video.display.gl.fullscreen"));
	addControl(new CheckboxUi(ui->glNoDecorate, settings, "video.display.gl.nodecorate"));
	addControl(new CheckboxUi(ui->glNoVsync, settings, "video.display.gl.novsync"));
	addControl(new CheckboxUi(ui->sdlDeinterlace, settings, "video.display.sdl.deinterlace"));
	addControl(new CheckboxUi(ui->sdlFullscreen, settings, "video.display.sdl.fullscreen"));
	addControl(new RadioButtonUi(ui->ldgmSimpCpuRadio, "CPU", settings, "network.fec.ldgm.device"));
	addControl(new RadioButtonUi(ui->ldgmSimpGpuRadio, "GPU", settings, "network.fec.ldgm.device"));
	addControl(new RadioButtonUi(ui->ldgmCpuRadio, "CPU", settings, "network.fec.ldgm.device"));
	addControl(new RadioButtonUi(ui->ldgmGpuRadio, "GPU", settings, "network.fec.ldgm.device"));
	addControl(new CheckboxUi(ui->decodeAccelCheck, settings, "decode.hwaccel"));
}
