#include <stdio.h>
#include <iostream>
#include <QMetaType>
#include <QLabel>
#include <QLineEdit>
#include <QFormLayout>
#include <QCheckBox>
#include <functional>
#include "settings_ui.hpp"
#include "video_opts.hpp"

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

	std::unique_ptr<VideoBitrateCallbackData> vidBitrateCallData(new VideoBitrateCallbackData());

	vidBitrateCallData->availSettings = availableSettings;
	vidBitrateCallData->lineEditUi = videoBitrate;
	vidBitrateCallData->label = mainWin->videoBitrateLabel;

	auto extraPtr = vidBitrateCallData.get();
	videoBitrate->registerCustomCallback("video.compress",
			Option::Callback(videoCompressBitrateCallback, extraPtr),
			std::move(vidBitrateCallData));


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

void SettingsUi::refreshAllCallback(Option&, bool, void *opaque){
	static_cast<SettingsUi *>(opaque)->refreshAll();
}

void vuMeterCallback(Option &opt, bool /*suboption*/, void *opaque){
	static_cast<Ui::UltragridWindow *>(opaque)->vuMeter->setVisible(opt.isEnabled());
}

void SettingsUi::addCallbacks(){
	settings->getOption("audio.compress").addOnChangeCallback(
			Option::Callback(&audioCompressionCallback, mainWin));

	settings->getOption("advanced").addOnChangeCallback(
			Option::Callback(&SettingsUi::refreshAllCallback, this));

	settings->getOption("vuMeter").addOnChangeCallback(
			Option::Callback(&vuMeterCallback, mainWin));
}

void SettingsUi::test(){
	printf("%s\n", settings->getLaunchParams().c_str());
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
	addControl(new CheckboxUi(ui->errorsFatalBox, settings, "errors_fatal"));
	addControl(new LineEditUi(ui->encryptionLineEdit, settings, "encryption"));
	addControl(new CheckboxUi(ui->advModeCheck, settings, "advanced"));

	buildSettingsCodecList();
	connect(settingsWin->codecList, &QListWidget::currentItemChanged,
			this, &SettingsUi::settingsCodecSelected);
}

void SettingsUi::buildSettingsCodecList(){
	QListWidget *list = settingsWin->codecList;
	list->clear();

	auto codecs = getVideoCompress(availableSettings);

	for(const auto& codec : codecs){
		QListWidgetItem *item = new QListWidgetItem(list);
		item->setText(QString::fromStdString(codec.name));
		item->setData(Qt::UserRole, QVariant::fromValue(codec));

		list->addItem(item);
	}
}

void SettingsUi::settingsCodecSelected(QListWidgetItem *curr, QListWidgetItem *){
	const SettingItem &settingItem = curr->data(Qt::UserRole).value<SettingItem>();

	auto modIt = std::find_if(settingItem.opts.begin(), settingItem.opts.end(),
			[](const SettingValue& si){ return si.opt == "video.compress"; });

	if(modIt == settingItem.opts.end())
		return;

	const std::string& modName = modIt->val;

	auto codecIt = std::find_if(settingItem.opts.begin(), settingItem.opts.end(),
			[modName](const SettingValue& si){
				return si.opt == "video.compress." + modName + ".codec"; });

	if(codecIt == settingItem.opts.end())
		return;

	buildCodecOptControls(modName, codecIt->val);
}

void SettingsUi::buildCodecOptControls(const std::string& mod, const std::string& codec){
	codecControls.clear();

	QWidget *container = new QWidget();
	QFormLayout *formLayout = new QFormLayout(container);

	QComboBox *encoderCombo = new QComboBox();
	formLayout->addRow("Encoder", encoderCombo);

	WidgetUi *encoderComboUi = new ComboBoxUi(encoderCombo,
			settings,
			"video.compress." + mod + ".codec." + codec + ".encoder",
			std::bind(getCodecEncoders, availableSettings,  mod, codec));

	codecControls.emplace_back(encoderComboUi);
	connect(encoderComboUi, &WidgetUi::changed, this, &SettingsUi::changed);

	for(const auto& compMod : availableSettings->getVideoCompressModules()){
		if(compMod.name == mod){
			for(const auto& modOpt : compMod.opts){
				QLabel *label = new QLabel(QString::fromStdString(modOpt.displayName));
				QWidget *field = nullptr;
				WidgetUi *widgetUi = nullptr;
				std::string optKey = "video.compress." + mod + "." + modOpt.key;
				if(modOpt.booleanOpt){
					QCheckBox *checkBox = new QCheckBox();
					field = checkBox;

					widgetUi = new CheckboxUi(checkBox,
							settings,
							optKey);
				} else {
					QLineEdit *lineEdit = new QLineEdit();
					field = lineEdit;

					widgetUi = new LineEditUi(lineEdit,
							settings,
							optKey);
				}
				label->setToolTip(QString::fromStdString(modOpt.displayDesc));
				field->setToolTip(QString::fromStdString(modOpt.displayDesc));

				formLayout->addRow(label, field);

				codecControls.emplace_back(widgetUi);
				connect(widgetUi, &WidgetUi::changed, this, &SettingsUi::changed);
			}
		}
	}

	container->setLayout(formLayout);

	delete settingsWin->scrollContents;
	settingsWin->scrollContents = container;
	settingsWin->codecOptScroll->setWidget(container);
}
