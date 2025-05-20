#include <stdio.h>
#include <iostream>
#include <QMetaType>
#include <QLabel>
#include <QLineEdit>
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

	addCallbacks();
#ifdef DEBUG
	connect(mainWin->actionTest, SIGNAL(triggered()), this, SLOT(test()));
#endif

	connect(ui->networkModeTabWidget, &QTabWidget::currentChanged,
			this, &SettingsUi::networkModeTabChanged);

	addControl(new CheckboxUi(ui->fECCheckBox, settings, "network.fec.enabled"));
	addControl(new CheckboxUi(ui->previewCheckBox, settings, "preview"));
	addControl(
			new LineEditUi(ui->networkDestinationEdit, settings, "network.destination")
			);
	addControl(
			new LineEditUi(ui->networkDestinationEdit_clientTab, settings, "network.destination")
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
	addControl(new LineEditUi(ui->portLineEdit_serverTab, settings, "network.port"));
	addControl(new LineEditUi(ui->portLineEdit_clientTab, settings, "network.port"));
}

void SettingsUi::refreshAll(){
	for(auto &i : uiControls){
		i->refresh();
	}
	buildSettingsCodecList();
	addDeviceTabs();
}

void SettingsUi::refreshAllCallback(Option&, bool, void *opaque){
	static_cast<SettingsUi *>(opaque)->refreshAll();
}

static void vuMeterCallback(Option &opt, bool /*suboption*/, void *opaque){
	static_cast<Ui::UltragridWindow *>(opaque)->vuMeter->setVisible(opt.isEnabled());
	static_cast<Ui::UltragridWindow *>(opaque)->vuMeterSend->setVisible(opt.isEnabled());
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
	addControl(new CheckboxUi(ui->controlPortRandom, settings, "network.control_port.random"));
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
	addControl(new RadioButtonUi(ui->ldgmSimpCpuRadio, "CPU", settings, "network.fec.ldgm.device"));
	addControl(new RadioButtonUi(ui->ldgmSimpGpuRadio, "GPU", settings, "network.fec.ldgm.device"));
	addControl(new RadioButtonUi(ui->ldgmCpuRadio, "CPU", settings, "network.fec.ldgm.device"));
	addControl(new RadioButtonUi(ui->ldgmGpuRadio, "GPU", settings, "network.fec.ldgm.device"));
	addControl(new CheckboxUi(ui->decodeAccelCheck, settings, "decode.hwaccel"));
	addControl(new CheckboxUi(ui->errorsFatalBox, settings, "errors_fatal"));
	addControl(new LineEditUi(ui->encryptionLineEdit, settings, "encryption"));
	addControl(new LineEditUi(ui->extraOptsLineEdit, settings, "extraOpts"));
	addControl(new CheckboxUi(ui->advModeCheck, settings, "advanced"));

	addDeviceTabs();

	buildSettingsCodecList();
	connect(settingsWin->codecList, &QListWidget::currentItemChanged,
			this, &SettingsUi::settingsCodecSelected);
}

void SettingsUi::addDeviceTabs(){
	struct{
		const char *label;
		SettingType type;
		const char *keyPrefix;
	} tabs[] = {
		{"Audio playback", AUDIO_PLAYBACK, "audio.playback"},
		{"Audio capture", AUDIO_SRC, "audio.source"},
		{"Display", VIDEO_DISPLAY, "video.display"},
		{"Capture", VIDEO_SRC, "video.source"},
	};

	deviceTabs.clear();

	for(const auto& i : tabs){
		DeviceOptTab tab;
		tab.keyPrefix = i.keyPrefix;
		tab.page = std::make_unique<QWidget>();
		auto layout = new QHBoxLayout();
		tab.page->setLayout(layout);

		tab.listWidget = new QListWidget();
		layout->addWidget(tab.listWidget, 1);
		tab.scrollArea = new QScrollArea();
		layout->addWidget(tab.scrollArea, 3);

		tab.listWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
		tab.scrollArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

		buildSettingsDeviceList(tab.listWidget, i.type);

		if(tab.listWidget->count() == 0){
			continue;
		}

		connect(tab.listWidget, &QListWidget::currentItemChanged,
				this, &SettingsUi::settingsDeviceSelected);

		settingsWin->tabWidget->insertTab(0, tab.page.get(), i.label);
		deviceTabs.emplace(i.type, std::move(tab));
	}
}

Q_DECLARE_METATYPE(Codec);

void SettingsUi::buildSettingsCodecList(){
	if(!settingsWin)
		return;

	QListWidget *list = settingsWin->codecList;
	list->clear();

	for(const auto& codec : availableSettings->getVideoCompressCodecs()){
		QListWidgetItem *item = new QListWidgetItem(list);
		item->setText(QString::fromStdString(codec.name));
		item->setData(Qt::UserRole, QVariant::fromValue(codec));

		list->addItem(item);
	}
}

Q_DECLARE_METATYPE(Device);

void SettingsUi::buildSettingsDeviceList(QListWidget *list, SettingType type){
	if(!settingsWin)
		return;

	list->clear();

	for(const auto& dev : availableSettings->getDevices(type)){
		if(dev.opts.empty())
			continue;

		QListWidgetItem *item = new QListWidgetItem(list);

		item->setText(QString::fromStdString(dev.name));
		item->setData(Qt::UserRole, QVariant::fromValue(dev));

		list->addItem(item);
	}
}

void SettingsUi::networkModeTabChanged(int index){
	Option &modeOpt = settings->getOption("network.mode");
	switch(index){
		case 0:
			modeOpt.setValue(" ");
			break;
		case 1:
			modeOpt.setValue(" -S");
			break;
		case 2:
			modeOpt.setValue(" -C");
			break;
	}

	emit changed();
}

void SettingsUi::settingsCodecSelected(QListWidgetItem *curr, QListWidgetItem *){
	if(!curr){
		codecControls.clear();
		return;
	}

	const auto &codec = curr->data(Qt::UserRole).value<Codec>();

	buildCodecOptControls(codec);
}

void SettingsUi::settingsDeviceSelected(QListWidgetItem *curr, QListWidgetItem *){
	if(!curr){
		return;
	}

	const auto &dev = curr->data(Qt::UserRole).value<Device>();

	buildDeviceOptControls(dev, deviceTabs.at(dev.settingType));
}

void SettingsUi::buildCodecOptControls(const Codec& codec){
	ControlForm form;

	QComboBox *encoderCombo = new QComboBox();
	form.formLayout->addRow("Encoder", encoderCombo);

	WidgetUi *encoderComboUi = new ComboBoxUi(encoderCombo,
			settings,
			"video.compress." + codec.module_name + ".codec." + codec.name + ".encoder",
			std::bind(getCodecEncoders, codec));

	form.uiControls.emplace_back(encoderComboUi);
	connect(encoderComboUi, &WidgetUi::changed, this, &SettingsUi::changed);

	for(const auto& compMod : availableSettings->getVideoCompressModules()){
		if(compMod.name == codec.module_name){
			fillFormOptions(form, "video.compress." + compMod.name + ".", compMod.opts);
		}
	}

	codecControls = std::move(form.uiControls);
	settingsWin->codecOptScroll->setWidget(form.uiContainer.release());
}

void SettingsUi::buildDeviceOptControls(const Device& dev, DeviceOptTab& tab){
	ControlForm form;

	std::string optPrefix = tab.keyPrefix + "." + dev.type + ".device." + dev.deviceOpt + ".";
	fillFormOptions(form, optPrefix, dev.opts);

	tab.uiControls = std::move(form.uiControls);
	tab.scrollArea->setWidget(form.uiContainer.release());
}

void SettingsUi::fillFormOptions(ControlForm& form,
		std::string_view keyPrefix,
		const std::vector<CapabOpt>& opts)
{
	for(const auto& modOpt : opts){
		QLabel *label = new QLabel(QString::fromStdString(modOpt.displayName));
		QWidget *field = nullptr;
		WidgetUi *widgetUi = nullptr;
		std::string optKey = std::string(keyPrefix) + modOpt.key;
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

		form.formLayout->addRow(label, field);

		form.uiControls.emplace_back(widgetUi);
		connect(widgetUi, &WidgetUi::changed, this, &SettingsUi::changed);
	}
}
