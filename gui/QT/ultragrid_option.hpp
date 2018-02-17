#ifndef ULTRAGRID_OPTION_HPP
#define ULTRAGRID_OPTION_HPP

#include <QString>
#include <QObject>
#include "ui_ultragrid_window.h"

class QComboBox;
class QLineEdit;
class QLabel;

class UltragridWindow;

class UltragridOption : public QObject{
	Q_OBJECT
public:
	virtual QString getLaunchParam() = 0;
	virtual void queryAvailOpts() = 0;

	void setExecutable(const QString &executable) { ultragridExecutable = executable; }
	virtual void setAdvanced(bool advanced) { this->advanced = advanced; }

signals:
	void changed();

protected:
	UltragridOption(const QString& ultragridExecutable,
			const QString& opt) :
		ultragridExecutable(ultragridExecutable),
		opt(opt),
		advanced(false) {  }

	QStringList getAvailOpts(const QString& executable,
			const QString& helpCommand);

	void resetComboBox(QComboBox *box);

	void setItem(QComboBox *box, const QVariant &data);
	void setItem(QComboBox *box, const QString &data);

	QString ultragridExecutable;
	QString opt;
	bool advanced;

private:

};

class SourceOption : public UltragridOption{
	Q_OBJECT
public:

	SourceOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
	QString getCard() const;

private:
	Ui::UltragridWindow *ui;
	const static QStringList whiteList;

private slots:
	void srcChanged();
};

class DisplayOption : public UltragridOption{
	Q_OBJECT
public:
	DisplayOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
	QString getCard() const;

private:
	Ui::UltragridWindow *ui;
	bool preview;

	const static QStringList whiteList;

private slots:
	void enablePreview(bool);
};

class VideoCompressOption : public UltragridOption{
	Q_OBJECT
public:
	VideoCompressOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
private:
	Ui::UltragridWindow *ui;

private slots:
	void compChanged();
};

class AudioSourceOption : public UltragridOption{
	Q_OBJECT
public:
	AudioSourceOption(Ui::UltragridWindow *ui,
			const SourceOption *videoSrc,
			const QString& ultragridExecutable);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
private:
	Ui::UltragridWindow *ui;
	const SourceOption *videoSource;

private slots:
	void update();
};

class AudioPlaybackOption : public UltragridOption{
	Q_OBJECT
public:
	AudioPlaybackOption(Ui::UltragridWindow *ui,
			const DisplayOption *videoDisplay,
			const QString& ultragridExecutable);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
private:
	Ui::UltragridWindow *ui;
	const DisplayOption *videoDisplay;

private slots:
	void update();
};

class AudioCompressOption : public UltragridOption{
	Q_OBJECT
public:
	AudioCompressOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
private:
	Ui::UltragridWindow *ui;

private slots:
	void compChanged();
};

class GenericOption : public UltragridOption{
public:
	GenericOption(QComboBox *box,
			const QString& ultragridExecutable,
			QString opt);

	QString getLaunchParam() override;
	void queryAvailOpts() override;
private:
	QComboBox *box;
};

class FecOption : public UltragridOption{
	Q_OBJECT
public:
	FecOption(Ui::UltragridWindow *ui);

	QString getLaunchParam() override;
	void queryAvailOpts() override {  }

	void setAdvanced(bool advanced) override {
		this->advanced = advanced;
		update();
	}

private:
	Ui::UltragridWindow *ui;

private slots:
	void update();
};

#endif
