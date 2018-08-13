#ifndef ULTRAGRID_OPTION_HPP
#define ULTRAGRID_OPTION_HPP

#include <QString>
#include <QObject>
#include <QMap>
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
	virtual void setAdvanced(bool advanced);

signals:
	void changed();

public slots:
	virtual void update();

protected:
	UltragridOption() :
		advanced(false) {  }

	bool advanced;
};

class ComboBoxOption : public UltragridOption{
public:
	ComboBoxOption(QComboBox *box,
			const QString& ultragridExecutable,
			QString opt);

	virtual QString getLaunchParam() override;
	virtual void queryAvailOpts() override;

	QString getCurrentValue() const;

protected:
	QStringList getAvailOpts();

	virtual void resetComboBox(QComboBox *box);

	// Used to filter options with whitelists, blacklists, etc.
	virtual bool filter(const QString & /*item*/) { return true; }

	// Used to query options not returned from the help command, e.g. v4l2 devices
	virtual void queryExtraOpts(const QStringList & /*opts*/) {  }

	// Returns extra launch params like video mode, bitrate, etc.
	virtual QString getExtraParams() { return ""; }

	void setItem(const QVariant &data);

	QString ultragridExecutable;
	QComboBox *box;
	QString opt;
};

class VideoSourceOption : public ComboBoxOption{
	Q_OBJECT
public:

	VideoSourceOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

protected:
	virtual bool filter(const QString &item) override;
	virtual void queryExtraOpts(const QStringList &opts) override;
	virtual QString getExtraParams() override;

	virtual void resetComboBox(QComboBox *box) override;

private:
	Ui::UltragridWindow *ui;

private slots:
	void srcChanged();
};

class VideoDisplayOption : public ComboBoxOption{
	Q_OBJECT
public:
	VideoDisplayOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

	virtual bool filter(const QString &item) override;
	virtual QString getLaunchParam() override;

private:
	Ui::UltragridWindow *ui;
	bool preview;

private slots:
	void enablePreview(bool);
};

class VideoCompressOption : public ComboBoxOption{
	Q_OBJECT
public:
	VideoCompressOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);

protected:
	virtual QString getExtraParams() override;
	virtual bool filter(const QString &item) override;
	virtual void queryExtraOpts(const QStringList &opts) override;

private:
	Ui::UltragridWindow *ui;

private slots:
	void compChanged();
};

class AudioSourceOption : public ComboBoxOption{
	Q_OBJECT
public:
	AudioSourceOption(Ui::UltragridWindow *ui,
			const VideoSourceOption *videoSrc,
			const QString& ultragridExecutable);

private:
	Ui::UltragridWindow *ui;
	const VideoSourceOption *videoSource;
	
protected:
	virtual QString getExtraParams() override;
	virtual bool filter(const QString &item) override;
};

class AudioPlaybackOption : public ComboBoxOption{
	Q_OBJECT
public:
	AudioPlaybackOption(Ui::UltragridWindow *ui,
			const VideoDisplayOption *videoDisplay,
			const QString& ultragridExecutable);

protected:
	virtual bool filter(const QString &item) override;

private:
	Ui::UltragridWindow *ui;
	const VideoDisplayOption *videoDisplay;
};

class AudioCompressOption : public ComboBoxOption{
	Q_OBJECT
public:
	AudioCompressOption(Ui::UltragridWindow *ui,
			const QString& ultragridExecutable);
	
protected:
	virtual QString getExtraParams() override;

private:
	Ui::UltragridWindow *ui;

private slots:
	void compChanged();
};

class FecOption : public UltragridOption{
	Q_OBJECT
public:
	FecOption(Ui::UltragridWindow *ui);

	QString getLaunchParam() override;
	void queryAvailOpts() override {  }

private:
	QString opt;
	Ui::UltragridWindow *ui;

public slots:
	void update() override;
};

class ArgumentOption : public UltragridOption{
	Q_OBJECT
public:
		ArgumentOption(Ui::UltragridWindow *ui);

		QString getLaunchParam() override;
		void queryAvailOpts() override {  }

		void setArg(QString name, QString value);
		void unsetArg(QString name);

		void setParam(QString name, QString value);
		void unsetParam(QString name);

protected:
		struct Argument {
			bool enabled = false;
			QString name = "";
			QString value = "";

			QString getArg() const {
				return name + " " + value;
			}

			QString getParam() const {
				if(value != "")
					return name + "=" + value;

				return name;
			}

			void setArg(QString name, QString value){
				this->name = name;
				this->value = value;
				enabled = true;
			}

			void unsetArg(){
				enabled = false;
			}
		};

		QMap<QString, Argument> args;
		QMap<QString, Argument> params;

private:
		Ui::UltragridWindow *ui;

public slots:
	void update() override;
};
#endif
