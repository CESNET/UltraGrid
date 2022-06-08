#ifndef UG_PROCESS_MANAGER_HPP
#define UG_PROCESS_MANAGER_HPP

#include <QProcess>
#include <QString>
#include <QTimer>
#include <QStringList>

class UgProcessManager : public QObject{
	Q_OBJECT

public:
	UgProcessManager();

	void setUgExecutable(QString executable);

	enum class State{
		NotRunning = 0,
		PreviewRunning,
		PreviewToUg,
		UgRunning,
		UgToPreview,
		PreviewToPreview,
		StoppingAll
	};

	const QProcess& getUgProcess(){ return ugProcess; }
	const QProcess& getPreviewProcess(){ return previewProcess; }

	State getState() const { return state; }

	QByteArray readUgOut() { return ugProcess.readAll(); }

public slots:
	void launchUg(QStringList launchArgs);
	void launchPreview(QStringList launchArgs);
	void stopPreview(bool block = false);
	void stopUg(bool block = false);
	void stopAll(bool block = false);

private:
	void doLaunch(State target);
	void changeStateAndEmit(State state);

	int terminateTimeoutMs = 1000;

	QString ultragridExecutable;
	QProcess ugProcess;
	QProcess previewProcess;

	QStringList launchArgs;

	QTimer killTimer;

	State state = State::NotRunning;

signals:
	void stateChanged(State state);
	void ugOutputAvailable();
	void unexpectedExit(State oldState, int exitCode, QProcess::ExitStatus es);

private slots:
	void processFinished(int exitCode, QProcess::ExitStatus es);
	void killTimerTimeout();
};

#endif //UG_PROCESS_MANAGER_HPP
