#ifndef LAUNCH_MANAGAER_HPP_7a0f2a6b7dfc
#define LAUNCH_MANAGAER_HPP_7a0f2a6b7dfc

#include <QProcess>
#include <QString>
#include <QTimer>
#include <QStringList>
#include <memory>
#include "line_buffer.hpp"

class LaunchContext : public QObject{
	Q_OBJECT
public:
	enum class Type {
		None = 0,
		Preview,
		Run,
		Query
	};

	struct ExtraData{
		virtual ~ExtraData() = default;
	};

	LaunchContext();

	Type type = Type::None;
	QString executablePath;
	QStringList args;
	QProcess process;
	LineBuffer lineBuf;
	std::unique_ptr<ExtraData> extraData;

	void terminate();

signals:
	void processTerminated(int code, QProcess::ExitStatus es, bool wasRequested);
	void processOutputRead(QString str);
	void processOutputLine(std::string_view sv);

private:
	static constexpr int terminateTimeoutMs = 1000;
	QTimer killTimer;
	bool terminationRequested = false;

private slots:
	void processTerminationHandler(int code, QProcess::ExitStatus es);
	void processOutput();

};

class LaunchManager : public QObject{
	Q_OBJECT
public:
	~LaunchManager();

	void launch(std::unique_ptr<LaunchContext>&& ctx);
	void stop();

	LaunchContext::Type getCurrentStatus() const;

	/* Functions to allow modifying and relaunching the current LaunchContext
	 * from the processTerminated signal. At that time LaunchManager::processFinished
	 * is still pending execution, so the usual launch() cannot be used
	 */
	std::unique_ptr<LaunchContext> extractCurrentCtx();
	void reLaunch(std::unique_ptr<LaunchContext>&& ctx);

private:
	std::unique_ptr<LaunchContext> currentLaunch;
	std::unique_ptr<LaunchContext> queuedLaunch;

	void doLaunch(std::unique_ptr<LaunchContext>&& ctx);

private slots:
	void processFinished();

};


#endif
