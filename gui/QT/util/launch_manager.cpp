#include "launch_manager.hpp"
#include "debug.hpp"

LaunchManager::~LaunchManager(){
	if(currentLaunch){
		auto& p = currentLaunch->process;
		p.blockSignals(true);
		p.terminate();
		p.waitForFinished(500);
		if(p.state() != QProcess::NotRunning){
			p.kill();
			p.waitForFinished(1000);
		}
	}
	queuedLaunch.reset();
}

void LaunchManager::launch(std::unique_ptr<LaunchContext>&& ctx){
	if(getCurrentStatus() == LaunchContext::Type::None){
		doLaunch(std::move(ctx));
	} else {
		if(queuedLaunch && queuedLaunch->type == LaunchContext::Type::Run){
			return;
		}
		queuedLaunch = std::move(ctx);
		currentLaunch->terminate();
	}
}

void LaunchManager::doLaunch(std::unique_ptr<LaunchContext>&& ctx){
	currentLaunch = std::move(ctx);

	connect(&currentLaunch->process,
			QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
			this, &LaunchManager::processFinished);

	currentLaunch->process.start(currentLaunch->executablePath, currentLaunch->args);

	if(!currentLaunch->process.waitForStarted()){
		//Shouldn't happen as we probe ug executability at startup
		abort_msgBox("Failed to launch process. Did the ug executable disappear?");
	}
}

void LaunchManager::stop(){
	if(!currentLaunch)
		return;

	currentLaunch->terminate();
}

void LaunchManager::processFinished(){
	currentLaunch.reset();

	if(queuedLaunch)
		doLaunch(std::move(queuedLaunch));
}

LaunchContext::Type LaunchManager::getCurrentStatus() const{
	if(!currentLaunch)
		return LaunchContext::Type::None;

	return currentLaunch->type;
}

LaunchContext::LaunchContext(){
	connect(&killTimer, &QTimer::timeout, [=](){ process.kill(); });
	killTimer.setSingleShot(true);
	connect(&process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
			this, &LaunchContext::processTerminationHandler);

	process.setProcessChannelMode(QProcess::MergedChannels);
	connect(&process, &QProcess::readyReadStandardOutput, this, &LaunchContext::processOutput);
	connect(&process, &QProcess::readyReadStandardError, this, &LaunchContext::processOutput);
}

void LaunchContext::processTerminationHandler(int code, QProcess::ExitStatus es){
	emit processTerminated(code, es, terminationRequested);
}

void LaunchContext::processOutput(){
	QString str = process.isReadable() ? process.readAll() : QByteArray();

	emit processOutputRead(str);

	lineBuf.write(str.toStdString());

	while(lineBuf.hasLine()){
		emit processOutputLine(lineBuf.peek());
		lineBuf.pop();
	}
}

void LaunchContext::terminate(){
	terminationRequested = true;
	process.terminate();
	killTimer.start(terminateTimeoutMs);
}
