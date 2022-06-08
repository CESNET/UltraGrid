#include "ug_process_manager.hpp"

UgProcessManager::UgProcessManager(){
	connect(&killTimer, &QTimer::timeout, this, &UgProcessManager::killTimerTimeout);
	connect(&ugProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &UgProcessManager::processFinished);
	connect(&ugProcess, &QProcess::readyReadStandardOutput, this, &UgProcessManager::ugOutputAvailable);
	connect(&ugProcess, &QProcess::readyReadStandardError, this, &UgProcessManager::ugOutputAvailable);
	connect(&previewProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &UgProcessManager::processFinished);

	killTimer.setSingleShot(true);
}

void UgProcessManager::setUgExecutable(QString executable){
	ultragridExecutable = std::move(executable);
}

void UgProcessManager::launchUg(QStringList args){
	assert(state != State::UgRunning || state != UgToPreview)

	launchArgs = std::move(args);

	if(previewProcess.state() == QProcess::NotRunning){
		doLaunch(State::UgRunning);
		return;
	}

	changeStateAndEmit(State::PreviewToUg);
	previewProcess.terminate();
	killTimer.start(terminateTimeoutMs);
}

void UgProcessManager::launchPreview(QStringList args){
	launchArgs = std::move(args);

	if(previewProcess.state() != QProcess::NotRunning){
		DEBUG_ASSERT(ugProcess.state() == QProcess::NotRunning);
		changeStateAndEmit(State::PreviewToPreview);
		previewProcess.terminate();
		killTimer.start(terminateTimeoutMs);
		return;
	}

	if(ugProcess.state() != QProcess::NotRunning){
		DEBUG_ASSERT(previewProcess.state() == QProcess::NotRunning);
		changeStateAndEmit(State::UgToPreview);
		ugProcess.terminate();
		killTimer.start(terminateTimeoutMs);
		return;
	}

	doLaunch(State::PreviewRunning);
}

void UgProcessManager::stopAll(bool block){
	stopPreview(block);
	stopUg(block);
}

namespace{
void waitAndKill(QProcess& p){
	p.waitForFinished(500);
	if(p.state() != QProcess::NotRunning){
		p.kill();
		p.waitForFinished(1000);
	}
}
} // anon namespace

void UgProcessManager::stopPreview(bool block){
	if(previewProcess.state() != QProcess::NotRunning){
		if(block){
			previewProcess.terminate();
			waitAndKill(previewProcess);
			changeStateAndEmit(State::NotRunning);
		} else {
			changeStateAndEmit(State::StoppingAll);
			previewProcess.terminate();
			killTimer.start(terminateTimeoutMs);
		}
	}
}

void UgProcessManager::stopUg(bool block){
	if(ugProcess.state() != QProcess::NotRunning){
		if(block){
			ugProcess.terminate();
			waitAndKill(ugProcess);
			changeStateAndEmit(State::NotRunning);
		} else {
			changeStateAndEmit(State::StoppingAll);
			ugProcess.terminate();
			killTimer.start(terminateTimeoutMs);
		}
	}
}

void UgProcessManager::doLaunch(State target){
	switch(target){
	case State::PreviewRunning:
		previewProcess.start(ultragridExecutable, launchArgs);
		break;
	case State::UgRunning:
		ugProcess.setProcessChannelMode(QProcess::MergedChannels);
		ugProcess.start(ultragridExecutable, launchArgs);
		break;
	default:
		assert("Invalid doLaunch target" && false);
	}
	killTimer.stop();
	changeStateAndEmit(target);
}

void UgProcessManager::processFinished(int exitCode, QProcess::ExitStatus es){
	killTimer.stop();
	switch(state){
	case State::StoppingAll:
		assert(ugProcess.state() == QProcess::NotRunning);
		assert(previewProcess.state() == QProcess::NotRunning);
		changeStateAndEmit(State::NotRunning);
		break;
	case State::PreviewToUg:
		doLaunch(State::UgRunning);
		break;
	case State::UgToPreview:
	case State::PreviewToPreview:
		doLaunch(State::PreviewRunning);
		break;
	default:
		auto oldState = state;
		changeStateAndEmit(State::NotRunning);
		emit unexpectedExit(oldState, exitCode, es);
		break;
	}
}

void UgProcessManager::killTimerTimeout(){
	switch(state){
	case State::PreviewToUg:
		previewProcess.kill();
		break;
	case State::UgToPreview:
		ugProcess.kill();
		break;
	case State::StoppingAll:
		previewProcess.kill();
		ugProcess.kill();
		break;
	default:
		assert("killTimer expired in invalid state" && false);
	}
}

void UgProcessManager::changeStateAndEmit(State state){
	this->state = state;
	emit stateChanged(state);
}
