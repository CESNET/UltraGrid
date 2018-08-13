#ifndef LOG_WINDOW_HPP
#define LOG_WINDOW_HPP

#include <vector>
#include <memory>

#include "ui_log_window.h"

class LogWindow : public QDialog{
	Q_OBJECT

public:
	LogWindow(QWidget *parent = 0);
	void write(const QString& str);

private:
	Ui::LogWindow ui;

public slots:
	void copyToClipboard();
	void saveToFile();

private slots:
};
#endif
