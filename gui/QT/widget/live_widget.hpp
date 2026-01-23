#ifndef LIVE_WIDGET_HPP_14CFE779EB7746999FBF8CE66D903392
#define LIVE_WIDGET_HPP_14CFE779EB7746999FBF8CE66D903392

#include <QWidget>

class LiveWidget : public QWidget{
	Q_OBJECT
public:
	LiveWidget(QWidget *parent);


protected:
	void paintEvent(QPaintEvent *paintEvent);

public slots:
	void setLive(bool live);


private:
	QString liveText;
	QRect boundingRect;
	bool live = false;
};

#endif //LIVE_WIDGET_HPP_14CFE779EB7746999FBF8CE66D903392