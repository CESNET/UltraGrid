#ifndef LIVE_WIDGET_HPP_14CFE779EB7746999FBF8CE66D903392
#define LIVE_WIDGET_HPP_14CFE779EB7746999FBF8CE66D903392

#include <QWidget>

class LiveWidget final : public QWidget{
	Q_OBJECT
public:
	LiveWidget(QWidget *parent);


protected:
	void paintEvent(QPaintEvent *paintEvent) override;

public slots:
	void setLive(bool live);


private:
	QString liveText;
	QRect boundingRect;
	bool live = false;
};

#endif //LIVE_WIDGET_HPP_14CFE779EB7746999FBF8CE66D903392