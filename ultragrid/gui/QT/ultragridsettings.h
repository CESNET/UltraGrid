#ifndef ULTRAGRIDSETTINGS_H
#define ULTRAGRIDSETTINGS_H

#include <QObject>
#include <QString>
#include <QFile>
#include <QSet>

struct Settings {
    QString mtu;
    QString display;
    QString capture;
    QString audio_cap;
    QString audio_play;
    QString display_details;
    QString capture_details;

    QString other;
};

class UltragridSettings : public QObject
{
    Q_OBJECT
public:
    explicit UltragridSettings(QObject *parent = 0);
    Settings getSettings();
    void setSettings(Settings settings);
    void save();

    QSet<QString> getHistory();
    void saveHistory(QSet<QString>);

private:
    Settings settings;
    QString fileName;

    QString histFileName;

signals:

public slots:

};

#endif // ULTRAGRIDSETTINGS_H
