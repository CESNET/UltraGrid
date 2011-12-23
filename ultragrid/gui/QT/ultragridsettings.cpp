#include "ultragridsettings.h"
#include <stdlib.h>
#include <QTextStream>
#include <QDataStream>

UltragridSettings::UltragridSettings(QObject *parent) :
    QObject(parent)
{
    fileName = QString(getenv("HOME")) + "/" + ".ultragridrc";
    histFileName = QString(getenv("HOME")) + "/" + ".ultragrid.hist";
    QFile file(fileName);
    file.open(QIODevice::ReadOnly);
    QDataStream stream(&file);
    stream >> settings.audio_cap >> settings.audio_play >>
              settings.capture >> settings.capture_details >> settings.display
              >> settings.display_details >> settings.mtu >> settings.other;
    file.close();
}

Settings UltragridSettings::getSettings()
{
    return settings;
}

void UltragridSettings::setSettings(Settings settings)
{
    this->settings = settings;
}

void UltragridSettings::save()
{
    QFile file(fileName);
    file.open(QIODevice::WriteOnly);
    QDataStream stream(&file);
    stream << settings.audio_cap << settings.audio_play <<
              settings.capture << settings.capture_details << settings.display
              << settings.display_details << settings.mtu << settings.other;
    //QTextStream (&file) << settings.audio_cap;
    file.close();
}

QSet<QString> UltragridSettings::getHistory()
{
    QSet<QString> history;

    QFile file(histFileName);
    file.open(QIODevice::ReadOnly);
    QDataStream stream(&file);
    stream >> history;
    file.close();
    return history;
}

void UltragridSettings::saveHistory(QSet<QString> history)
{
    QFile file(histFileName);
    file.open(QIODevice::WriteOnly);
    QDataStream stream(&file);
    stream << history;
    //QTextStream (&file) << settings.audio_cap;
    file.close();
}
