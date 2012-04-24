#include "ultragridsettings.h"
#include <stdlib.h>
#include <QTextStream>
#include <QDataStream>
#include <QStringList>

UltragridSettings::UltragridSettings(QObject *parent) :
    QObject(parent)
{
    fileName = QString(getenv("HOME")) + "/" + ".ultragridrc";
    histFileName = QString(getenv("HOME")) + "/" + ".ultragrid.hist";
    QFile file(fileName);
    file.open(QIODevice::ReadOnly);
    QTextStream stream(&file);

    QString line;
    while((line = stream.readLine()) != QString()) {
        QStringList tokenized = line.split(" ");
        settings.insert(tokenized[0], tokenized[1]);
    }

    /*stream >> "audio_cap " >> settings.audio_cap >> '\n' >>
              "audio_play " >> settings.audio_play >> '\n' >>
              "capture " >> settings.capture >> '\n' >>
              "capture_details " >> settings.capture_details >> '\n' >>
              "display " >> settings.display >> '\n' >>
              "display_details " >> settings.display_details >> '\n' >>
              "mtu " >> settings.mtu >> '\n' >>
              "other" >> settings.other >> '\n';*/
    file.close();
}

QHash<QString, QString> UltragridSettings::getSettings()
{
    return settings;
}

QString UltragridSettings::getValue(QString key)
{
    if(settings.find(key) != settings.end())
        return settings.find(key).value();
    else
        return QString();
}

void UltragridSettings::setValue(QString key, QString value)
{
    settings.insert(key, value);
}

void UltragridSettings::setSettings(QHash<QString, QString> settings)
{
    this->settings = settings;
}

void UltragridSettings::save()
{
    QFile file(fileName);
    file.open(QIODevice::WriteOnly);
    QTextStream stream(&file);


    QHash<QString, QString>::iterator i = settings.begin();
    while (i != settings.end()) {
        stream << i.key() <<  ' ' << i.value() << '\n';
        ++i;
    }

    /*stream << settings.audio_cap << settings.audio_play <<
              settings.capture << settings.capture_details << settings.display
              << settings.display_details << settings.mtu << settings.other;*/
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
