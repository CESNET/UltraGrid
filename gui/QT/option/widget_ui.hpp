#ifndef WIDGET_UI
#define WIDGET_UI

#include <QObject>
#include <set>
#include "settings.hpp"

class WidgetUi : public QObject{
Q_OBJECT

public:
    WidgetUi(Settings *settings, const std::string &opt);
	WidgetUi(const WidgetUi&) = delete;
	WidgetUi(WidgetUi&&) = delete;

	WidgetUi& operator=(const WidgetUi&) = delete;
	WidgetUi& operator=(WidgetUi&&) = delete;

    virtual ~WidgetUi();

    void setOpt(const std::string &opt);

    virtual void refresh() {  }

    void registerCallback(const std::string &option);

protected:
    Settings *settings;
    std::string opt;
    std::map<std::string, Option::Callback> registeredCallbacks;

    void registerCallback();

	static void optChangeCallbackStatic(Option&, bool, void *);

    virtual void connectSignals() = 0;
    virtual void updateUiState() = 0;
	virtual void optChangeCallback(Option &opt, bool suboption) = 0;

signals:
    void changed();
};

#endif
