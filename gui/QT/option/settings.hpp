#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>

class Settings;

class Option{
public:
	class Callback{
		public:
			using fcn_type = void (*)(Option &, bool, void *);

			Callback(fcn_type func_ptr, void *opaque);

			void operator()(Option &, bool) const;
			bool operator==(const Callback&) const;

		private:
			fcn_type func_ptr;
			void *opaque;
	};

	enum OptType{
		StringOpt,
		BoolOpt,
		SilentOpt
	};

	Option(Settings *settings,
			const std::string &name = "",
			OptType type = OptType::StringOpt,
			const std::string &param = "",
			const std::string &defaultValue = "",
			bool enabled = true) :
		name(name),
		param(param),
		value(defaultValue),
		defaultValue(defaultValue),
		enabled(enabled),
		type(type),
		settings(settings)	{  }

	virtual ~Option() {  }

	std::string getName() const;
	std::string getValue() const;
	std::string getSubVals() const;
	std::string getParam() const;
	OptType getType() const { return type; };
	virtual std::string getLaunchOption() const;

	virtual void setValue(const std::string &val, bool suppressCallback = false);
	virtual void setDefaultValue(const std::string &val);
	void setParam(const std::string &p) { param = p; }

	bool isEnabled() const { return enabled; }
	void setEnabled(bool enable, bool suppressCallback = false);

	void setType(OptType type) { this->type = type; };

	void addSuboption(Option *sub, const std::string &limit = "");
	void addOnChangeCallback(Callback callback);

	Settings *getSettings();
	void changed();

protected:
	std::string name;
	std::string param;
	std::string value;
	std::string defaultValue;

	bool enabled;

	OptType type;

	static void suboptionChanged(Option &opt, bool suboption, void *opaque);

	std::vector<Callback> onChangeCallbacks;
	std::vector<std::pair<std::string, Option *>> suboptions;

	Settings *settings;
};

class Settings{
public:
	Settings();

	std::string getLaunchParams() const;
	std::string getPreviewParams() const;

	const Option& getOption(const std::string &opt) const;
	Option& getOption(const std::string &opt);

	Option& addOption(std::string name,
			Option::OptType type,
			const std::string &param,
			const std::string &value = "",
			bool enabled = true,
			const std::string &parent = "",
			const std::string &limit = "");

	bool isAdvancedMode();

	const std::map<std::string, std::unique_ptr<Option>>& getOptionMap() const;

	void changedAll();

private:

	std::map<std::string, std::unique_ptr<Option>> options;

	const Option dummy;
};

#endif //SETTINGS_HPP
