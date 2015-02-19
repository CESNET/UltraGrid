#include <cstdlib>
#include <stdexcept>

class ug_runtime_error : public std::runtime_error {
public:
        ug_runtime_error(const std::string& what_arg, int code = EXIT_FAILURE);
        int get_code() const noexcept;
private:
        int m_code;
};

