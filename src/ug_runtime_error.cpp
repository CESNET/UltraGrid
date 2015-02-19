#include "ug_runtime_error.h"

ug_runtime_error::ug_runtime_error(const std::string& what_arg, int code) : 
        std::runtime_error(what_arg), m_code(code)
{
}

int ug_runtime_error::get_code() const noexcept
{
        return m_code;
}

