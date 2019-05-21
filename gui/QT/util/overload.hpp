#ifndef OVERLOAD_HPP
#define OVERLOAD_HPP

#include <QtGlobal>

#if QT_VERSION >= QT_VERSION_CHECK(5, 7, 0)

template<typename... Args>
using Overload = QOverload<Args...>;

#else

template<typename... Args>
struct Overload {
	template<typename R, typename C>
		static constexpr auto of(R (C::*ptr)(Args...)) noexcept -> decltype(ptr) {
			return ptr;
		}
};

#endif //QT_VERSION

#endif
