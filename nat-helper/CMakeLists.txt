cmake_minimum_required(VERSION 3.14)

project(nat-helper
	LANGUAGES CXX
	VERSION 0.0.1
	)

set(SOURCES
	main.cpp
	nat-helper.cpp
	client.cpp
	message.cpp
	room.cpp
	)

add_executable(nat-helper main.cpp ${SOURCES})

set_property(TARGET nat-helper PROPERTY CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
	include(CheckPIESupported)
	check_pie_supported()
	set_target_properties(nat-helper PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
	target_compile_definitions(nat-helper PRIVATE FORTIFY_SOURCE=2)
	target_compile_options(nat-helper PRIVATE -fstack-protector-strong)
	target_compile_options(nat-helper PRIVATE -Wall -Wextra -pedantic)
	target_link_options(nat-helper PRIVATE "SHELL:-z relro")
	target_link_options(nat-helper PRIVATE "SHELL:-z now")
endif()

target_link_libraries(nat-helper PRIVATE
	pthread
		)

