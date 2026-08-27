cmake_minimum_required(VERSION 3.10)

set(UG_GUI_PROJ_NAME uv-qt CACHE STRING "Name of the executable")

project(${UG_GUI_PROJ_NAME}
        LANGUAGES CXX C
        VERSION 0.0.1
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGL OpenGLWidgets Network)

execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_CURRENT_SHA1
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND git name-rev --name-only HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_CURRENT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions("-DGIT_CURRENT_SHA1=\"${GIT_CURRENT_SHA1}\"")
add_definitions("-DGIT_CURRENT_BRANCH=\"${GIT_CURRENT_BRANCH}\"")
add_definitions("-DGUI_BUILD")

set(CMAKE_AUTOUIC_SEARCH_PATHS ui)
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/window
        ${CMAKE_CURRENT_SOURCE_DIR}/util
        ${CMAKE_CURRENT_SOURCE_DIR}/widget
        ${CMAKE_CURRENT_SOURCE_DIR}/option
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tools
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

if(Qt6Core_VERSION VERSION_GREATER_EQUAL 6.3.0)
    qt_standard_project_setup()
else()
    #Compat for Ubuntu 22.04, TODO: Remove once CI finally upgrades
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    include(GNUInstallDirs)
    set(USE_FOLDERS ON)
endif()

set(UG_SOURCES
        ../../tools/astat.cpp
        ../../tools/ipc_frame.cpp
        ../../tools/ipc_frame_unix.cpp
        ../../src/compat/platform_pipe.c
        ../../src/utils/string_view_utils.cpp
)

set(SOURCES
        uv-qt.rc
        window/ultragrid_window.cpp
        option/available_settings.cpp
        option/settings.cpp
        option/settings_ui.cpp
        widget/previewWidget.cpp
        window/log_window.cpp
        widget/vuMeterWidget.cpp
        widget/recv_report.cpp
        widget/recv_loss.cpp
        widget/bandwidth_widget.cpp
        widget/live_widget.cpp
        window/settings_window.cpp
        option/widget_ui.cpp
        option/checkable_ui.cpp
        option/checkbox_ui.cpp
        option/actionCheckable_ui.cpp
        option/textOpt_ui.cpp
        option/lineedit_ui.cpp
        option/spinbox_ui.cpp
        option/combobox_ui.cpp
        option/groupBox_ui.cpp
        option/radioButton_ui.cpp
        option/audio_opts.cpp
        option/video_opts.cpp
        util/line_buffer.cpp
        util/control_port.cpp
        util/launch_manager.cpp
        ${UG_SOURCES}
        main.cpp
)

qt_add_executable(${UG_GUI_PROJ_NAME} ${SOURCES})

set_target_properties(${UG_GUI_PROJ_NAME} PROPERTIES
        WIN32_EXECUTABLE TRUE
        MACOSX_BUNDLE_INFO_PLIST MacOSXBundleInfo.plist.in
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_GUI_IDENTIFIER cz.cesnet.ultragrid.uv-qt
        MACOSX_BUNDLE_ICON_FILE icon.icns
)

install(TARGETS ${UG_GUI_PROJ_NAME}
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(APPLE OR NOT UNIX)

    qt_generate_deploy_app_script(
            TARGET ${UG_GUI_PROJ_NAME}
            OUTPUT_SCRIPT deploy_script
    )

    install(SCRIPT ${deploy_script})

endif()

target_link_libraries(${UG_GUI_PROJ_NAME} PRIVATE Qt6::Core Qt6::Widgets Qt6::OpenGL Qt6::OpenGLWidgets Qt6::Network)