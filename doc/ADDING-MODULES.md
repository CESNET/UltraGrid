Adding UltraGrid modules
========================

**Note:** Modular build is currently supported only for Linux. The
process of adding a module is, however, unified across platforms and
whether or not building standalone modules.

Table of contents
-----------------
- [Table of contents](#table-of-contents)
- [Adding a module to the build system](#adding-a-module-to-the-build-system)
  * [Naming convention](#naming-convention)
  * [Creating target](#creating-target)
- [Module API](#module-api)
- [Registering modules to UltraGrid runtime](#registering-modules-to-ultragrid-runtime)
- [Examples](#examples)
  * [Blank capture filter](#blank-capture-filter)
    + [configure.ac](#configureac)
    + [capture\_filter/blank.cpp](#capture_filterblankcpp)
  * [DeckLink capture](#decklink-capture)

Adding a module to the build system
-----------------------------------

Easiest way is to add a module object to *OBJ* variable in Makefile.in.
However, this variant is recommended only if module has no additional
dependencies and should be linked statically with the binary, if it is
not true, it is advisable to create a new target of the module as
described below.

### Naming convention

Module name is in format `ultragrid_<class>_<name>.so`. Names of
categories can be found in file `lib_common.cpp`, in *struct
library\_class\_info*.

### Creating target

You will need to use a macro `ADD_MODULE(name, objs, libs)` defined in
*configure.ac* to register module to build system:

-   name can be an empty string (\"\"). In that case module will be
    linked only if UltraGrid is built statically (without modules
    support)
-   if you provide name, it should be in format `<class>_<name>` (eg.
    *vidcap\_gl*), *\_ultragrid* will be appended automatically
-   second parameter should be a variable containing required objects
    for the module
-   third parameter are libraries that are needed for the module to link
    with

Module API
----------

Individual module have different APIs, these are most commonly described
in its corresponding header file (eg. *capture\_filter.h* or
*video\_display.h*), documentation for individual APIs should be in
corresponding header files. Resulting module has to use that API.

Registering modules to UltraGrid runtime
----------------------------------------

Finally, when module is written, it should be registered with
`REGISTER_MODULE` macro from *lib\_common.h*. General syntax is:

    REGISTER_MODULE(name, info, class_type, abi_version);

Eg.:

    REGISTER_MODULE(blank, &capture_filter_blank, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

Parameters are following:

1.  **name** of the module (should be the same as `<name>` of the
    module), not in quotation marks!
2.  **info** - pointer to structure defining API functions
3.  **class\_type** is one of member from `enum library_class` defined
    in *lib\_common.h*
4.  **abi\_version** is a class-specific number specifying ABI version
    number. It is usually defined in a header file corresponding with
    the class (eg. *video\_capture.h*)library specific ABI version

Examples
--------

### Blank capture filter

This example shows capture filter blank.

#### configure.ac

Usual parameter check:

    BLANK_LIBS=
    BLANK_OBJ=
    blank=no
    define(blank_dep, libswscale)
    AC_ARG_ENABLE(blank,
    [  --disable-blank        disable blank capture filter (default is auto)]
    [                         Requires: blank_dep],
       [blank_req=$enableval],
       [blank_req=$build_default]
    ) 

Then **pkg-config** can check the requirements:

`PKG_CHECK_MODULES([BLANK], [blank_dep], FOUND_BLANK_DEP=yes, FOUND_BLANK_DEP=no)`

If user didn't have explicitly **disabled** (`req != no`) the module and
dependencies were found, use values from *pkg-config*:

    if test $blank_req != no -a $FOUND_BLANK_DEP = yes
    then
            CFLAGS="$CFLAGS ${BLANK_CFLAGS}"
            CXXFLAGS="$CXXFLAGS ${BLANK_CFLAGS}"
            BLANK_OBJ="src/capture_filter/blank.o"
            ADD_MODULE("vcapfilter_blank", "$BLANK_OBJ", "$BLANK_LIBS")
            blank=yes
    fi

Following is not mandatory but useful for debugging:

    if test $blank_req = yes -a $blank = no; then
            AC_MSG_ERROR([Blank dep not found (libswscale)]);
    fi

Also adding *blank* to the summary at the end of *configure.ac* is
useful for user.

#### capture\_filter/blank.cpp

Only showing relevant parts of capture filter API registration.

    #include <lib_common.h>
    #include <video_capture.h>

    static const struct capture_filter_info capture_filter_blank = {
            .name = "blank",
            .init = init,
            .done = done,
            .filter = filter,
    };

    REGISTER_MODULE(blank, &capture_filter_blank, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

### DeckLink capture

Another example of registration of the **DeckLink** video capture in
*video\_capture/decklink.cpp*:

``` {.C}
static const struct video_capture_info vidcap_decklink_info = {
        vidcap_decklink_probe,
        vidcap_decklink_init,
        vidcap_decklink_done,
        vidcap_decklink_grab,
};

REGISTER_MODULE(decklink, &vidcap_decklink_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
```
