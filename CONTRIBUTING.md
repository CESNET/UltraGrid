# How to contribute to UltraGrid
First, thanks to contribute to UltraGrid by using UltraGrid, reporting bugs,
adding code or ideas. Further section contains basic informations how to
contribute. Feel free to inform us if you need to contribute in a different
way or extend this document.

## Table of Contents

* [Resources](#resources)
* [Reporting bugs](#reporting-bugs)
* [Feature requests](#feature-requests)
* [Pull requests](#pull-requests)
* [Coding standards](#coding-standards)
   * [Style](#style)
   * [Naming conventions](#naming-conventions)
* [Conclusion](#conclusion)

## Resources

- [Adding modules HOWTO](doc/ADDING-MODULES.md)
- [Doxygen](https://frakira.fi.muni.cz/~xpulec/ultragrid-doxygen/html/)
- [GitHub devel page](https://github.com/CESNET/UltraGrid/wiki/Developer-Documentation)
- [Trello board](https://trello.com/b/PjZW4sas/ultragrid-development) tracking the development

## Reporting bugs

Prefered way to report a bug is to open an issue on GitHub. If not possible,
you can also contact the development team directly with an e-mail.

Please look also at detailed instructions here [here](doc/REPORTING-BUGS.md).

For general questions, ideas or remarks you are encountered to use the project's
[GitHub Discussions](https://github.com/CESNET/UltraGrid/discussions).

## Feature requests
Feature requests can be submitted with the same channels as bugs.

## Pull requests

Pull request should be issued as a single branch originating from repository
master branch, preferably rebased to current version. If multiple features are
created, multiple pull requests containing every single feature should be
issued.

## Coding standards
**TODO:** incomplete

### Style

Recommended style is modified LLVM clang-format style:
```
BasedOnStyle: LLVM
IndentWidth: 8
BreakBeforeBraces: Linux
````

The style is deduced from original _rtp/rtp.{c,h}_ formatting and it
is similar to [Linux Kernel
style](https://www.kernel.org/doc/Documentation/process/coding-style.rst)
(see [here](https://clang.llvm.org/docs/ClangFormatStyleOptions.html#examples))
without using tabs.

### Naming conventions

## Conclusion
Aside of what is enumerated above, we will be glad if you share with us your
setup, photos or just ideas.

