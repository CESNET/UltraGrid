# How to contribute to UltraGrid
First, thanks to contribute to UltraGrid by using UltraGrid, reporting bugs,
adding code or ideas. Further section contains basic informations how to
contribute. Feel free to inform us if you need to contribute in a different
way or extend this document.

## Table of Contents

* [Table of contents](#table-of-contents)
* [Resources](#resources)
* [Reporting bugs](#reporting-bugs)
* [Feature requests](#feature-requests)
* [Pull requests](#pull-requests)
* [Coding standards](#coding-standards)
   * [Indenting and whitespaces](#indenting-and-whitespaces)
   * [Control structures](#control-structures)
   * [Spaces](#spaces)
   * [Line length and wrapping](#line-length-and-wrapping)
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

### Indenting and whitespaces
- Default indent is 8 spaces width and consists solely of spaces, no tabs.
- Lines should be LF-terminated and there should not contain any whitespaces at the end.

### Control structures
Control structures like if, while, switch etc. should have opening block bracket at the
same line as the statement itself:
```
if (foo) {
        do_something();
} else {
        do_something_else();
}
```

Functions and methods, on the other hand, should have opening curly braces at
the following line.

### Spaces
### Line length and wrapping
### Naming conventions

## Conclusion
Aside of what is enumerated above, we will be glad if you share with us your
setup, photos or just ideas.

