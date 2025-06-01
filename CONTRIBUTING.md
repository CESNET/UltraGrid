# How to contribute to UltraGrid
First, thanks to contribute to UltraGrid by using UltraGrid, reporting bugs,
adding code or ideas. Further section contains basic information how to
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

Preferred way to report a bug is to open an issue on GitHub. If not possible,
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

The pull request should represent a coherent unit, like a single feature added.
More features should be separated to multiple pull request (of course if possible
and eg. not firmly bound together).

The contained commits should present clearly what was changed. Ideally,
if refactoring is done, it should not be mixed with functional changes
in a single commit, similarly for the code movement.

### Copyright and license

Code added to existing file must keep the license. Newly created files should be
3-clause BSD licensed.

## Coding standards
**TODO:** incomplete

### Style

Recommended style is modified LLVM clang-format style:
```
BasedOnStyle: LLVM
AlignArrayOfStructures: Left
AlignConsecutiveAssignments: true
AlignConsecutiveDeclarations: true
AlignConsecutiveMacros: true
AlignEscapedNewlines: DontAlign
AlwaysBreakAfterReturnType: TopLevelDefinitions
BreakBeforeBraces: Linux
Cpp11BracedListStyle: false
IndentWidth: 8
ReflowComments: true
SpaceAfterCStyleCast: true
````

The style is deduced from original _rtp/rtp.{c,h}_ formatting and it
is similar to [Linux Kernel
style](https://www.kernel.org/doc/Documentation/process/coding-style.rst)
(see [here](https://clang.llvm.org/docs/ClangFormatStyleOptions.html#examples))
without using tabs.

### Naming conventions

Logged module name (`MOD_NAME`) should be lower-case.

#### Macros

It is recommended for the function-like macros to be typed upper-case.

## Conclusion
Aside of what is enumerated above, we will be glad if you share with us your
setup, photos or just ideas.

