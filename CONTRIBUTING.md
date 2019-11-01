# How to contribute to UltraGrid
First, thanks to contribute to UltraGrid by using UltraGrid, reporting bugs,
adding code or ideas. Further section contains basic informations how to
contribute. Feel free to inform us if you need to contribute in a different
way or extend this document.

## Reporting bugs
You can either fill an issue at GitHub (preferred) or contact our development
team directly with the e-mail. If you suspect that the issue may not be always
replicable, you can use a script `ultragrid-bugreport-collect.sh` to collect
data about a computer and attach its result.

## Feature requests
Feature requests can be submitted with same channels as bugs.

## Coding standards
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

## Documentation
Project documenataion is done mainly using Doxygen.

Other pieces of documentation can be found
[in GitHub wiki](https://github.com/CESNET/UltraGrid/wiki/Developer-Documentation).
There is also a
[Trello board](https://trello.com/b/PjZW4sas/ultragrid-development) tracking the
development.

## Conclusion
Aside of what is enumerated above, we will be glad if you share with us your
setup, photos or just ideas.

