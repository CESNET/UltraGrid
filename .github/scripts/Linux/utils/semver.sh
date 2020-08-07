#!/bin/sh

#https://gist.github.com/akutz/4bf84cce21dfb49dd55ca19014e2668f

#
# usage: semver.sh A B
#
#   Compares two semantic version strings and prints:
#
#     -1 if A<B
#      0 if A=B
#      1 if A>B
#
#   A non-zero exit code means there was an error comparing the
#   provided values.
#
#   The script may also be executed with TEST_SEMVER_COMP=1 to
#   run several unit tests that validate the comparison logic.
#   For example:
#
#        $ TEST_SEMVER_COMP=1 semver.sh 
#                      a               b   exp   act
#                 v2.1.3         2.1.3.0     0     0
#                  v10.0            v1.0     1     1
#                   v1.0           v10.0    -1    -1
#

set -e
#set -o pipefail

# A basic regex pattern for matching a semantic version string.
semverPatt='^\(v\{0,1\}\)\([[:digit:]]\{1,\}\)\{0,1\}\(.[[:digit:]]\{1,\}\)\{0,1\}\(.[[:digit:]]\{1,\}\)\{0,1\}\(.[[:digit:]]\{1,\}\)\{0,1\}\(.\{0,\}\)$'

# Returns a successful exit code IFF the provided string is a semver.
is_semver() {
  echo "${1}" | grep -q "${semverPatt}"
}

# Extracts the MAJOR component of a semver.
get_major() {
  echo "${1}" | sed -e 's/'"${semverPatt}"'/\2/g'
}
# Extracts the MINOR component of a semver.
get_minor() {
  _v=$(echo "${1}" | sed -e 's/'"${semverPatt}"'/\3/g' | tr -d '.')
  [ -n "${_v}" ] || _v=0; echo "${_v}"
}
# Extracts the PATCH component of a semver.
get_patch() {
  _v=$(echo "${1}" | sed -e 's/'"${semverPatt}"'/\4/g' | tr -d '.')
  [ -n "${_v}" ] || _v=0; echo "${_v}"
}
# Extracts the BUILD component of a semver.
get_build() {
  _v=$(echo "${1}" | sed -e 's/'"${semverPatt}"'/\5/g' | tr -d '.')
  [ -n "${_v}" ] || _v=0; echo "${_v}"
}
# Extracts the SUFFIX component of a semver.
get_suffix() {
  echo "${1}" | sed -e 's/'"${semverPatt}"'/\6/g'
}
# Extracts the MAJOR.MINOR.PATCH.BUILD portion of a semver.
get_major_minor_patch_build() {
  printf '%d.%d.%d.%d' \
    "$(get_major "${1}")" \
    "$(get_minor "${1}")" \
    "$(get_patch "${1}")" \
    "$(get_build "${1}")"
}

# Returns 0 if $1>$2
version_gt() {
  test "$(printf '%s\n' "${@}" | sort -V | head -n 1)" != "${1}"
}

# Compares two semantic version strings:
#  -1 if a<b
#   0 if a=b
#   1 if a>b
semver_comp() {
  is_semver "${1}" || { echo "invalid semver: ${1}" 1>&2; return 1; }
  is_semver "${2}" || { echo "invalid semver: ${2}" 1>&2; return 1; }

  # Get the MAJOR.MINOR.PATCH.BUILD string for each version.
  _a_mmpb="$(get_major_minor_patch_build "${1}")"
  _b_mmpb="$(get_major_minor_patch_build "${2}")"

  # Record whether or not the two MAJOR.MINOR.PATCH.BUILD are equal.
  [ "${_a_mmpb}" = "${_b_mmpb}" ] && _a_eq_b=1

  # Get the suffix components for each version.
  _a_suffix="$(get_suffix "${1}")"
  _b_suffix="$(get_suffix "${2}")"

  # Reconstitute $1 and $2 as $_va and $_vb by filling in any
  # components missing from the original semver values.
  _va="${_a_mmpb}${_a_suffix}"
  _vb="${_b_mmpb}${_b_suffix}"

  # If the two reconstituted version strings are equal then the versions
  # are equal.
  if [ "${_va}" = "${_vb}" ]; then
    _result=0

  # If neither version have a suffix or if both versions have a suffix
  # then the versions may be compared with sort -V.
  elif { [ -z "${_a_suffix}" ] && [ -z "${_b_suffix}" ]; } || \
     { [ -n "${_a_suffix}" ] && [ -n "${_b_suffix}" ]; }; then
    { version_gt "${_va}" "${_vb}" && _result=1; } || _result=-1

  # If $1 does not have a suffix and the two MAJOR.MINOR.PATCH.BUILD
  # version strings are equal, then $1>$2.
  elif [ -z "${_a_suffix}" ] && [ -n "${_a_eq_b}" ]; then
    _result=1

  # If $1 does have a suffix and the two MAJOR.MINOR.PATCH.BUILD
  # version strings are equal, then $1<$2.
  elif [ -n "${_a_suffix}" ] && [ -n "${_a_eq_b}" ]; then
    _result=-1

  # Otherwise compare the two versions using sort -V.
  else
    { version_gt "${_va}" "${_vb}" && _result=1; } || _result=-1
  fi

  echo "${_result}"
}

[ -z "${TEST_SEMVER_COMP}" ] && { semver_comp "${@}"; exit "${?}"; }

printf '%-35s %-35s % 5s % 5s\n' 'a' 'b' 'exp' 'act'
test_semver_comp() {
  result="$(semver_comp "${1}" "${2}")"
  printf '%-35s %-35s % 5d % 5d\n' \
    "${1}" "${2}" "${3}" "${result}"
}

test_semver_comp v1.0 v1.0 0
test_semver_comp v2.1.3 v2.1.3.0 0
test_semver_comp v2.1.3 2.1.3.0 0
test_semver_comp v10.0 v1.0 1
test_semver_comp v1.0 v10.0 -1
test_semver_comp v1.14 v1.14.alpha1 1
test_semver_comp v1.14 v1.14.0.alpha1 1
test_semver_comp v1.14.0 v1.14.alpha1 1
test_semver_comp v1.14.alpha1 v1.14 -1
test_semver_comp 1.14.alpha1 v1.14 -1
test_semver_comp v1.140.alpha1 v1.14 1
test_semver_comp v1.14 v1.14.0-alpha.1.363 1
test_semver_comp v1.14.0-alpha.1.363+8bce3620b02b2a 1.14 -1

exit 0

