# Source this from your shell before running `uv sync` in a non-git checkout.
#   source scripts/dev-env.sh
#
# setuptools-scm normally reads the version from git tags. This tarball has no
# .git directory, so we hand it a pretend version per workspace member.
# Bump these if you sync to a newer MemMachine release.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MEMMACHINE_CLIENT=0.3.8
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MEMMACHINE_SERVER=0.3.8
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MEMMACHINE_COMMON=0.3.8
