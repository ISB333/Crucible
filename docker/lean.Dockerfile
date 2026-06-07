FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# elan-init.sh URL verified 2026-06-06 against https://github.com/leanprover/elan (README)
# and https://elan.lean-lang.org/elan-init.sh (confirmed real bootstrap script, #!/bin/sh).
# The canonical install URL is https://elan.lean-lang.org/elan-init.sh (not the GitHub raw URL).
#
# Lean 4 stable tag verified 2026-06-06 from https://github.com/leanprover/lean4/releases:
# latest stable = leanprover/lean4:v4.30.0 (released 2025-05-26).
# v4.31.0-rc1 is a pre-release — excluded per plan requirement for explicit stable pin.
#
# DockerSandbox.run passes --user uid:gid (non-root), so elan toolchain MUST be
# world-readable/executable. Mitigation:
#   - ELAN_HOME=/opt/elan  (install to a system path, not /root/.elan)
#   - After install: chmod -R a+rX /opt/elan  (make readable by all users)
#   - ENV PATH includes /opt/elan/bin  (persistent for all users at runtime)
#   - elan needs a writable HOME at runtime; non-root smoke test sets -e HOME=/tmp.
ENV ELAN_HOME=/opt/elan
ENV PATH="/opt/elan/bin:${PATH}"

RUN curl -sSfL https://elan.lean-lang.org/elan-init.sh \
    | sh -s -- -y --default-toolchain leanprover/lean4:v4.30.0 \
 # elan defers toolchain download to first `lean` invocation — force it now so the
 # image is fully self-contained and DockerSandbox's --network=none containers work.
 && /opt/elan/bin/elan toolchain install leanprover/lean4:v4.30.0 \
 && chmod -R a+rX /opt/elan

WORKDIR /work
