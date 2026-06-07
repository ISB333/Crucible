FROM python:3.12-slim

# libatomic1 is required by pyright's pre-built node bundle on Debian/slim.
# Without it, pyright --version exits 127 due to missing libatomic.so.1.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libatomic1 \
 && rm -rf /var/lib/apt/lists/*

# Direct pyright to install its node bundle into /opt/pyright-nodeenv (world-readable)
# rather than the default $HOME/.cache/pyright-python/nodeenv (root-only).
# PYRIGHT_PYTHON_ENV_DIR is honoured at both install time and runtime.
# DockerSandbox.run passes --user uid:gid (non-root), so the bundle MUST be
# accessible to arbitrary users — chmod -R a+rX ensures that.
ENV PYRIGHT_PYTHON_ENV_DIR=/opt/pyright-nodeenv

# pyright's pip wrapper downloads its node bundle on first run — pre-warm at build
# time because DockerSandbox runs containers with --network=none.
RUN pip install --no-cache-dir "pytest>=8" "pyright>=1.1.380" \
 && pyright --version \
 && chmod -R a+rX /opt/pyright-nodeenv

WORKDIR /work
