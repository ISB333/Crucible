# Crucible verifier images

Build once per machine:

    docker build -t crucible-py:0   -f docker/py.Dockerfile   docker/
    docker build -t crucible-lean:0 -f docker/lean.Dockerfile docker/
    docker build -t crucible-solidity:0 -f docker/solidity.Dockerfile docker/

Used by `DockerSandbox(image="crucible-py:0")` / `DockerSandbox(image="crucible-lean:0")` / `DockerSandbox(image="crucible-solidity:0")`.
Runs get `--network=none` unless the Task declares `network=True`.

## crucible-solidity:0

Foundry (forge/cast/anvil) with a pre-warmed solc 0.8.x, world-readable for non-root offline runs.
Used by the `Forge` verifier and `examples/solidity/`.

## crucible-chem:0

    docker build -t crucible-chem:0 -f docker/chem.Dockerfile docker/

RDKit + a baked deterministic SMILES scorer (`/opt/score_smiles.py`, ESOL-style logS).
Used by the `Chem` verifier and `examples/chem/`.
