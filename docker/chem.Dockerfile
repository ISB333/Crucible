FROM python:3.12-slim
# RDKit wheels are self-contained; pre-import at build so the first offline run is warm.
RUN pip install --no-cache-dir "rdkit>=2024.3" \
 && python -c "import rdkit; from rdkit import Chem; print(rdkit.__version__)"
COPY chem/score_smiles.py /opt/score_smiles.py
RUN chmod a+rX /opt/score_smiles.py
WORKDIR /work
