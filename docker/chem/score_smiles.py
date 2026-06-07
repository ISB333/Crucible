"""Deterministic SMILES scorer for Crucible's molecular-property example.

Usage: python score_smiles.py <molecule.smi path> [scaffold_smarts]
Prints ONE JSON line: {"valid": bool, "score": float, "has_scaffold": bool, "error": str}

score = ESOL-style predicted aqueous log-solubility (logS); higher = more soluble.
Deterministic given a fixed RDKit version (baked into the image), so a Scored/Ok verdict
reproduces under fresh_reverify.
"""

# rdkit resolves only inside crucible-chem:0; the host never imports it (Constitution).
# pyright: reportMissingImports=false

import json
import sys


def read_smiles(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or "crucible:region" in s:
                continue
            return s
    return ""


def main() -> int:
    path = sys.argv[1]
    scaffold = sys.argv[2] if len(sys.argv) > 2 else ""
    smiles = read_smiles(path)
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski
    except Exception as exc:  # rdkit missing — degrade, do not crash
        print(json.dumps({"valid": False, "score": 0.0, "has_scaffold": False, "error": f"rdkit import: {exc}"}))
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(json.dumps({"valid": False, "score": 0.0, "has_scaffold": False, "error": f"invalid SMILES: {smiles!r}"}))
        return 0
    has_scaffold = True
    if scaffold:
        patt = Chem.MolFromSmarts(scaffold)
        has_scaffold = patt is not None and mol.HasSubstructMatch(patt)
    clogp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Lipinski.NumRotatableBonds(mol)
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    heavy = mol.GetNumHeavyAtoms() or 1
    ap = aromatic / heavy
    logs = 0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rb - 0.74 * ap
    print(json.dumps({"valid": True, "score": round(logs, 4), "has_scaffold": has_scaffold, "error": ""}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
