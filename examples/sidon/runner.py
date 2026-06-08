"""Dev helper: run this locally to test problem.py output. Not part of the Crucible task."""
import json
import sys

sys.path.insert(0, ".")
from problem import generate_sidon_set

result = generate_sidon_set()
print(json.dumps(result))
