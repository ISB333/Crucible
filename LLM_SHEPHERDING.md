# LLM Shepherding — Guide pratique

**LLM shepherding** permet à un modèle "worker" faible/rapide de consulter un modèle "advisor" plus fort quand il est bloqué. C'est optionnel — le système fonctionne sans.

---

## Quick Start : gemma4:12-it-qat (Ollama) + Gemini advisor

### 1. Prérequis

```bash
# Ollama doit être installé et le modèle déjà pullé
ollama list  # doit montrer gemma4:12-it-qat

# Installer les dépendances Gemini
uv pip install "google-generativeai>=0.8"

# Exporter la clé API Gemini
export GOOGLE_API_KEY="votre-cle-ici"
# ou ajouter à .env : GOOGLE_API_KEY=votre-cle-ici
```

### 2. Lancer un run avec advisor

```bash
# CLI — avec advisor Gemini
crucible run problem.py \
  --editable solution \
  --verifier pytest:tests/ \
  --model ollama/gemma4:12-it-qat \
  --base-url http://localhost:11434/v1 \
  --advisor gemini-2.0-flash \
  --advisor-max-calls 5 \
  --advisor-fail-streak 3
```

**Ce que font les flags :**
- `--model ollama/gemma4:12-it-qat` : worker = gemma via Ollama (local, rapide, gratuit)
- `--base-url http://localhost:11434/v1` : endpoint OpenAI-compatible d'Ollama
- `--advisor gemini-2.0-flash` : advisor = Gemini (fort, paie à l'usage)
- `--advisor-max-calls 5` : max 5 consultations advisor sur tout le run
- `--advisor-fail-streak 3` : après 3 épisodes sans amélioration, l'advisor est consulté automatiquement

### 3. SDK — même chose en Python

```python
from crucible import AdvisorPolicy, Task, run, budgets

result = run(
    task=Task.from_path("problem.py", editable=["solution"]),
    verifier=...,  # ton verifier
    model="ollama/gemma4:12-it-qat",
    base_url="http://localhost:11434/v1",
    advisor=AdvisorPolicy(
        model="gemini-2.0-flash",  # advisor model
        max_calls_per_episode=1,   # 1 consult max par épisode
        max_calls_per_run=5,       # 5 consults max sur tout le run
        plateau_trigger=True,      # activer le trigger automatique
        fail_streak=3,             # après 3 échecs,咨询 advisor
        scope="suggestions",       # "suggestions" | "steering"
    ),
    workers=5,
    episode=budgets(edits=30, turns=15),
)
```

---

## Comment ça marche

### Deux déclencheurs

1. **Self-trigger** (manuel) : Le worker appelle `consult_advisor` quand il se sait bloqué.
2. **Engine-trigger** (automatique) : Le système consulte l'advisor après `fail_streak` épisodes sans amélioration.

### Caps (limites)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `max_calls_per_episode` | 1 | Max consultations par épisode |
| `max_calls_per_run` | None | Max consultations sur tout le run (None = illimité) |
| `fail_streak` | 3 | Nombre d'échecs avant engine-trigger |

### Graceful degradation

Si l'advisor est indisponible (rate limit, API down, clé invalide), le worker reçoit :
```
(advisor unavailable — proceed on your own)
```
Le run **ne plante pas** — il continue sans advisor.

---

## Configuration Ollama

### Models supportés

Tout modèle Ollama avec une interface OpenAI-compatible marche :

```bash
# Lister les modèles locaux
ollama list

# Puller un modèle si besoin
ollama pull gemma4:12-it-qat
```

### Nom du modèle

Dans Crucible, préfixer avec `ollama/` :

```bash
--model ollama/gemma4:12-it-qat
--model ollama/llama3.1:8b
--model ollama/mistral:7b
```

### Base URL

Ollama écoute par défaut sur `http://localhost:11434`. L'endpoint OpenAI-compatible est :

```bash
--base-url http://localhost:11434/v1
```

---

## Configuration Advisor

### Providers supportés

| Provider | Modèles | Env var |
|----------|---------|---------|
| Anthropic | `claude-opus-4-8`, `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Gemini | `gemini-2.0-flash`, `gemini-1.5-pro` | `GOOGLE_API_KEY` |
| OpenAI | `gpt-4o`, `o3-mini` | `OPENAI_API_KEY` |
| Ollama | `ollama/...` | (aucune) |

**Astuce** : Tu peux utiliser un modèle Ollama local comme advisor aussi :

```bash
--advisor ollama/llama3.1:8b
```

### Stratégies recommandées

| Worker | Advisor | Use case |
|--------|---------|----------|
| `ollama/gemma4:12-it-qat` | `gemini-2.0-flash` | Local-first, advisor cloud pour les blocages |
| `ollama/llama3.1:8b` | `claude-sonnet-4-6` | Rapide + fort |
| `gpt-4o-mini` | `gpt-4o` | Même provider, upgrade de modèle |
| `gemini-1.5-flash` | `gemini-2.0-flash` | Google-only |

---

## Exemples complets

### Exemple 1 : Sidon set avec gemma4 + Gemini advisor

```bash
cd /home/isb/Crucible

uv run python examples/sidon/run_sidon.py \
  --model ollama/gemma4:12-it-qat \
  --advisor gemini-2.0-flash \
  --advisor-max-calls 5 \
  --advisor-fail-streak 3 \
  --workers 5 \
  --target 70
```

### Exemple 2 : Chemistry avec gemma4 + Claude advisor

```bash
# Nécessite ANTHROPIC_API_KEY dans .env
uv run python examples/chem/run_chem.py \
  --model ollama/gemma4:12-it-qat \
  --advisor claude-sonnet-4-6 \
  --advisor-max-calls 3
```

### Exemple 3 : Kata personnalisé

```bash
# problem.py
# crucible:region start name=solution
def solve(x: int) -> int:
    raise NotImplementedError
# crucible:region end

# tests/test_solve.py
from problem import solve

def test_solve():
    assert solve(5) == 25  # on veut x²
```

```bash
crucible run problem.py \
  --editable solution \
  --verifier pytest:tests/ \
  --model ollama/gemma4:12-it-qat \
  --base-url http://localhost:11434/v1 \
  --advisor gemini-2.0-flash \
  --workers 3 \
  --episode-edits 30 \
  --run-budget 1h
```

---

## Inspecter les consultations advisor

### Voir les événements dans la DB

```bash
# Lister les runs
uv run crucible runs --db crucible.db

# Voir les consultations advisor pour un run
sqlite3 crucible.db "SELECT kind, payload_json FROM events WHERE kind='advisor_consult' ORDER BY id;"
```

### Voir le reasoning complet

```bash
# Dernier run
uv run crucible reasoning --db crucible.db

# Run spécifique
uv run crucible reasoning 7 --db crucible.db

# Filtrer par worker/épisode
uv run crucible reasoning --worker 0 --episode 2 --db crucible.db
```

---

## Dépannage

### "Advisor unavailable"

- Vérifier la clé API : `echo $GOOGLE_API_KEY`
- Tester l'endpoint : `curl https://generativelanguage.googleapis.com/v1beta/models -d "key=$GOOGLE_API_KEY"`
- Pour Ollama : `curl http://localhost:11434/api/tags`

### Ollama ne répond pas

```bash
# Redémarrer Ollama
sudo systemctl restart ollama
# ou
ollama serve  # en foreground
```

### Rate limit advisor

Réduire `--advisor-max-calls` ou augmenter `--advisor-fail-streak` pour consulter moins souvent.

---

## Coûts estimés (Gemini advisor)

| Modèle | Input | Output | ~1000 tokens |
|--------|-------|--------|--------------|
| `gemini-2.0-flash` | Gratuit (50k/jour) | Gratuit | $0 |
| `gemini-1.5-pro` | $0.000125 | $0.0005 | ~$0.10 |

**Astuce** : `gemini-2.0-flash` est gratuit dans les limites quotidiennes — parfait pour l'advisor.

---

## Résumé des commandes

```bash
# Worker local (Ollama) + advisor cloud (Gemini)
crucible run problem.py \
  --editable solution \
  --verifier pytest:tests/ \
  --model ollama/gemma4:12-it-qat \
  --base-url http://localhost:11434/v1 \
  --advisor gemini-2.0-flash \
  --advisor-max-calls 5 \
  --advisor-fail-streak 3 \
  --workers 5
```

**C'est tout** — le worker itère localement, et consulte Gemini uniquement quand il est vraiment bloqué.
