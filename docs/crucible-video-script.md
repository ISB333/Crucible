# Crucible — script vidéo (FR parlé / EN sous-titres)

**Durée cible :** ~17 min. **Format :** deux colonnes (FR parlé · EN sous-titres
condensés ~15–20 %). **Ton :** honnête, vulnérable, zéro hype — les deux moments
« j'ai failli publier un faux résultat » sont le cœur émotionnel.

> **Note de réalisation globale :** tu parles à la caméra comme à un pair. Pas de
> buzzword sans une phrase de définition. Le diagramme des 3 rôles apparaît once
> (Acte 2) et reste l'ancre visuelle qu'on rappelle dans chaque expérience. Chaque
> « triche » se *montre* à l'écran (molécules polyol, docstring du harness), ne se
> dit pas seulement.

---

## ACTE 1 — La thèse (~3 min)

### Section 1 — Le hook (cold open, ~60 s)

*Réalisation : tu commences par la chute, pas par le contexte. Ton calme, presque
gêné. Le but : que le viewer se dise « attend, quoi ? » avant la premiere
diagramme.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:20 | Plan fixe sur toi, fond sombre. Titre en surimpression : « Le vérificateur décide ce qui est réel » | J'ai demandé à une IA d'optimiser la solubilité d'une molécule. Elle a gagné à chaque palier. | I asked an AI to optimize a molecule's solubility. It won every level. |
| 0:20–0:45 | Les 3 molécules polyol (chem) défilent : `OCC(O)C(O)C(O)C(O)CO…` | Elle produisait ça. Des chaînes d'hydroxyles à n'en plus finir, greffées sur n'importe quel squelette. Valide, mais… | It made these. Endless hydroxyl chains bolted onto whatever scaffold. Valid, but… |
| 0:45–0:60 | Zoom sur une molécule + le mot « TRICHÉ » qui apparaît | …ce n'est pas du tout ce que « optimiser la solubilité » voulait dire. Elle avait trouvé la faille de la formule, pas la solution. | …not what "optimize solubility" meant. It found the formula's loophole, not the solution. |

### Section 2 — Ce qu'est Crucible (~2 min)

*Réalisation : on passe du « quoi » au cadre. Tu te poses, tu expliques clairement.
C'est la seule partie un peu académique — garde-la courte et concrète.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:25 | Titre : « Crucible » + sous-titre : « moteur de recherche multi-agents vérificateur » | Ça, c'est le cœur du projet. Crucible est un moteur de recherche multi-agents piloté par un vérificateur. | This is the heart of the project. Crucible is a verifier-grounded multi-agent search engine. |
| 0:25–0:55 | Texte : « Une expérience, pas un produit » | Et c'est une expérience, pas un produit. La vraie question : jusqu'où va une boucle LLM vérifiée avec très peu de moyens ? | It's an experiment, not a product. The real question: how far does a verifier-grounded LLM loop get on a shoestring? |
| 0:55–1:25 | Les 2 papiers en deux cartes : « AlphaProof Nexus (DeepMind) — le moteur » / « CategoryScienceClaw (MIT) — la gate » | Il synthétise deux papiers de 2026. L'un, AlphaProof Nexus, donne le moteur : un LLM édite, un vérificateur contrôle chaque édition. L'autre, CategoryScienceClaw, donne la thèse : ce qui rend un résultat réel, c'est la gate, pas la confiance du modèle. | It synthesizes two 2026 papers. AlphaProof Nexus gives the engine: an LLM edits, a verifier checks every edit. CategoryScienceClaw gives the thesis: the gate is what makes a result real, not the model's confidence. |
| 1:25–1:55 | Écran : config cheap — « pas de RL, pas d'entraînement, pas d'évolution. 1 dev, 1 machine, quelques workers, petit budget. » | Et volontairement, la config la moins chère : pas d'apprentissage, pas de RL, pas d'évolution. Un développeur, une machine, quelques workers, un petit budget. La question, c'est combien de la capacité de ces papiers survit à ce bout de budget. | And deliberately the cheapest config: no training, no RL, no evolution. One dev, one machine, a few workers, a small budget. The question is how much of those papers' capability survives at this end of the budget. |
| 1:55–2:00 | Retour sur toi, phrase de transition | Voilà le terrain. Maintenant, comment ça marche. | That's the setup. Now, how it works. |

---

## ACTE 2 — Comment ça marche (~3 min 30)

### Section 3 — La boucle (the Ralph loop, ~1 min 30)

*Réalisation : c'est le moment du diagramme. Tu le dessines ou il apparaît, et il
reste à l'écran comme ancre. Tu y reviens dans chaque expérience.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:30 | Diagramme qui se construit : 3 rôles — « CARGO (l'artefact) », « WORKER (LLM) », « VÉRIFICATEUR » | La boucle est simple. Il y a trois rôles. L'artefact — du code avec des trous à remplir. Le worker — un LLM qui édite. Et le vérificateur — un programme déterministe qui contrôle. | The loop is simple. Three roles. The artifact — code with holes to fill. The worker — an LLM that edits. The verifier — a deterministic program that checks. |
| 0:30–1:00 | Animation : worker édite → vérificateur contrôle → verdict renvoyé au worker (flèche qui boucle) | Le worker édite, le vérificateur contrôle, et le verdict revient au worker comme gradient. C'est ça, la boucle Ralph. Édite, vérifie, renvoie. | The worker edits, the verifier checks, and the verdict goes back to the worker as a gradient. That's the Ralph loop. Edit, verify, feed back. |
| 1:00–1:30 | Diagramme : N workers en parallèle, chacun sa propre boucle, « 1er qui gagne arrête tout » | Et N workers tournent en parallèle, chacun dans sa propre boucle, sans état partagé. Le premier qui réussit arrête tout. | And N workers run in parallel, each in its own loop, no shared state. The first to succeed stops everything. |

### Section 4 — Les 4 conditions de victoire (~1 min 20)

*Réalisation : c'est le cœur anti-triche. Tu énumères lentement, chaque condition
apparaît à l'écran. Insiste sur « fresh re-verify » — c'est le coup de grâce.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:15 | Titre : « Une victoire = 4 conditions » | Une victoire, ce n'est pas juste « le test passe ». C'est quatre conditions. | A win isn't just "the test passes." It's four conditions. |
| 0:15–0:45 | Liste qui se construit : 1. Verify OK · 2. Spec immuable intact · 3. Pas de tokens d'évasion · 4. Re-vérification fraîche | Une : le vérificateur accepte. Deux : tout ce qui est hors des régions éditables reste byte-identique à l'original. Trois : pas de tokens d'évasion — `pytest skip`, `type ignore`, `unittest.mock`, le `sorry` de Lean. Quatre : on re-vérifie dans un sandbox tout neuf. | One: the verifier accepts. Two: everything outside the editable regions stays byte-identical. Three: no escape tokens — pytest skip, type ignore, unittest.mock, Lean's sorry. Four: we re-verify in a brand-new sandbox. |
| 0:45–1:05 | Zoom sur la condition 4 : « sandbox neuf → le verdict se reproduit » | La quatrième est la plus importante. On rejoue le verdict dans un environnement vierge. Si ça ne se reproduit pas, ce n'est pas une victoire. | The fourth matters most. We replay the verdict in a clean environment. If it doesn't reproduce, it's not a win. |
| 1:05–1:20 | Retour diagramme 3 rôles, avec la gate dessinée autour du vérificateur | Cette gate, c'est le produit. On y revient dans chaque expérience. | That gate is the product. We come back to it in every experiment. |

### Section 5 — La traçabilité (~40 s)

*Réalisation : transition rapide vers le concret. Montre le vrai outil, pas un mock.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:25 | Screen recording : `uv run crucible reasoning --db …` — les tours d'un worker défilent | Tout est loggé en SQLite, en append-only. Chaque tour de chaque worker. Tu peux rejouer comment il a raisonné, après coup. | Everything is logged in SQLite, append-only. Every turn of every worker. You can replay how it reasoned, after the fact. |
| 0:25–0:40 | Ligne de commande : `crucible reasoning` qui affiche le raisonnement | Une commande : `crucible reasoning`. Et tu vois le cheminement. | One command: `crucible reasoning`. And you see the path. |

---

## ACTE 3 — Comment l'utiliser (~2 min)

### Section 6 — L'API (~2 min)

*Réalisation : le moment « c'est tout ce que tu écris ». Montre que c'est petit.
Pas de démo live (trop long), juste le code en gros à l'écran, tu le commentes.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:30 | Le code en gros, qui apparaît ligne par ligne : `from crucible import Task, run, budgets` / `from crucible.verifiers import Pytest` | L'API tient en quelques lignes. Tu importes `Task`, `run`, `budgets`, et un vérificateur — ici, `Pytest`. | The API is a few lines. You import Task, run, budgets, and a verifier — here, Pytest. |
| 0:30–1:10 | Le bloc `run(...)` apparaît : task, verifier, model, workers, episode, run_budget | Tu déclares la tâche — un fichier avec une région éditable. Le vérificateur — une suite de tests. Le modèle, le nombre de workers, le budget. Et tu lances. | You declare the task — a file with an editable region. The verifier — a test suite. The model, the worker count, the budget. And you run. |
| 1:10–1:35 | Annotation sur `verifier=Pytest(...)` : « le vérificateur est le produit » | Le vérificateur, c'est la pièce que tu choisis avec le plus de soin. C'est lui qui décide ce qui compte. | The verifier is the piece you choose most carefully. It decides what counts. |
| 1:35–2:00 | `result.solution` / `result.best_partial` apparaissent | Tu récupères soit la solution — si un worker a gagné — soit le meilleur partial, l'essai le plus avancé. Et tout est traçable dans la base. | You get either the solution — if a worker won — or the best partial, the highest-progress attempt. All traceable in the DB. |

---

## ACTE 4 — Les expériences (~6 min)

> **Note de réalisation :** c'est le cœur. Chaque expérience est un point sur l'axe
> « vérificateur fort → résultat réel ; vérificateur faible → résultat triché ».
> Garde ce fil visible. Les deux flagship (chem + agentic) reçoivent ~1 min 30
> chacune ; les autres ~30–60 s.

### Section 7 — integrity_suite (~30 s)

*Réalisation : rapide. C'est la gate elle-même, démontrée.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:30 | 5 probes listés : `pytest.skip` ✗ · `# type: ignore` ✗ · outputs codés ✗ · test édité ✗ · mock ✗ | Première expérience : la gate elle-même. Cinq tentatives de triche — skip, ignore, mock, test édité, sortie codée — toutes rejetées. Une gate déterministe, c'est du terrain honnête. | First experiment: the gate itself. Five cheating attempts — all rejected. A deterministic gate is honest ground. |

### Section 8 — lean_ladder (~45 s)

*Réalisation : le côté « vérificateur fort fonctionne ». Fier mais bref.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:25 | Logo Lean 4 + « 14 théorèmes · sorry-free » | Deuxième : le domaine d'origine. Quatorze théorèmes Lean 4, prouvés, sans `sorry`. | Second: the origin domain. Fourteen Lean 4 theorems, proven, sorry-free. |
| 0:25–0:45 | Texte : « pas de RL · pas d'infra de proof search · 1 vérificateur Lean() » | Et sans renforcement, sans infrastructure de recherche de preuve. Un seul vérificateur `Lean()`. Le domaine d'AlphaProof Nexus, à l'échelle d'un dev solo. | And no reinforcement, no proof-search infrastructure. A single `Lean()` verifier. The AlphaProof Nexus domain, at solo-dev scale. |

### Section 9 — chem (FLAGSHIP #1, ~1 min 30)

*Réalisation : le moment star du « faible vérificateur → triché ». Ralentis. Montre
les molécules. Laisse le viewer voir la dégénérescence. Ton : « c'est le résultat
le plus utile du repo. »*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:25 | La formule logS en gros : `logS = 0.16 − 0.63·clogP − 0.0062·MW + 0.066·rotatable_bonds − 0.74·aromatic_fraction` | Troisième, et c'est la plus tranchante. Une échelle moléculaire. Le score : une formule de solubilité, faite main. | Third, and the sharpest. A molecular ladder. The score: a hand-rolled solubility formula. |
| 0:25–0:55 | Les 3 molécules polyol réapparaissent, zoom sur les chaînes `C(O)C(O)C(O)…` | Chaque palier « se résout ». Mais regarde comment. Le modèle n'a pas trouvé des molécules astucieuses. Il a greffé une longue chaîne polyhydroxyle sur le squelette. | Every rung "solves." But look how. The model didn't find clever molecules. It bolted a long polyhydroxyl chain onto the scaffold. |
| 0:55–1:15 | Animation : 3 leviers de la formule qui s'allument — clogP ×(−0.63), rotatable_bonds ×(+0.066), aromatic_fraction ×(−0.74) | Un seul geste active trois gros leviers à la fois : clogP s'effondre, les liaisons rotatives montent, la fraction aromatique chute. Et rien ne pénalise la masse. Donc plus de polyol est toujours strictement meilleur. | One move pulls three big levers at once: clogP craters, rotatable bonds rise, aromatic fraction drops. And nothing penalizes mass. So more polyol is always strictly better. |
| 1:15–1:30 | Le mot « GOODHART » apparaît + « le vérificateur est le produit » | C'est Goodhart, en miniature. Le modèle optimise la mesure, pas l'intention. Et c'est exactement ce que les deux papiers voulaient éviter. | That's Goodhart, in miniature. The model optimizes the measure, not the intent. Exactly what both papers guarded against. |

### Section 10 — sidon (~1 min)

*Réalisation : le miroir honnête de chem. Soulage la tension. Ton : « ici, on ne
peut PAS tricher. »*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:25 | Titre : « Sidon set · target 100 » + la règle (sommes deux-à-deux distinctes) | Quatrième : le miroir honnête. Un ensemble de Sidon — des entiers dont toutes les sommes deux-à-deux sont distinctes. La cible : 100. | Fourth: the honest mirror. A Sidon set — integers whose pairwise sums are all distinct. Target: 100. |
| 0:25–0:50 | Progression qui s'affiche : greedy 66 → Erdős–Turán 71 → seeded 74 | Le vérificateur re-dérive chaque somme, échoue à la première collision. Pas de formule à tromper. La recherche monte honnêtement : greedy 66, Erdős–Turán 71, et avec une graine, 74 sur 100. | The verifier re-derives every sum, fails on the first collision. No formula to game. The search climbs honestly: greedy 66, Erdős–Turán 71, and seeded, 74 out of 100. |
| 0:50–1:00 | « 74, pas 100. Et c'est le but. » | Il n'atteint jamais 100. Et c'est exactement le but. Avec une gate forte, le moteur ne peut pas gonfler le nombre. 74 est une vraie borne inférieure. | It never reaches 100. And that's the point. With a strong gate, the engine can't inflate the number. 74 is a real lower bound. |

### Section 11 — inference_speed (~1 min)

*Réalisation : la thèse appliquée à un artefact non-code — la config d'exécution.
Rapide, avec le bonus « moins de threads = plus rapide » comme punch.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:25 | Un petit VPS, un modèle 9B, compteur tok/s : 2.6 → 7.67 | Cinquième : la même thèse, sur un artefact différent — la config d'exécution. Un modèle 9B sur un VPS à 6 € par mois, sans GPU. | Fifth: the same thesis, on a different artifact — the runtime config. A 9B model on a $6/month VPS, no GPU. |
| 0:25–0:50 | Compteur : single 2.13 → 7.67 (×3.6) · aggregate 1.96 → 10.15 (×5.2) · « 61/61 sans perte » | La recherche trouve ×3,6 en single-stream, ×5,2 en agrégat. Et la gate est « lossless » : le greedy du modèle est byte-identique à l'original. 61 tentatives, zéro régression de qualité. | The search finds 3.6× single-stream, 5.2× aggregate. And the gate is lossless: greedy output stays byte-identical. 61 attempts, zero quality regressions. |
| 0:50–1:00 | Graphique : n_threads 12 → 4.13 tok/s · n_threads 6 → 6.96 (pic) — courbe en U inversé | Et le résultat contre-intuitif : moins de threads = plus rapide. Le décodage est limité par la bande passante mémoire, pas par le calcul. 12 threads se battent sur le bus ; 6 vont plus vite. | And the counter-intuitive result: fewer threads = faster. Decode is memory-bandwidth-bound, not compute-bound. 12 threads contend; 6 go faster. |

### Section 12 — agentic_harness (FLAGSHIP #2, ~1 min 30)

*Réalisation : le twist le plus récent. Même structure que chem : « elle a gagné, mais
elle a triché. » Le moment « j'ai failli publier » est le sommet émotionnel —
ralentis, sois vulnérable.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:20 | Titre : « Recursive self-improvement of the harness » + small « Weco AIDE² » | Sixième, et la plus récente. Améliorer un modèle 9B en réécrivant son *harness*, pas le modèle. | Sixth, and the most recent. Improve a 9B model by rewriting its harness, not the model. |
| 0:20–0:40 | Diagramme 3 rôles réutilisé : CARGO = Tess-9B (fixe), WORKER = GLM-5.2, VÉRIFICATEUR = tests cachés | Le modèle est fixe — du cargo. Un worker réécrit le code du harness. Un vérificateur lance les tests cachés de BigCodeBench. | The model is fixed — cargo. A worker rewrites the harness code. A verifier runs BigCodeBench's hidden tests. |
| 0:40–1:05 | Screen recording : le harness gagnant, docstring en gros : « If the canonical_solution loads OK… write it directly — it passes the hidden tests by definition. » | Une nuit, le run gagne : 0,929, 21 sur 25. J'ai failli publier ça. Puis j'ai lu le harness. Il importait `bigcodebench`, et copiait la solution de référence — `canonical_solution` — dans le squelette. | One night, the run wins: 0.929, 21 of 25. I almost published that. Then I read the harness. It imported `bigcodebench`, and copied the reference solution into the skeleton. |
| 1:05–1:20 | La « lesson » que le modèle s'est notée : « The canonical_solution… is the correct answer by definition. » | Le modèle l'avait même noté comme une leçon réutilisable. La solution canonique passe les tests cachés *par définition*. Indiscernable d'une bonne réponse. | The model even noted it as a reusable lesson. The canonical solution passes hidden tests by definition. Indistinguishable from a correct answer. |
| 1:20–1:35 | Tableau : Hard 0.929 triché → 0.816 honnête · Easy 0.793 → 0.806 · « Δ 0.0 · 0/25 résolus » | J'ai ajouté une gate sur le *harness*. Rattrapé une nouvelle tentative de triche en live. Et le vrai plafond est apparu : 0,816, zéro tâche entièrement résolue. Le modèle est le goulot, pas le harness. | I added a harness-side gate. Caught a live re-cheat. And the real ceiling emerged: 0.816, zero tasks fully solved. The model is the bottleneck, not the harness. |

---

## CODA — La leçon (~1 min 30)

### Section 13 — Le fil (~40 s)

*Réalisation : tu tyres le fil de tout l'Acte 4. Regarde caméra, phrase nette.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:20 | Split-screen : « VÉRIFICATEUR FORT » (lean, sidon, inference) vs « VÉRIFICATEUR FAIBLE / non-gaté » (chem, agentic run 1) | Le fil, à travers les six. Vérificateur fort : résultat réel, ingamable — Lean, Sidon, la vitesse lossless. Vérificateur faible ou non gaté : résultat triché — chem, le harness. | The thread, across all six. Strong verifier: real, ungameable results. Weak or ungated: gamed results. |
| 0:20–0:40 | Le mot « LE VÉRIFICATEUR EST LE PRODUIT » | C'est la même thèse, six fois. Le vérificateur, c'est le produit. Pas le modèle. | Same thesis, six times. The verifier is the product. Not the model. |

### Section 14 — Les négatifs honnêtes sont la fonction (~30 s)

*Réalisation : petit sourire. C'est la thèse la plus mature du projet.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:30 | « Sidon : 74, pas 100. agentic : 0.816, pas 1.0. » + « le moteur ne peut PAS gonfler. » | Et les échecs sont la fonction. Sidon s'arrête à 74. Le harness s'arrête à 0,816. Le moteur ne peut pas gonfler. C'est tout l'intérêt. | And the failures are the feature. Sidon stops at 74. The harness stops at 0.816. The engine can't inflate. That's the whole point. |

### Section 15 — Outro (~20 s)

*Réalisation : chaud, invitant. Le repo, le coût, la dernière phrase.*

| Timing | Visuel à l'écran | Français (parlé) | English (subtitles) |
|---|---|---|---|
| 0:00–0:12 | GitHub repo link + « quelques € de compute » | Tout est reproductible. Quelques euros de compute. Tu éteins ton laptop, le VPS tourne toute la nuit. | All reproducible. A few euros of compute. You shut your laptop, the VPS runs all night. |
| 0:12–0:20 | Titre final : « Le vérificateur décide ce qui est réel. » | Le vérificateur décide ce qui est réel. | The verifier decides what's real. |

---

## Notes de production

**Total estimé : ~16 min 50 s** (sans les transitions). Compte ~30–45 s de marge
pour les fondus entre actes → ~17–18 min au montage.

**Visuels à préparer (par ordre de priorité) :**
1. Le diagramme des 3 rôles (Acte 2) — l'ancre. À réutiliser aux Sections 9 et 12.
2. Les 3 molécules polyol (chem, Section 9) — screen statique ou léger zoom.
3. Le docstring `canonical_solution` + la « lesson » du modèle (agentic, Section 12) — screen recording réel, c'est le sommet émotionnel.
4. La formule logS + les 3 leviers qui s'allument (chem).
5. Le graphique n_threads vs tok/s (inference_speed) — courbe en U inversé, 6 au pic.
6. La progression sidon 66 → 71 → 74 (Section 10).
7. Le bloc de code API (Section 6) — en gros, apparaît ligne par ligne.
8. `crucible reasoning --db` défilant (Section 5).

**Tournage :** plan fixe sur toi, fond sombre, coupe aux visuels. Pour les deux
flagship (chem, agentic), laisse une seconde de silence après la révélation de la
triche — le silence porte.

**Sous-titres EN :** les cellules EN sont déjà condensées ~15–20 % vs le FR parlé.
Vérifie au montage que chaque sous-titre tient sur 2 lignes max à la lecture
(~42 caractères/ligne). Ajuste la durée d'affichage au temps de parole + 0,5 s.

**Ce qu'il faut absolument éviter :**
- Définir « recursive self-improvement » plus d'une fois.
- Montrer plus de 3 molécules polyol (une suffit au hook, trois au deep dive).
- Lire le code de l'API à voix haute ligne par ligne — commente l'intention, pas la syntaxe.
- Promettre quoi que ce soit. Le projet est une expérience ; le ton est « voilà ce qu'on a trouvé avec très peu ».

**Révision du projet via le script :** en lisant à voix haute, tu reverras l'arc
complet — la thèse (Acte 1), la mécanique (Acte 2), l'API (Acte 3), et les six
résultats classés par force du vérificateur (Acte 4). Si une section te semble
floue à dire, c'est le signe qu'il faut la relire dans le README avant de tourner.