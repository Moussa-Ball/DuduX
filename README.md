# Dudux — Nano Neural Network binaire 0/1

Dudux est une base C++/CUDA minimaliste pour construire des réseaux et mémoires associatives binaires 1‑bit (poids et activations ∈ {0,1}), optimisés pour la mémoire et l’énergie via bit‑packing et popcount. Le dépôt propose des primitives d’attention binaire, une multi‑tête (MHA) partageant K/V, un routeur de candidats léger, des métriques BitOPs, et un backend CUDA optionnel avec exécution multi‑stream.

— Inspiré par le cerveau, pensé pour ≤4 GB VRAM et des déploiements contraints.

## Points clés
- 1‑bit strict en avant: AND + popcount, accumulateurs entiers, aucun float en chemin critique (hors échelles/τ).
- Bit‑packing en uint64 pour clés/valeurs; popcount vectorisable (CPU) et `__popcll` (CUDA) côté GPU.
- Attention binaire: similarité = popcount(q & k), top‑k streaming (tas O(k)).
- MHA 1‑bit: H têtes partageant le même K/V; variantes non‑pondérée/pondérée/masquée.
- GPU optionnel (CUDA): kernels AND+POPC, top‑k device (Thrust), buffers device persistants, multi‑stream (1 flux par tête).
- Gating (Candidates): routeur « postings » génère un sous‑ensemble d’indices à scorer (CAND) pour réduire compute/mémoire.
- Métriques: compteur global des popcounts, empreinte mémoire, VRAM (bench), latence.
- Tests unitaires CTest; benchmark reproductible.

## Structure du dépôt
- `cpp/` C++17 + (optionnel) CUDA
  - `include/dudux/nn/attention1b.hpp` — Attention binaire 0/1
  - `include/dudux/nn/mha1b.hpp` — Multi‑Head Attention 0/1 (K/V partagés)
  - `src/gpu/attention1b.cu` — Kernels CUDA AND+POPC et top‑k device
  - `src/benchmark.cpp` — Bench perf et fonctionnalités (router, attention, MHA)
  - `tests/` — Tests unitaires (bitops, attention, MHA, router, MLP, multistream)
- `docs/` Notes et guides (mémoire, configs, GPU)
- `.github/copilot-instructions.md` Contrat technique détaillé et principes de conception

## Prérequis
- Linux, CMake ≥ 3.12, compilateur C++17 (GCC/Clang)
- CUDA (optionnel) pour le backend GPU

## Compilation
Exemples (adapter selon votre environnement):

```bash
# Build CPU only
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Build avec CUDA
cmake -S cpp -B build -DDUDUX_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=75   # override recommandé
cmake --build build -j
```

Notes:
- `CMAKE_CUDA_ARCHITECTURES` doit cibler votre GPU (ex: 61=Pascal, 75=Turing, 86=Ampere).
- Flags utiles: `-DDUDUX_ENABLE_NATIVE=ON` (par défaut), `-DDUDUX_ENABLE_METRICS=ON`.

## Lancer le benchmark
L’exécutable `dudux_perf` accepte des paramètres positionnels:

```
NBIT NITEMS Q M KLIST VBIT METRICS QBATCH HEADS KH_LIST CAND MULTISTREAM
```

- `NBIT` taille (bits) des clés (et requêtes)
- `NITEMS` nombre d’items
- `Q` requêtes
- `M` mémoires pour le routeur
- `KLIST` liste K (ex: "8,16")
- `VBIT` taille (bits) des valeurs
- `METRICS` 0/1 active les métriques runtime (sinon via env `DUDUX_METRICS`)
- `QBATCH` facteur de répétition
- `HEADS` nombre de têtes MHA
- `KH_LIST` K par tête (optionnel, ex: "2,4,8,8")
- `CAND` candidats (0=ALL, sinon restreint)
- `MULTISTREAM` 0/1 (CUDA) active exécution 1 flux/tête

Exemple (CUDA + multistream + CAND):

```bash
./build/dudux_perf 16384 2000 100 4 8,16 1024 0 1 8 0 256 1
```

Extrait de sortie (variable selon machine):

```
CAND (candidates per query per head): 256
MHA multistream: ON
Router M=4, K=8: ... popcount_calls: 0 ...
Attention VBIT=1024, K=8: ... popcount_calls: 0 ...
MHA H=8, VBIT=1024, K=8: ...
MHA_multistream H=8, VBIT=1024, K=8: ...
```

Astuce VRAM ≤4 GB:
- Préférer `NBIT` 8–32k, `VBIT` 512–2048, `K` 4–16, `HEADS` 4–8
- Activer `CAND` (128–512) et `MULTISTREAM=1`
- Limiter copies H2D/D2H; réutiliser buffers

## Tests

```bash
ctest --test-dir build -V
```

Tests inclus: bit‑ops, attention, MHA (concat), router, MLP, et test d’équivalence multistream (si CUDA).

## API (aperçu)
- `NanoAttention1b`:
  - `topk_into(q, k, out)` / `topk_into_candidates(...)` (+ versions stream CUDA)
  - `attend(...)`, `attend_weighted(...)`, variantes masquées
- `NanoMultiHeadAttention1b`:
  - `attend(...)`, `attend_candidates(...)`
  - `attend_candidates_stream(..., stream)` et `attend_candidates_multistream(...)`

Les clés/valeurs sont des `BitVector` packés (uint64). Les chemins GPU exposent un `stream_handle` optionnel.

## Conception et contraintes
- 1‑bit strict, aucune multiplication flottante en inference (sauf α/τ scalaires)
- STE pour l’apprentissage (poids « shadow » float32, binarisation en avant)
- Reproductibilité: tie‑break stable (score décroissant, index croissant)
- Pas d’allocations dans le chemin critique (buffers persistants)

Voir `.github/copilot-instructions.md` pour le contrat détaillé (algèbre binaire, STE, métriques, GPU/streams).

## Roadmap
- Pipeline d’entraînement minimal (STE), export poids packés
- Tri/selection top‑k device plus efficient (partial‑selection CUB/Thrust)
- Indices compacts (uint16 quand N≤65k)
- Variantes GPU masquées/pondérées end‑to‑end

## Contribution
Issues et PR bienvenues. Merci de garder le code simple, mesuré, et testé (CTests). Pensez à documenter les choix (tie‑break, layout, limites VRAM) et à ajouter des tests.
