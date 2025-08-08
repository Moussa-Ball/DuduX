# Guide budget mémoire (≤ 4GB)

Notation: P = #params, C = #canaux, H = #heads, L = #layers, D = dim

Poids (inference)
- Binaire: P bits = P/8 bytes
- Alphas/taus: ≈ (C * 8..16) bytes

État entraînement
- STE naïf: w_real fp32 → 4P B
- STE compact: w_real int8 + scale → ~P B
- BOP: momentum int8 + rare flips → ~P B, poids restent 1‑bit

Activations
- Binarisées, caches évités (recompute) → ~batch * L * D/8 B

Règles rapides
- 200M params: poids ~25MB; STE int8 état ~200MB; activations (seq 1K, D 512, L 16, batch 4) ~4*16*512/8*1K ≈ 0.5GB (réduire batch/recompute).
- Objectif 4GB: viser P≤200M, activer recompute, optimiser état à ≤1B/param (int8), dataset streaming.
