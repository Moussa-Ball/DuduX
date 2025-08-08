# Config exemple: Nano‑LLM 1B (3–4GB)

Objectif: entraîner un 1B binaire en ≤ 4GB (cible ~3GB typique), avec forward 100% binaire.

Proposition
- Params: 1B (poids ~119MB en 1‑bit)
- Couches: 24, Dim: 1024, Heads: 16, VBIT: 2048, Top‑k: 16 par tête
- Séquence: 2048, Micro‑batch: 1–2, Recompute activations: ON
- Optim: STE int8 (1B/param), α/τ par canal

Estimation (tools/estimate_memory.py):
- Weights ≈ 119MB
- Alpha/Tau ≈ ~384KB
- Optim (ste8) ≈ ~954MB
- Activations (recompute, seq=2048, batch=1) ≈ ~3MB
- Top‑k indices (16 heads, k=16) ≈ ~48MB
- Total ≈ ~1.1GB → large marge < 4GB

Remarques
- Le budget <4GB tient grâce au binaire strict + recompute + état optim int8.
- Si on passe en BOP (bit‑flip) l’état optim tombe aussi ~1B/param (EMA int8), potentiellement encore plus bas si on compresse.
- On peut augmenter la séquence ou le micro‑batch tant qu’on reste sous ~3–4GB mesuré.

Étapes suivantes
- Squelette d’entraînement (STE8) + test MNIST-like texte synthétique.
- Export packé Dudux pour inférence ultra‑légère.
