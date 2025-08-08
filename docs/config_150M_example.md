# Config exemple: Nano‑LLM 150M (≤4GB)

- Params: 150M (poids ~18.75MB)
- Couches: 16, Dim: 512, Heads: 8, VBIT: 2048, K per‑head: 8
- Séquence: 1024, Batch micro: 2, Recompute activations: ON
- Optim: STE int8 (1B/param), α/τ par canal

Estimation (tools/estimate_memory.py):
- Weights ≈ 18.75MB
- Alpha/Tau ≈ ~131KB
- Optim (ste8) ≈ 150MB
- Activations (recompute) ≈ ~1.00GB
- Total ≈ ~1.17GB → marge pour buffers/IO < 4GB

Next: augmenter seq à 2048 ou batch 4 si marge disponible.
