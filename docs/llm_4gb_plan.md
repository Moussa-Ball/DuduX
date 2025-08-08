# LLM binaire ≤ 4GB: plan concret

Objectif: un modèle type "Nano-LLM" entraînable sur une seule machine avec <4GB de RAM/VRAM pour l’optimisation.

Principes
- Représentation BIT1 (poids et activations binaires au forward), accumulateurs entiers, échelles (alpha) en float32 par canal.
- Entraînement par STE (shadow weights float32) ou BOP (bit‑flip). Par défaut: STE pour stabilité.
- Contrainte mémoire stricte: batch micro, optim état réduit, checkpoints gradient off, et dataset streaming.

Budget mémoire (indicatif)
- Poids binaire: ~1 bit/param. 
- Alphas/taus: ~8–16B par canal (minoritaire).
- État optim (STE): 4B/param (fp32) si naïf; on compresse à 8‑bit EMA ou on bascule BOP.

Cibles de taille
- 100–200M params binaires (≈12.5–25MB en 1‑bit) avec état optim compressé (≤2–3GB) ou BOP (≤0.5GB), séquence 1–2K tokens.
- Alternative micro: 30–60M params (quelques MB), pour entraînement plus confortable.

Architecture Nano
- Embedding: 1‑bit, dim 256–512. Positional encod. rotatoire discret.
- Blocs: 12–24 couches NanoMLP1b + NanoMHA1b, H=8–12, K top‑k 8–16, V bit‑width 1024–2048.
- Router/mémoire associative pour récupération locale.

Entraînement
- STE: w_real (fp32) → binarize(w_real, τ) au forward; backward via STE; clamp w_real∈[−a,a]. α/τ apprenables par canal. Optim: Adam8bit/Lion8bit.
- BOP: momentum EMA 8‑bit; flip bit si |m|>θ; rareté des flips.
- Mixed objective: LM + sparsité + balance 0/1.

Données
- Tokenizer binaire (Bloom encoder) + corpus texte compressé. Streaming par shards.

Plan d’itération
1) Pré‑entraîner NanoGPT‑mini (30–60M) STE 8‑bit, seq 512.
2) Élargir à 150–200M, seq 1K–2K, activer MHA_masked + top‑k par tête.
3) Distillation depuis un prof enseignant pour stabilité.

Export Dudux
- Quantize -> pack bits -> dump headers pour include/flash.

Prochaines étapes
- Squelette d’entraînement C++ minimal (STE + BOP) + tests.
- Script d’estimation mémoire/temps.
