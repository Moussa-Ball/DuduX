# BSO — Binary Spiking Online Optimization Algorithm (ICML 2025)

Lien: https://icml.cc/virtual/2025/poster/45087

Auteurs: Yu Liang · Yu Yang · Wenjie Wei · Ammar Belatreche · Shuai Wang · Malu Zhang · Yang Yang

Note: l’accès à l’abstract/vidéo requiert un login; cette fiche liste ce que l’on peut déduire du titre et prépare l’intégration potentielle dans Dudux.

## Ce que suggère le titre
- « Binary Spiking »: signaux/activations discrets (spikes 0/1) — compatible avec Dudux (0/1 strict).
- « Online Optimization »: mise à jour incrémentale par échantillon/temps (streaming), localement stable.
- Cible probable: apprentissage SNN/évènementiel; applicable aux MLP/attention 0/1 si la règle d’update est locale et binaire-compatible.

## Questions à clarifier (lecture papier)
- Domaine binaire utilisé: {0,1} (AND+popcount) ou {−1,+1} (XNOR)?
- Surrogates: quel STE/surrogate gradient? bornage/clip? bande de sensibilité autour du seuil?
- Règle « online »: purement locale (pré/post-synaptique + erreur) ou nécessite un signal global? nécessite des accumulations temporelles (membrane/refractory)?
- Stabilité: contraintes sur le LR, normalisations (τ/α), régularisations (balance 0/1, corrélation)?
- Résultats: tâches évaluées, gains vs. STE classique/BOP, coûts mémoire/compute réels.

## Intérêt pour Dudux
- Alignement fort: Dudux est 0/1 strict en inference, STE en entraînement, bit‑packing + popcount.
- Si BSO propose une règle d’update locale/online mieux conditionnée, on peut remplacer/compléter notre STE (perceptron/MLP) par un optimiseur « BSO-like ».
- Cas d’usage: apprentissage en flux (routing/attention candidates), mise à jour rapide de petits adaptateurs binaires sous budget VRAM.

## Plan d’intégration (proposé)
1) Interface optimiseur online
   - `include/dudux/opt/online.hpp`: `struct OnlineUpdateCtx { const BitVector& x; const BitVector* h; uint8_t y; uint8_t yhat; uint32_t tau; float lr; }`
   - `class NanoOptimOnline { virtual void update_linear_row(std::vector<float>& w_real_row, const OnlineUpdateCtx& ctx)=0; }`
2) Implémentation BSO (après lecture papier)
   - `NanoOptimBSO` avec paramètres (clip, fenêtre, éventuelle trace pré/post) adaptés à {0,1}.
3) Wiring rapide
   - `train_ste.cpp` et `train_mlp_ste.cpp`: option `--optim bso|ste` pour commuter la règle.
4) Métriques et sanity checks
   - Stabilité LR vs. acc; balance 0/1 par couche; BitOPs invariants; reproductibilité (seed).
5) Bench/Docs
   - Mini benchmark « online » (dataset jouet), doc d’usage et limites.

## Tâches
- [ ] Lire le papier/slide ICML et extraire l’algorithme exact.
- [ ] Décider du mapping {−1,+1} → {0,1} si nécessaire (documenter l’algèbre).
- [ ] Implémenter `NanoOptimBSO` + tests unitaires (updates locales vs. référence).
- [ ] Intégrer dans `dudux_train_*` avec flags et logs (τ/α, balance, acc).
- [ ] Exemple reproductible + README.

---

Suivi: dès réception du PDF/lien PMLR, compléter cette fiche (abstract, pseudo‑code, hypothèses) et réaliser un POC dans `train_mlp_ste`.
