# Guide GPU 4GB — Dudux binaire

Objectif: entraîner et inférer en binaire sur un GPU 4GB, sans surcharger le CPU.

Build (option CUDA)
- cmake -S cpp -B build -DDUDUX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
- cmake --build build -j

Conseils mémoire (4GB)
- Binaire strict + recompute activations ON
- STE int8 (1 B/param) ou BOP 4–8 bit
- Seq 1024–2048, micro‑batch 1–2
- Attention: top‑k 8–16, cand (router) 128–256

Runtime
- Dudux utilise les kernels GPU pour scorer (AND+POPC) et le top‑k (Thrust) quand la build CUDA est activée.
- Si OOM: baisser cand/top‑k, batch, ou seq.

Notes
- Les kernels fournis sont un squelette minimal. On pourra étendre aux opérations MHA complètes et ajouter des buffers réutilisés pour éviter les allocations.
