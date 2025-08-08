# Dudux C++ (prototype binaire 0/1)

Objectif: démontrer une ossature C++ pure inspirée du cerveau où les neurones sont binaires (0/1).

- Représentation: BitVector packé en mots 64-bit
- Algèbre: AND/XOR + popcount, distance de Hamming
- Mémoire associative binaire (clé->étiquette) et encodeur Bloom-like

## Construire

```zsh
# depuis la racine du dépôt
cmake -S cpp -B build
cmake --build build -j
```

## Exécuter

```zsh
./build/dudux "bonjour"
```

## Benchmark

```zsh
./build/dudux_perf

### Test perf (optionnel)
Activez le test CTest "perf_smoke" (peut être lent):

```zsh
cmake -S cpp -B build -DDUDUX_ENABLE_PERF_TEST=ON
ctest --test-dir build -R dudux_perf_smoke --output-on-failure
```
```

## Idées d’extensions
- GPU: kernels bitwise + popc, top-k accéléré
- Apprentissage: ajuster les bits via règles locales/STE, mémoires à grande échelle
- I/O: sérialisation des BitVector, index Hamming par blocs
