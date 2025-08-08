#include <cassert>
#include <vector>
#include <iostream>
#include "dudux/nn/mha1b.hpp"
#include "dudux/core/bitvector.hpp"

using dudux::nn::NanoMultiHeadAttention1b;
using dudux::core::BitVector;

int main(){
#ifdef DUDUX_ENABLE_CUDA
    const size_t H=4, KB=256, VB=64, N=128;
    NanoMultiHeadAttention1b mha(H, KB, VB);
    // Construire un petit corpus pseudo-déterministe
    for(size_t i=0;i<N;++i){
        BitVector k(KB), v(VB);
        for(size_t b=0;b<KB;b+=7) if(((i+b)%3)==0) k.set(b,true);
        for(size_t b=0;b<VB;b+=5) if(((i+b)%2)==0) v.set(b,true);
        mha.add(k,v);
    }
    // Requêtes par tête identiques (stress du tie-break)
    std::vector<BitVector> q_heads(H, BitVector(KB));
    for(size_t b=0;b<KB;b+=11) q_heads[0].set(b,true);
    for(size_t h=1; h<H; ++h) q_heads[h] = q_heads[0];

    // Générer un pool de candidats simple [0..CAND-1]
    std::vector<size_t> candidates; const size_t CAND = 64;
    for(size_t i=0;i<CAND;++i) candidates.push_back(i);

    const size_t K=8; const uint32_t tau=(K+1)/2;
    // Référence: exécution séquentielle stream unique
    std::vector<BitVector> out_ref;
    mha.attend_candidates(q_heads, K, tau, candidates, out_ref);

    // Multi-stream: doit être bit-identique
    std::vector<BitVector> out_ms;
    mha.attend_candidates_multistream(q_heads, K, tau, candidates, out_ms);

    assert(out_ref.size()==out_ms.size());
    for(size_t h=0; h<H; ++h){
        assert(out_ref[h].size()==out_ms[h].size());
        for(size_t b=0;b<out_ref[h].size();++b){
            if(out_ref[h].get(b)!=out_ms[h].get(b)){
                std::cerr << "Mismatch at head="<<h<<" bit="<<b<<"\n";
                return 1;
            }
        }
    }
    std::cout << "OK\n";
#else
    std::cout << "SKIP (CUDA disabled)\n";
#endif
    return 0;
}
