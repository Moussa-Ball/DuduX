#include <cassert>
#include <vector>
#include <cstdint>
#include "dudux/nn/attention1b.hpp"
#include "dudux/core/bitvector.hpp"

using dudux::nn::NanoAttention1b;
using dudux::core::BitVector;

int main(){
    // Prépare 4 entrées simples de 8 bits
    NanoAttention1b att(8, 8);
    BitVector k0(8), v0(8);
    BitVector k1(8), v1(8);
    BitVector k2(8), v2(8);
    BitVector k3(8), v3(8);

    // clés
    // k0 = 11110000, k1 = 11001100, k2 = 10101010, k3 = 00001111
    for(int i=0;i<4;++i) { k0.set(i, true); }
    for(int i=4;i<8;++i) { k3.set(i, true); }
    k1.set(0,true); k1.set(1,true); k1.set(4,true); k1.set(5,true);
    k2.set(0,true); k2.set(2,true); k2.set(4,true); k2.set(6,true);

    // valeurs distinctes
    // v0 = 00000000
    // v1 = 11111111
    for(int i=0;i<8;++i) v1.set(i,true);
    // v2 = 10101010
    for(int i=0;i<8;i+=2) v2.set(i,true);
    // v3 = 01010101
    for(int i=1;i<8;i+=2) v3.set(i,true);

    att.add(k0,v0); att.add(k1,v1); att.add(k2,v2); att.add(k3,v3);

    // Requête q = 11110000 (match fort avec k0, puis k1)
    BitVector q(8); for(int i=0;i<4;++i) q.set(i,true);

    auto tk = att.topk(q, 2);
    assert(tk.size()==2);
    // k0 overlap=4, k1 overlap=2, k2 overlap=2, k3 overlap=0 -> tie-break par index pour score=2 (k1 avant k2)
    assert(tk[0].first==0 && tk[0].second==4);
    assert(tk[1].first==1 && tk[1].second==2);

    BitVector out(8);
    // majorité stricte sur k=3 avec tau=2
    att.attend(q, 3, 2, out);
    // top3 indices attendus: k0(4), k1(2), k2(2) -> valeurs v0, v1, v2 -> bit i=0: (0,1,1)=2 >=2 -> 1 ; i=1: (0,1,0)=1 -> 0
    assert(out.get(0)==true);
    assert(out.get(1)==false);

    // k=0 -> out=0
    att.attend(q, 0, 1, out);
    for(int i=0;i<8;++i) assert(out.get(i)==false);

    // attend_into (zéro alloc via scratch)
    std::vector<std::pair<size_t,uint32_t>> scratch;
    att.attend_into(q, 3, 2, out, scratch);
    assert(out.get(0)==true && out.get(1)==false);

    // Version pondérée: tau_weight= somme des 2 meilleurs bits sur i=0
    // top3: k0(4), k1(2), k2(2). Sur bit0: v0=0, v1=1 (poids 2), v2=1 (poids 2) -> w1=4
    att.attend_weighted(q, 3, /*tau_weight=*/3, out);
    assert(out.get(0)==true); // 4 >= 3
    // bit1: v0=0, v1=1 (2), v2=0 -> w1=2; tau_weight=3 -> 0
    assert(out.get(1)==false);

    // Masquage: exclure k0 (mask[0]=0) et causal val_upto=1
    std::vector<uint8_t> mask = {0,1,1,1};
    att.attend_masked(q, 3, 1, &mask, 1, out);
    // Sans k0, top viennent de k1/k2 => out bit0 = 1 via v1/v2
    assert(out.get(0)==true);

    return 0;
}
