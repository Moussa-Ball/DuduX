#include <cassert>
#include <vector>
#include "dudux/nn/mha1b.hpp"
#include "dudux/core/bitvector.hpp"

using dudux::nn::NanoMultiHeadAttention1b;
using dudux::core::BitVector;

int main(){
    const size_t H=2, KB=8, VB=4;
    NanoMultiHeadAttention1b mha(H, KB, VB);
    // 3 items simples
    BitVector k0(KB), k1(KB), k2(KB);
    BitVector v0(VB), v1(VB), v2(VB);
    // k0=11110000, v0=0000
    for(int i=0;i<4;++i) k0.set(i,true);
    // k1=11001100, v1=1111
    v1.set(0,true); v1.set(1,true); v1.set(2,true); v1.set(3,true);
    k1.set(0,true); k1.set(1,true); k1.set(4,true); k1.set(5,true);
    // k2=00001111, v2=0101
    for(int i=4;i<8;++i) k2.set(i,true);
    v2.set(0,true); v2.set(2,true);
    mha.add(k0,v0); mha.add(k1,v1); mha.add(k2,v2);

    // Deux têtes avec requêtes différentes
    BitVector qh0(KB), qh1(KB);
    for(int i=0;i<4;++i) qh0.set(i,true); // match k0 fort
    for(int i=4;i<8;++i) qh1.set(i,true); // match k2 fort

    std::vector<BitVector> q_heads = {qh0, qh1};
    std::vector<BitVector> out_heads;
    mha.attend(q_heads, /*k=*/2, /*tau_votes=*/1, out_heads);
    assert(out_heads.size()==H);
    // Tête 0 -> proche k0 => v0=0000
    for(size_t i=0;i<VB;++i) assert(out_heads[0].get(i)==false);
    // Tête 1 -> proche k2 => v2=0101
    assert(out_heads[1].get(0)==true);
    assert(out_heads[1].get(1)==false);
    assert(out_heads[1].get(2)==true);
    assert(out_heads[1].get(3)==false);

    // Concat
    BitVector out_concat(H*VB);
    mha.attend_concat(q_heads, 2, 1, out_concat);
    for(size_t i=0;i<VB;++i) assert(out_concat.get(i)==false);
    assert(out_concat.get(VB+0)==true);
    assert(out_concat.get(VB+1)==false);
    assert(out_concat.get(VB+2)==true);
    assert(out_concat.get(VB+3)==false);

    return 0;
}
