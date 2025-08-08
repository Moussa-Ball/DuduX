#include <cassert>
#include <vector>
#include <string>
#include "dudux/core/bitvector.hpp"
#include "dudux/memory/associative_memory.hpp"
#include "dudux/memory/router.hpp"

int main() {
    using dudux::core::BitVector;
    using dudux::memory::AssociativeMemory;
    using dudux::memory::RouterTopK;

    const size_t nbits = 16;
    BitVector q(nbits);
    for (int i = 0; i < 16; i += 2) q.set(i); // 1010...

    // Two memories, overlapping labels with different distances to ensure routing works
    AssociativeMemory m1(nbits);
    AssociativeMemory m2(nbits);

    auto make = [&](std::initializer_list<int> ones){
        BitVector x(nbits);
        for (int i : ones) x.set(static_cast<size_t>(i));
        return x;
    };

    // m1 entries
    m1.add(make({0,2,4,6,8,10,12,14}), "a"); // exact match -> d=0
    m1.add(make({0,2,4,6,8,10,12}),     "b"); // d=1

    // m2 entries with same distance as b to test tie-break by label
    m2.add(make({0,2,4,6,8,10,12}),     "c"); // d=1 (tie with b), expect b before c
    m2.add(make({1,3,5,7,9,11,13,15}),  "z"); // opposite -> d=16

    std::vector<const AssociativeMemory*> mems = { &m1, &m2 };
    RouterTopK router(mems);

    // k=2 -> expect a (0), b (1)
    auto r2 = router.query_topk(q, 2);
    assert(r2.size() == 2);
    assert(r2[0].first == std::string("a") && r2[0].second == 0);
    assert(r2[1].first == std::string("b") && r2[1].second == 1);

    // k>=N -> all + sorted (a,b,c,z)
    auto rall = router.query_topk(q, 10);
    assert(rall.size() == 4);
    assert(rall[0].first == std::string("a") && rall[0].second == 0);
    assert(rall[1].first == std::string("b") && rall[1].second == 1);
    assert(rall[2].first == std::string("c") && rall[2].second == 1);
    assert(rall[3].first == std::string("z") && rall[3].second == 16);

    // k=0 -> empty
    auto r0 = router.query_topk(q, 0);
    assert(r0.empty());

    // into variant
    std::vector<std::pair<std::string, uint32_t>> out;
    out.reserve(2);
    router.query_topk_into(q, 2, out);
    assert(out.size() == 2);
    assert(out[0].first == std::string("a") && out[0].second == 0);
    assert(out[1].first == std::string("b") && out[1].second == 1);

    return 0;
}
