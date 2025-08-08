#include <cassert>
#include <iostream>
#include "dudux/core/bitvector.hpp"
#include "dudux/core/bitops.hpp"
#include "dudux/memory/associative_memory.hpp"

int main() {
    using namespace dudux::core;
    using dudux::memory::AssociativeMemory;

    // popcount basic
    assert(popcount_u64(0ull) == 0);
    assert(popcount_u64(1ull) == 1);
    assert(popcount_u64(0xFFFFFFFFFFFFFFFFull) == 64);

    // BitVector set/get and popcount
    BitVector v(130);
    v.set(0); v.set(64); v.set(129);
    assert(v.get(0) && v.get(64) && v.get(129));
    assert(v.popcount() == 3);

    // Hamming
    BitVector a(10), b(10);
    for (int i = 0; i < 10; i += 2) a.set(i);
    for (int i = 1; i < 10; i += 2) b.set(i);
    assert(a.hamming(b) == 10);

    // Bitwise ops
    BitVector c = bit_or(a, b);
    assert(c.popcount() == 10);
    BitVector d = bit_and(a, b);
    assert(d.popcount() == 0);

    // === AssociativeMemory: top-k, tie-break, and into-variant ===
    {
        const size_t nbits = 8;
        BitVector q(nbits);
        q.set(0); q.set(2); q.set(4); // pattern 10101000

        AssociativeMemory mem(nbits);
        // helper to make vectors by copying q and toggling bits
        auto make = [&](std::initializer_list<int> ones){
            BitVector x(nbits);
            for (int i : ones) x.set(static_cast<size_t>(i));
            return x;
        };

        // Distances to q:
        // "a": identical -> 0
        mem.add(make({0,2,4}), "a");
        // "b": distance 1 (flip one)
        mem.add(make({0,2,4,6}), "b");
        // "c": also distance 1 (tie with b); label tie-break expects b before c
        mem.add(make({0,2}), "c");
        // "d": distance 3
        mem.add(make({1,3,5}), "d");

        // k=2 -> expect [("a",0), ("b",1)] (tie-break by label between b/c)
        auto r2 = mem.query_topk(q, 2);
        assert(r2.size() == 2);
        assert(r2[0].first == std::string("a") && r2[0].second == 0);
        assert(r2[1].first == std::string("b") && r2[1].second == 1);

        // k>=N -> all sorted by (distance,label)
        auto rall = mem.query_topk(q, 10);
        assert(rall.size() == 4);
        assert(rall[0].first == std::string("a") && rall[0].second == 0);
        assert(rall[1].first == std::string("b") && rall[1].second == 1);
        assert(rall[2].first == std::string("c") && rall[2].second == 1);
        // last is d with distance 3
        assert(rall[3].first == std::string("d") && rall[3].second == 3);

        // k=0 -> empty
        auto r0 = mem.query_topk(q, 0);
        assert(r0.empty());

        // into variant: pre-fill and reserve
        std::vector<std::pair<std::string, uint32_t>> out;
        out.reserve(2);
        out.emplace_back("zz", 999); // should be cleared by into
        mem.query_topk_into(q, 2, out);
        assert(out.size() == 2);
        assert(out[0].first == std::string("a") && out[0].second == 0);
        assert(out[1].first == std::string("b") && out[1].second == 1);
    }

    std::cout << "All tests passed.\n";
    return 0;
}
