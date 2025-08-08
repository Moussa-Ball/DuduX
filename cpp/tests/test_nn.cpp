#include <cassert>
#include <vector>
#include "dudux/core/bitvector.hpp"
#include "dudux/nn/linear1b.hpp"
#include "dudux/nn/act1b.hpp"

int main() {
    using dudux::core::BitVector;
    using dudux::nn::NanoLinear1b;
    using dudux::nn::NanoAct1b;

    const size_t in = 10, out = 3;
    NanoLinear1b lin(in, out);

    // Input x: bits at even indices
    BitVector x(in);
    for (size_t i = 0; i < in; i += 2) x.set(i);

    // Rows: r0 = x (popcount = ceil(in/2)), r1 = empty (0), r2 = ones at odd indices (no overlap)
    BitVector r0(in); for (size_t i = 0; i < in; i += 2) r0.set(i);
    BitVector r1(in);
    BitVector r2(in); for (size_t i = 1; i < in; i += 2) r2.set(i);
    lin.set_weight_row(0, r0);
    lin.set_weight_row(1, r1);
    lin.set_weight_row(2, r2);

    std::vector<uint32_t> s; lin.matvec_popcnt(x, s);
    assert(s.size() == out);
    assert(s[0] == (in + 1) / 2);
    assert(s[1] == 0u);
    assert(s[2] == 0u);

    // Activation: threshold = 1 -> only row0 passes
    NanoAct1b act(1);
    BitVector y; act.forward(s, y);
    assert(y.size() == out);
    assert(y.get(0) == true);
    assert(y.get(1) == false);
    assert(y.get(2) == false);

    return 0;
}
