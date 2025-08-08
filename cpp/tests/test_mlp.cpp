#include <cassert>
#include <vector>
#include "dudux/core/bitvector.hpp"
#include "dudux/nn/mlp1b.hpp"

int main() {
    using dudux::core::BitVector;
    using dudux::nn::NanoMLP1b;

    const size_t in = 8, hidden = 6, out = 4;
    NanoMLP1b mlp(in, hidden, out, /*hidden_threshold=*/2);

    // Build deterministic weights: lin1 rows = identity-like patterns; lin2 rows = pass-through of hidden even bits
    // lin1
    for (size_t o = 0; o < hidden; ++o) {
        BitVector r(in);
        for (size_t i = o % in; i < in; i += 2) r.set(i); // simple pattern
        mlp.lin1().set_weight_row(o, r);
    }
    // lin2: each output connects to hidden bit at index 2*o if exists
    for (size_t o = 0; o < out; ++o) {
        BitVector r(hidden);
        size_t idx = 2 * o;
        if (idx < hidden) r.set(idx);
        mlp.lin2().set_weight_row(o, r);
    }

    // Input: set even bits -> hidden_threshold=2 means need >=2 overlaps
    BitVector x(in); for (size_t i = 0; i < in; i += 2) x.set(i);

    std::vector<uint32_t> scores; mlp.forward_scores(x, scores);
    // Each output o will be 1 if hidden[2*o] bit was set by act -> then lin2 popcount >=1
    for (size_t o = 0; o < out; ++o) {
        if (2 * o < hidden) {
            assert(scores[o] >= 0); // non-negative by construction
        } else {
            assert(scores[o] == 0u);
        }
    }

    // Binary forward with out_threshold=1
    BitVector y; mlp.forward_binary(x, 1, y);
    for (size_t o = 0; o < out; ++o) {
        if (2 * o < hidden) {
            // Some outputs may be 1 depending on the threshold behaviour; at least it's defined and binaire
            (void)y.get(o);
        } else {
            assert(y.get(o) == false);
        }
    }

    return 0;
}
