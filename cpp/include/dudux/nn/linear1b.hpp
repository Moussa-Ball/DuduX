#pragma once
/**
 * @file linear1b.hpp
 * @brief Couche linéaire binaire {0,1} -> accumulateurs entiers via AND+popcount.
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include "dudux/core/bitvector.hpp"
#include "dudux/core/bitops.hpp"

namespace dudux { namespace nn {

using dudux::core::BitVector;
using dudux::core::popcount_u64;

class NanoLinear1b {
public:
    NanoLinear1b(size_t in_features, size_t out_features)
        : in_(in_features), out_(out_features), words_((in_features + 63) / 64),
          W_(out_features * words_, 0ull) {}

    size_t in_features() const noexcept { return in_; }
    size_t out_features() const noexcept { return out_; }
    size_t words() const noexcept { return words_; }

    // Définit une ligne de poids depuis un BitVector binaire {0,1}
    void set_weight_row(size_t o, const BitVector& wrow) {
        if (o >= out_) throw std::out_of_range("NanoLinear1b: row index");
        if (wrow.size() != in_) throw std::invalid_argument("NanoLinear1b: wrow size mismatch");
        const uint64_t* src = wrow.data();
        uint64_t* dst = &W_[o * words_];
        for (size_t i = 0; i < words_; ++i) dst[i] = src[i];
        // Le BitVector source est déjà masqué en queue.
    }

    // Produit matrice-vecteur: renvoie les sommes entières (popcounts) par sortie.
    void matvec_popcnt(const BitVector& x, std::vector<uint32_t>& out_scores) const {
        if (x.size() != in_) throw std::invalid_argument("NanoLinear1b: input size mismatch");
        out_scores.assign(out_, 0u);
        const uint64_t* xv = x.data();
        for (size_t o = 0; o < out_; ++o) {
            const uint64_t* wr = &W_[o * words_];
            uint32_t acc = 0;
            for (size_t i = 0; i < words_; ++i) acc += popcount_u64(xv[i] & wr[i]);
            out_scores[o] = acc;
        }
    }

private:
    size_t in_;
    size_t out_;
    size_t words_;
    std::vector<uint64_t> W_; // layout: rows contiguës [out][words]
};

}} // namespace dudux::nn
