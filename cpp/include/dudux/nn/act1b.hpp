#pragma once
/**
 * @file act1b.hpp
 * @brief Activation binaire avec seuil τ: out[i] = 1 si s[i] >= τ, sinon 0.
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace dudux { namespace nn {

class NanoAct1b {
public:
    explicit NanoAct1b(uint32_t threshold = 0) : tau_(threshold) {}
    void set_threshold(uint32_t t) noexcept { tau_ = t; }
    uint32_t threshold() const noexcept { return tau_; }

    // Binarise des entiers (scores popcount) vers {0,1} BitVector en sortie.
    template<class Scores>
    void forward(const Scores& s, dudux::core::BitVector& out) const {
        out.resize(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] >= tau_) out.set(i, true); else out.set(i, false);
        }
    }
private:
    uint32_t tau_;
};

}} // namespace dudux::nn
