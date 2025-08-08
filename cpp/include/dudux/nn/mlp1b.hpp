#pragma once
/**
 * @file mlp1b.hpp
 * @brief MLP binaire: Linear1b -> Act1b -> Linear1b (entier-only en inference).
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include "dudux/core/bitvector.hpp"
#include "dudux/nn/linear1b.hpp"
#include "dudux/nn/act1b.hpp"

namespace dudux { namespace nn {

class NanoMLP1b {
public:
    NanoMLP1b(size_t in_features, size_t hidden, size_t out_features, uint32_t hidden_threshold = 1)
        : lin1_(in_features, hidden), act_(hidden_threshold), lin2_(hidden, out_features),
          hidden_bits_(hidden) {}

    size_t in_features() const noexcept { return lin1_.in_features(); }
    size_t hidden_features() const noexcept { return lin1_.out_features(); }
    size_t out_features() const noexcept { return lin2_.out_features(); }

    // Accès aux couches pour setter les poids
    dudux::nn::NanoLinear1b& lin1() noexcept { return lin1_; }
    dudux::nn::NanoLinear1b& lin2() noexcept { return lin2_; }
    const dudux::nn::NanoLinear1b& lin1() const noexcept { return lin1_; }
    const dudux::nn::NanoLinear1b& lin2() const noexcept { return lin2_; }
    dudux::nn::NanoAct1b& act() noexcept { return act_; }
    const dudux::nn::NanoAct1b& act() const noexcept { return act_; }

    // Forward: renvoie uniquement les scores entiers de la couche de sortie.
    void forward_scores(const dudux::core::BitVector& x, std::vector<uint32_t>& out_scores) const {
        // 1) Linear1b -> scores hidden
        std::vector<uint32_t> hid_scores;
        lin1_.matvec_popcnt(x, hid_scores);
        // 2) Act1b -> hidden bits
        act_.forward(hid_scores, hidden_bits_);
        // 3) Linear1b (2ème) -> out scores
        lin2_.matvec_popcnt(hidden_bits_, out_scores);
    }

    // Forward binaire: applique un seuil de sortie et écrit des bits
    void forward_binary(const dudux::core::BitVector& x, uint32_t out_threshold, dudux::core::BitVector& out_bits) const {
        std::vector<uint32_t> out_scores;
        forward_scores(x, out_scores);
        dudux::nn::NanoAct1b out_act(out_threshold);
        out_act.forward(out_scores, out_bits);
    }

private:
    dudux::nn::NanoLinear1b lin1_;
    dudux::nn::NanoAct1b    act_;
    dudux::nn::NanoLinear1b lin2_;
    // Buffer mutable pour éviter des allocations en chaîne (réutilisé)
    mutable dudux::core::BitVector hidden_bits_;
};

}} // namespace dudux::nn
