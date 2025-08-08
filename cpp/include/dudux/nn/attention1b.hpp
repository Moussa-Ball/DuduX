#pragma once
/**
 * @file attention1b.hpp
 * @brief Attention binaire: similarité = popcount(q & k_i), top-k streaming, agrégation V par majorité.
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <queue>
#include <utility>
#include <stdexcept>
#include <algorithm>

#include "dudux/core/bitvector.hpp"
#include "dudux/core/bitops.hpp"

namespace dudux { namespace nn {

class NanoAttention1b {
public:
    NanoAttention1b(size_t key_bits, size_t value_bits)
        : kbits_(key_bits), vbits_(value_bits) {}

    void add(const dudux::core::BitVector& key, const dudux::core::BitVector& value) {
        if (key.size() != kbits_ || value.size() != vbits_) throw std::invalid_argument("NanoAttention1b: size mismatch");
        keys_.push_back(key);
        values_.push_back(value);
    }

    size_t size() const noexcept { return keys_.size(); }

    // Calcule s_i = popcount(q & k_i) et renvoie top-k (index, score) triés (score décroissant, tie-break index croissant)
    std::vector<std::pair<size_t, uint32_t>> topk(const dudux::core::BitVector& q, size_t k) const {
        std::vector<std::pair<size_t, uint32_t>> result;
        topk_into(q, k, result);
        return result;
    }

    // Variante sans allocations supplémentaires côté API (remplit 'out')
    void topk_into(const dudux::core::BitVector& q, size_t k, std::vector<std::pair<size_t, uint32_t>>& out) const {
        if (q.size() != kbits_) throw std::invalid_argument("NanoAttention1b: query size mismatch");
        out.clear();
        if (k == 0 || keys_.empty()) return;
        struct Entry { uint32_t s; size_t idx; };
        struct MinCmp {
            bool operator()(const Entry& a, const Entry& b) const {
                if (a.s != b.s) return a.s > b.s; // min-heap sur score
                return a.idx < b.idx;             // pour scores égaux: idx plus grand est plus "petit" (pire)
            }
        };
        std::priority_queue<Entry, std::vector<Entry>, MinCmp> heap;
        const size_t words = (kbits_ + 63) / 64;
        const uint64_t* qw = q.data();
        for (size_t i = 0; i < keys_.size(); ++i) {
            const uint64_t* kw = keys_[i].data();
            uint32_t s = 0;
            for (size_t w = 0; w < words; ++w) s += dudux::core::popcount_u64(qw[w] & kw[w]);
            Entry e{s, i};
            if (heap.size() < k) {
                heap.push(e);
            } else {
                const Entry& worst = heap.top();
                if (e.s > worst.s || (e.s == worst.s && e.idx < worst.idx)) {
                    heap.pop();
                    heap.push(e);
                }
            }
        }
        out.resize(heap.size());
        for (size_t i = out.size(); i-- > 0;) { auto e = heap.top(); heap.pop(); out[i] = {e.idx, e.s}; }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){ if (a.second != b.second) return a.second > b.second; return a.first < b.first; });
    }

    // Attention: retourne une valeur agrégée par majorité sur les top-k (τ: seuil en votes pour bit=1)
    void attend(const dudux::core::BitVector& q, size_t k, uint32_t tau_votes, dudux::core::BitVector& out_value) const {
        std::vector<std::pair<size_t, uint32_t>> tk;
        topk_into(q, k, tk);
        attend_with_topk(tk, tau_votes, out_value);
    }

    // Variante qui réutilise un top-k déjà calculé
    void attend_with_topk(const std::vector<std::pair<size_t, uint32_t>>& tk, uint32_t tau_votes, dudux::core::BitVector& out_value) const {
        out_value.resize(vbits_);
        if (tk.empty()) { out_value.clear(); return; }
        for (size_t i = 0; i < vbits_; ++i) {
            uint32_t c1 = 0;
            for (auto& pr : tk) { if (values_[pr.first].get(i)) ++c1; }
            out_value.set(i, c1 >= tau_votes);
        }
    }

    // Variante zéro alloc: l'appelant fournit un buffer scratch pour le top-k
    void attend_into(const dudux::core::BitVector& q, size_t k, uint32_t tau_votes,
                     dudux::core::BitVector& out_value,
                     std::vector<std::pair<size_t, uint32_t>>& scratch_topk) const {
        scratch_topk.clear();
        topk_into(q, k, scratch_topk);
        attend_with_topk(scratch_topk, tau_votes, out_value);
    }

    // Majorité pondérée par score (poids = popcount overlap)
    void attend_weighted(const dudux::core::BitVector& q, size_t k, uint32_t tau_weight, dudux::core::BitVector& out_value) const {
        std::vector<std::pair<size_t, uint32_t>> tk;
        topk_into(q, k, tk);
        attend_weighted_with_topk(tk, tau_weight, out_value);
    }

    void attend_weighted_with_topk(const std::vector<std::pair<size_t, uint32_t>>& tk, uint32_t tau_weight, dudux::core::BitVector& out_value) const {
        out_value.resize(vbits_);
        if (tk.empty()) { out_value.clear(); return; }
        for (size_t i = 0; i < vbits_; ++i) {
            uint32_t w1 = 0;
            for (auto& pr : tk) { if (values_[pr.first].get(i)) w1 += pr.second; }
            out_value.set(i, w1 >= tau_weight);
        }
    }

private:
    size_t kbits_;
    size_t vbits_;
    std::vector<dudux::core::BitVector> keys_;
    std::vector<dudux::core::BitVector> values_;
};

}} // namespace dudux::nn
