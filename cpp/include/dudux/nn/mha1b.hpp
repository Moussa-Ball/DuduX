#pragma once
/**
 * @file mha1b.hpp
 * @brief Multi-Head Attention binaire minimale: H têtes partageant les mêmes K/V.
 */

#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "dudux/nn/attention1b.hpp"
#include "dudux/core/bitvector.hpp"
#ifdef DUDUX_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dudux { namespace nn {

class NanoMultiHeadAttention1b {
public:
    NanoMultiHeadAttention1b(size_t heads, size_t key_bits, size_t value_bits)
        : H_(heads), att_(key_bits, value_bits) {
        if (H_ == 0) throw std::invalid_argument("MHA1b: heads must be > 0");
    }

#ifdef DUDUX_ENABLE_CUDA
    ~NanoMultiHeadAttention1b() {
        if (streams_ready_) {
            for (auto s : streams_) if (s) cudaStreamDestroy(s);
            streams_.clear();
            streams_ready_ = false;
            streams_H_cached_ = 0;
        }
    }
#endif

    size_t heads() const noexcept { return H_; }
    size_t key_bits() const noexcept { return att_.key_bits(); }
    size_t value_bits() const noexcept { return att_.value_bits(); }

    void add(const dudux::core::BitVector& key, const dudux::core::BitVector& value) {
        att_.add(key, value); // K/V partagés entre têtes
    }

    size_t size() const noexcept { return att_.size(); }

    // Sorties par tête (majorité non pondérée)
    void attend(const std::vector<dudux::core::BitVector>& q_heads, size_t k, uint32_t tau_votes,
                std::vector<dudux::core::BitVector>& out_heads) const {
        if (q_heads.size() != H_) throw std::invalid_argument("MHA1b: q_heads size != heads");
        out_heads.resize(H_);
        for (size_t h = 0; h < H_; ++h) {
            att_.attend(q_heads[h], k, tau_votes, out_heads[h]);
        }
    }

    // Version pondérée
    void attend_weighted(const std::vector<dudux::core::BitVector>& q_heads, size_t k, uint32_t tau_weight,
                         std::vector<dudux::core::BitVector>& out_heads) const {
        if (q_heads.size() != H_) throw std::invalid_argument("MHA1b: q_heads size != heads");
        out_heads.resize(H_);
        for (size_t h = 0; h < H_; ++h) {
            att_.attend_weighted(q_heads[h], k, tau_weight, out_heads[h]);
        }
    }

    // Variante: MHA avec sous-ensemble de candidats partagé entre têtes
    void attend_candidates(const std::vector<dudux::core::BitVector>& q_heads, size_t k, uint32_t tau_votes,
                           const std::vector<size_t>& candidates,
                           std::vector<dudux::core::BitVector>& out_heads) const {
        if (q_heads.size() != H_) throw std::invalid_argument("MHA1b: q_heads size != heads");
        out_heads.resize(H_);
        std::vector<std::pair<size_t,uint32_t>> tk;
        for (size_t h = 0; h < H_; ++h) {
            tk.clear();
            att_.topk_into_candidates(q_heads[h], k, candidates, tk);
            att_.attend_with_topk(tk, tau_votes, out_heads[h]);
        }
    }

#ifdef DUDUX_ENABLE_CUDA
    // Variante stream: permet de passer un flux CUDA (p.ex. un flux par tête)
    void attend_candidates_stream(const std::vector<dudux::core::BitVector>& q_heads, size_t k, uint32_t tau_votes,
                                  const std::vector<size_t>& candidates,
                                  std::vector<dudux::core::BitVector>& out_heads,
                                  void* stream_handle) const {
        if (q_heads.size() != H_) throw std::invalid_argument("MHA1b: q_heads size != heads");
        out_heads.resize(H_);
        std::vector<std::pair<size_t,uint32_t>> tk;
        for (size_t h = 0; h < H_; ++h) {
            tk.clear();
            att_.topk_into_candidates_stream(q_heads[h], k, candidates, tk, stream_handle);
            att_.attend_with_topk(tk, tau_votes, out_heads[h]);
        }
    }

    // Exécution multi-stream: un flux CUDA par tête, appels parallèles et synchronisation finale
    void attend_candidates_multistream(const std::vector<dudux::core::BitVector>& q_heads, size_t k, uint32_t tau_votes,
                                       const std::vector<size_t>& candidates,
                                       std::vector<dudux::core::BitVector>& out_heads) const {
        if (q_heads.size() != H_) throw std::invalid_argument("MHA1b: q_heads size != heads");
        out_heads.resize(H_);
        const size_t H = H_;
        ensure_streams_();
        // Lancer top-k candidats par tête en parallèle
        std::vector<std::vector<std::pair<size_t,uint32_t>>> tk_heads(H);
        for (size_t h=0; h<H; ++h) {
            tk_heads[h].clear();
            att_.topk_into_candidates_stream(q_heads[h], k, candidates, tk_heads[h], (void*)streams_[h]);
        }
        // Synchroniser flux et agréger
        for (size_t h=0; h<H; ++h) cudaStreamSynchronize(streams_[h]);
        for (size_t h=0; h<H; ++h) {
            att_.attend_with_topk(tk_heads[h], tau_votes, out_heads[h]);
        }
    }
#endif

    // Concatène les sorties des H têtes dans un BitVector de taille H*value_bits
    void attend_concat(const std::vector<dudux::core::BitVector>& q_heads, size_t k, uint32_t tau_votes,
                       dudux::core::BitVector& out_concat) const {
        if (q_heads.size() != H_) throw std::invalid_argument("MHA1b: q_heads size != heads");
        out_concat.resize(H_ * value_bits());
        dudux::core::BitVector tmp(value_bits());
        for (size_t h = 0; h < H_; ++h) {
            att_.attend(q_heads[h], k, tau_votes, tmp);
            for (size_t i = 0; i < value_bits(); ++i) {
                out_concat.set(h * value_bits() + i, tmp.get(i));
            }
        }
    }

private:
    size_t H_;
    NanoAttention1b att_; // K/V partagés
#ifdef DUDUX_ENABLE_CUDA
    // Pool de streams CUDA persistants (créés à la demande)
    mutable std::vector<cudaStream_t> streams_{};
    mutable bool streams_ready_ = false;
    mutable size_t streams_H_cached_ = 0;
    void ensure_streams_() const {
        if (!streams_ready_ || streams_H_cached_ != H_) {
            // Détruire existants si la taille change
            if (!streams_.empty()) {
                for (auto s : streams_) if (s) cudaStreamDestroy(s);
                streams_.clear();
            }
            streams_.resize(H_);
            for (size_t h=0; h<H_; ++h) cudaStreamCreateWithFlags(&streams_[h], cudaStreamNonBlocking);
            streams_ready_ = true;
            streams_H_cached_ = H_;
        }
    }
#endif
};

}} // namespace dudux::nn
