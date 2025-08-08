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
#include <cstring>
#ifdef DUDUX_ENABLE_CUDA
#include <cuda_runtime.h>
extern "C" void dudux_topk_scores(unsigned int* d_scores, int* d_indices, int N, int k, cudaStream_t stream);
extern "C" __global__ void dudux_and_popcount_scores(const unsigned long long* __restrict__ q,
                                          const unsigned long long* __restrict__ keys,
                                          int N, int pack_words,
                                          unsigned int* __restrict__ scores);
extern "C" __global__ void dudux_and_popcount_scores_indexed(const unsigned long long* __restrict__ q,
                                           const unsigned long long* __restrict__ keys_all,
                                           const int* __restrict__ cand_idx,
                                           int N_cand, int pack_words,
                                           unsigned int* __restrict__ scores);
#endif

#include "dudux/core/bitvector.hpp"
#include "dudux/core/bitops.hpp"

namespace dudux { namespace nn {

class NanoAttention1b {
public:
    NanoAttention1b(size_t key_bits, size_t value_bits)
        : kbits_(key_bits), vbits_(value_bits) {}

#ifdef DUDUX_ENABLE_CUDA
    ~NanoAttention1b() {
    if (gpu_.d_keys) { cudaFree(gpu_.d_keys); }
    if (gpu_.d_scores) { cudaFree(gpu_.d_scores); }
    if (gpu_.d_idx) { cudaFree(gpu_.d_idx); }
    if (gpu_.d_cand) { cudaFree(gpu_.d_cand); }
    if (gpu_.d_q) { cudaFree(gpu_.d_q); }
    gpu_.d_keys=nullptr; gpu_.d_scores=nullptr; gpu_.d_idx=nullptr; gpu_.d_cand=nullptr; gpu_.d_q=nullptr;
    gpu_.cached_N=0; gpu_.capacity_N=0; gpu_.capacity_cand=0; gpu_.words=0; gpu_.dirty=true;
    }
#endif

    void add(const dudux::core::BitVector& key, const dudux::core::BitVector& value) {
        if (key.size() != kbits_ || value.size() != vbits_) throw std::invalid_argument("NanoAttention1b: size mismatch");
        keys_.push_back(key);
        values_.push_back(value);
#ifdef DUDUX_ENABLE_CUDA
    gpu_.dirty = true;
#endif
    }

    size_t size() const noexcept { return keys_.size(); }
    size_t key_bits() const noexcept { return kbits_; }
    size_t value_bits() const noexcept { return vbits_; }
    bool value_bit(size_t item_idx, size_t bit_idx) const { return values_[item_idx].get(bit_idx); }

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
#ifdef DUDUX_ENABLE_CUDA
    // Délègue à la version stream (flux par défaut 0)
    topk_into_stream(q, k, out, nullptr);
    return;
#endif
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

#ifdef DUDUX_ENABLE_CUDA
    // Variante GPU avec flux CUDA explicite (nullptr => stream 0)
    void topk_into_stream(const dudux::core::BitVector& q, size_t k,
                          std::vector<std::pair<size_t, uint32_t>>& out,
                          void* stream_handle) const {
        cudaStream_t stream = stream_handle ? reinterpret_cast<cudaStream_t>(stream_handle) : 0;
        ensure_keys_on_device_();
        const size_t words = gpu_.words;
        const size_t N = keys_.size();
        ensure_scratch_on_device_((int)N, (int)N);
        cudaMemcpyAsync(gpu_.d_q, q.data(), words * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream);
        int block=256; int grid=(int)((N + block - 1)/block);
        dudux_and_popcount_scores<<<grid,block,0,stream>>>(gpu_.d_q, gpu_.d_keys, (int)N, (int)words, gpu_.d_scores);
        dudux_topk_scores(gpu_.d_scores, gpu_.d_idx, (int)N, (int)k, stream);
        const size_t kk = std::min(k,N);
        std::vector<unsigned int> h_scores(kk); std::vector<int> h_idx(kk);
        cudaMemcpyAsync(h_scores.data(), gpu_.d_scores, kk*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_idx.data(),    gpu_.d_idx,    kk*sizeof(int),         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        out.clear(); out.reserve(std::min(k,N));
        for (size_t i=0;i<kk;++i) out.emplace_back((size_t)h_idx[i], (uint32_t)h_scores[i]);
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){ if (a.second != b.second) return a.second > b.second; return a.first < b.first; });
    }
#endif

    // Variante masquée: 'mask' optionnel (mask[i]==0 -> ignorer), et 'valid_upto' pour causal (exclure i>valid_upto)
    void topk_into_masked(const dudux::core::BitVector& q, size_t k,
                          const std::vector<uint8_t>* mask,
                          size_t valid_upto,
                          std::vector<std::pair<size_t, uint32_t>>& out) const {
        if (q.size() != kbits_) throw std::invalid_argument("NanoAttention1b: query size mismatch");
        out.clear();
        if (k == 0 || keys_.empty()) return;
        struct Entry { uint32_t s; size_t idx; };
        struct MinCmp {
            bool operator()(const Entry& a, const Entry& b) const {
                if (a.s != b.s) return a.s > b.s;
                return a.idx < b.idx;
            }
        };
        std::priority_queue<Entry, std::vector<Entry>, MinCmp> heap;
        const size_t words = (kbits_ + 63) / 64;
        const uint64_t* qw = q.data();
        const size_t n = std::min(valid_upto + 1, keys_.size());
        for (size_t i = 0; i < n; ++i) {
            if (mask && (i < mask->size()) && ((*mask)[i] == 0)) continue;
            const uint64_t* kw = keys_[i].data();
            uint32_t s = 0;
            for (size_t w = 0; w < words; ++w) s += dudux::core::popcount_u64(qw[w] & kw[w]);
            Entry e{s, i};
            if (heap.size() < k) heap.push(e);
            else {
                const Entry& worst = heap.top();
                if (e.s > worst.s || (e.s == worst.s && e.idx < worst.idx)) { heap.pop(); heap.push(e); }
            }
        }
        out.resize(heap.size());
        for (size_t i = out.size(); i-- > 0;) { auto e = heap.top(); heap.pop(); out[i] = {e.idx, e.s}; }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){ if (a.second != b.second) return a.second > b.second; return a.first < b.first; });
    }

    // Variante: top-k restreint à un sous-ensemble de candidats (indices d'items)
    void topk_into_candidates(const dudux::core::BitVector& q, size_t k,
                              const std::vector<size_t>& candidates,
                              std::vector<std::pair<size_t, uint32_t>>& out) const {
        if (q.size() != kbits_) throw std::invalid_argument("NanoAttention1b: query size mismatch");
        out.clear();
        if (k == 0 || keys_.empty() || candidates.empty()) return;
#ifdef DUDUX_ENABLE_CUDA
    // Délègue à la version stream (flux par défaut 0)
    topk_into_candidates_stream(q, k, candidates, out, nullptr);
    return;
#else
        struct Entry { uint32_t s; size_t idx; };
        struct MinCmp { bool operator()(const Entry& a, const Entry& b) const { if (a.s != b.s) return a.s > b.s; return a.idx < b.idx; } };
        std::priority_queue<Entry, std::vector<Entry>, MinCmp> heap;
        const size_t words = (kbits_ + 63) / 64;
        const uint64_t* qw = q.data();
        for (size_t j = 0; j < candidates.size(); ++j) {
            size_t i = candidates[j];
            if (i >= keys_.size()) throw std::out_of_range("candidate index out of range");
            const uint64_t* kw = keys_[i].data();
            uint32_t s = 0; for (size_t w = 0; w < words; ++w) s += dudux::core::popcount_u64(qw[w] & kw[w]);
            Entry e{s, i};
            if (heap.size() < k) heap.push(e);
            else { const Entry& worst = heap.top(); if (e.s > worst.s || (e.s == worst.s && e.idx < worst.idx)) { heap.pop(); heap.push(e);} }
        }
        out.resize(heap.size());
        for (size_t i = out.size(); i-- > 0;) { auto e = heap.top(); heap.pop(); out[i] = {e.idx, e.s}; }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){ if (a.second != b.second) return a.second > b.second; return a.first < b.first; });
#endif
    }

#ifdef DUDUX_ENABLE_CUDA
    // Variante GPU (candidats) avec flux CUDA explicite
    void topk_into_candidates_stream(const dudux::core::BitVector& q, size_t k,
                                     const std::vector<size_t>& candidates,
                                     std::vector<std::pair<size_t, uint32_t>>& out,
                                     void* stream_handle) const {
        cudaStream_t stream = stream_handle ? reinterpret_cast<cudaStream_t>(stream_handle) : 0;
        ensure_keys_on_device_();
        const size_t words = gpu_.words;
        const size_t N = candidates.size();
        ensure_scratch_on_device_((int)N, (int)keys_.size());
        cudaMemcpyAsync(gpu_.d_q, q.data(), words * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream);
        std::vector<int> h_cand(N);
        for (size_t j=0;j<N;++j) { size_t gi=candidates[j]; if (gi>=keys_.size()) throw std::out_of_range("candidate index out of range"); h_cand[j]=(int)gi; }
        cudaMemcpyAsync(gpu_.d_cand, h_cand.data(), N*sizeof(int), cudaMemcpyHostToDevice, stream);
        int block=256; int grid=(int)((N + block - 1)/block);
        dudux_and_popcount_scores_indexed<<<grid,block,0,stream>>>(gpu_.d_q, gpu_.d_keys, gpu_.d_cand, (int)N, (int)words, gpu_.d_scores);
        dudux_topk_scores(gpu_.d_scores, gpu_.d_idx, (int)N, (int)k, stream);
        const size_t kk = std::min(k,N);
        std::vector<unsigned int> h_scores(kk); std::vector<int> h_loc(kk);
        cudaMemcpyAsync(h_scores.data(), gpu_.d_scores, kk*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_loc.data(),    gpu_.d_idx,    kk*sizeof(int),         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        out.clear(); out.reserve(std::min(k,N));
        for (size_t r=0;r<kk;++r) { size_t local=(size_t)h_loc[r]; out.emplace_back(candidates[local], (uint32_t)h_scores[r]); }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){ if (a.second != b.second) return a.second > b.second; return a.first < b.first; });
    }
#endif

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

    // Versions masquées
    void attend_masked(const dudux::core::BitVector& q, size_t k, uint32_t tau_votes,
                       const std::vector<uint8_t>* mask, size_t valid_upto,
                       dudux::core::BitVector& out_value) const {
        std::vector<std::pair<size_t, uint32_t>> tk;
        topk_into_masked(q, k, mask, valid_upto, tk);
        attend_with_topk(tk, tau_votes, out_value);
    }

    void attend_weighted_masked(const dudux::core::BitVector& q, size_t k, uint32_t tau_weight,
                                const std::vector<uint8_t>* mask, size_t valid_upto,
                                dudux::core::BitVector& out_value) const {
        std::vector<std::pair<size_t, uint32_t>> tk;
        topk_into_masked(q, k, mask, valid_upto, tk);
        attend_weighted_with_topk(tk, tau_weight, out_value);
    }

private:
    size_t kbits_;
    size_t vbits_;
    std::vector<dudux::core::BitVector> keys_;
    std::vector<dudux::core::BitVector> values_;
#ifdef DUDUX_ENABLE_CUDA
    struct { unsigned long long* d_keys=nullptr; unsigned int* d_scores=nullptr; int* d_idx=nullptr; int* d_cand=nullptr; unsigned long long* d_q=nullptr; size_t cached_N=0; size_t capacity_N=0; size_t capacity_cand=0; size_t words=0; bool dirty=true; } gpu_;
    void ensure_keys_on_device_() const {
        const size_t words = (kbits_ + 63) / 64;
        const size_t N = keys_.size();
        if (!gpu_.d_keys || gpu_.cached_N != N || gpu_.words != words || gpu_.dirty) {
            if (gpu_.d_keys) { cudaFree(gpu_.d_keys); gpu_.d_keys=nullptr; }
            if (N==0) { gpu_.cached_N=0; gpu_.words=words; gpu_.dirty=false; return; }
            cudaMalloc(&gpu_.d_keys, N * words * sizeof(unsigned long long));
            std::vector<unsigned long long> h_pack(N*words);
            for (size_t i=0;i<N;++i) std::memcpy(&h_pack[i*words], keys_[i].data(), words*sizeof(unsigned long long));
            cudaMemcpy(gpu_.d_keys, h_pack.data(), N*words*sizeof(unsigned long long), cudaMemcpyHostToDevice);
            gpu_.cached_N = N; gpu_.words = words; gpu_.dirty = false;
        }
    }
    void ensure_scratch_on_device_(int needN, int needCand) const {
        // allocate or grow scratch buffers
        if (gpu_.capacity_N < (size_t)needN) {
            if (gpu_.d_scores) cudaFree(gpu_.d_scores);
            if (gpu_.d_idx) cudaFree(gpu_.d_idx);
            gpu_.capacity_N = (size_t)needN;
            cudaMalloc(&gpu_.d_scores, gpu_.capacity_N * sizeof(unsigned int));
            cudaMalloc(&gpu_.d_idx, gpu_.capacity_N * sizeof(int));
        }
        if (gpu_.capacity_cand < (size_t)needCand) {
            if (gpu_.d_cand) cudaFree(gpu_.d_cand);
            gpu_.capacity_cand = (size_t)needCand;
            cudaMalloc(&gpu_.d_cand, gpu_.capacity_cand * sizeof(int));
        }
        if (!gpu_.d_q) cudaMalloc(&gpu_.d_q, gpu_.words * sizeof(unsigned long long));
    }
#endif
};

}} // namespace dudux::nn
