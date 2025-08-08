#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#ifdef DUDUX_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#include "dudux/core/bitvector.hpp"
#include "dudux/core/encoder.hpp"
#include "dudux/core/metrics.hpp"
#include "dudux/memory/associative_memory.hpp"
#include "dudux/memory/router.hpp"
#include "dudux/nn/attention1b.hpp"
#include "dudux/nn/mha1b.hpp"

using clk = std::chrono::high_resolution_clock;

// Compile-time defaults (configurable via CMake -D flags)
#ifndef DUDUX_BENCHDEF_NBIT
#define DUDUX_BENCHDEF_NBIT 16384
#endif
#ifndef DUDUX_BENCHDEF_NITEMS
#define DUDUX_BENCHDEF_NITEMS 2000
#endif
#ifndef DUDUX_BENCHDEF_Q
#define DUDUX_BENCHDEF_Q 200
#endif
#ifndef DUDUX_BENCHDEF_M
#define DUDUX_BENCHDEF_M 4
#endif
#ifndef DUDUX_BENCHDEF_KLIST_STR
#define DUDUX_BENCHDEF_KLIST_STR "1,3,5"
#endif
#ifndef DUDUX_BENCHDEF_VBIT
#define DUDUX_BENCHDEF_VBIT 1024
#endif
#ifndef DUDUX_BENCHDEF_QBATCH
#define DUDUX_BENCHDEF_QBATCH 1
#endif
#ifndef DUDUX_BENCHDEF_HEADS
#define DUDUX_BENCHDEF_HEADS 4
#endif
#ifndef DUDUX_BENCHDEF_CAND
#define DUDUX_BENCHDEF_CAND 0 /* 0 => full set (no gating) */
#endif

static std::vector<size_t> parse_klist(const std::string& s) {
    std::vector<size_t> ks;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == ',')) ++i;
        size_t j = i;
        while (j < s.size() && isdigit(static_cast<unsigned char>(s[j]))) ++j;
        if (j > i) ks.push_back(static_cast<size_t>(std::stoul(s.substr(i, j - i))));
        i = j + 1;
    }
    if (ks.empty()) ks.push_back(1);
    return ks;
}

int main(int argc, char** argv) {
    using dudux::core::BitVector;
    using namespace dudux::core;
    using namespace dudux::memory;
    using dudux::nn::NanoAttention1b;
    using dudux::nn::NanoMultiHeadAttention1b;
    size_t NBIT = DUDUX_BENCHDEF_NBIT;
    size_t NITEMS = DUDUX_BENCHDEF_NITEMS;
    size_t Q = DUDUX_BENCHDEF_Q;
    size_t M = DUDUX_BENCHDEF_M;
    std::vector<size_t> Ks = parse_klist(DUDUX_BENCHDEF_KLIST_STR);
    size_t VBIT = DUDUX_BENCHDEF_VBIT;
    size_t QBATCH = DUDUX_BENCHDEF_QBATCH;
    size_t HEADS = DUDUX_BENCHDEF_HEADS;
    size_t CAND = DUDUX_BENCHDEF_CAND;

    // Optional CLI overrides: NBIT NITEMS Q M KLIST (comma-separated) VBIT METRICS(0/1) QBATCH HEADS KH_LIST CAND
    if (argc >= 2) NBIT = static_cast<size_t>(std::stoull(argv[1]));
    if (argc >= 3) NITEMS = static_cast<size_t>(std::stoull(argv[2]));
    if (argc >= 4) Q = static_cast<size_t>(std::stoull(argv[3]));
    if (argc >= 5) M = static_cast<size_t>(std::stoull(argv[4]));
    if (argc >= 6) Ks = parse_klist(argv[5]);
    if (argc >= 7) VBIT = static_cast<size_t>(std::stoull(argv[6]));
    // Optional: argv[7] is metrics_on; keep VBIT from default unless later extended

    std::vector<std::string> corpus;
    corpus.reserve(NITEMS);
    for (size_t i = 0; i < NITEMS; ++i) corpus.emplace_back("item_" + std::to_string(i));

    dudux::core::metrics::reset();
    auto t0 = clk::now();
    AssociativeMemory mem(NBIT);
    for (auto& s : corpus) mem.add(encode_string_bloom(s, NBIT), s);
    auto t1 = clk::now();

    std::vector<std::string> queries;
    std::vector<size_t> query_ids; query_ids.reserve(Q);
    for (size_t i = 0; i < Q; ++i) {
        size_t id = (i * 7) % NITEMS;
        query_ids.push_back(id);
        queries.emplace_back("item_" + std::to_string(id));
    }

    size_t hits = 0;
    for (auto& qstr : queries) {
        auto q = encode_string_bloom(qstr, NBIT);
        auto r = mem.query_topk(q, 5);
        if (!r.empty() && r[0].first == qstr) ++hits;
    }
    auto t2 = clk::now();

    double add_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double qry_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "Added " << NITEMS << " items in " << add_ms << " ms\n";
    std::cout << "Queried " << Q << " items in " << qry_ms << " ms (" << (Q / (qry_ms/1000.0)) << "/s)\n";
    std::cout << "Top-1 exact hits: " << hits << "/" << Q << "\n";
    std::cout << "popcount_calls (single-mem): " << dudux::core::metrics::popcount_calls() << "\n";

    // === Router multi-mémoires: top-k variable, BitOPs réels (si métriques activées) ===
    std::vector<AssociativeMemory> mems; mems.reserve(M);
    for (size_t i = 0; i < M; ++i) mems.emplace_back(NBIT);
    // Répartition round-robin des items
    for (size_t i = 0; i < NITEMS; ++i) {
        mems[i % M].add(encode_string_bloom(corpus[i], NBIT), corpus[i]);
    }
    std::vector<const AssociativeMemory*> mem_ptrs; mem_ptrs.reserve(M);
    for (size_t i = 0; i < M; ++i) mem_ptrs.push_back(&mems[i]);
    RouterTopK router(mem_ptrs);

    // Runtime toggle for metrics: env DUDUX_METRICS=0/1, CLI arg 8 (0/1)
    bool metrics_on =
#ifdef DUDUX_ENABLE_METRICS
        true;
#else
        false;
#endif
    if (const char* env = std::getenv("DUDUX_METRICS")) {
        metrics_on = (std::atoi(env) != 0);
    }
    if (argc >= 8) metrics_on = (std::atoi(argv[7]) != 0);
    if (argc >= 9) QBATCH = static_cast<size_t>(std::stoull(argv[8]));
    if (argc >= 10) HEADS = static_cast<size_t>(std::stoull(argv[9]));
    std::vector<size_t> KHs;
    if (argc >= 11) KHs = parse_klist(argv[10]);
    if (argc >= 12) CAND = static_cast<size_t>(std::stoull(argv[11]));
    dudux::core::metrics::enable(metrics_on);
    std::cout << "metrics_enabled: " << (metrics_on ? 1 : 0)
#ifdef DUDUX_ENABLE_METRICS
              << " (compile-time ON)\n";
#else
              << " (compile-time OFF)\n";
#endif
    
#ifdef DUDUX_ENABLE_CUDA
    size_t freeB=0,totalB=0; if (cudaMemGetInfo(&freeB, &totalB)==cudaSuccess) {
        auto toMB=[](size_t b){ return (double)b/1024.0/1024.0; };
        std::cout << std::fixed << std::setprecision(1)
                  << "GPU VRAM: total=" << toMB(totalB) << " MB, free=" << toMB(freeB) << " MB\n";
    }
#endif
    std::cout << "CAND (candidates per query per head): ";
    if (CAND == 0) std::cout << "ALL (no gating)"; else std::cout << CAND;
    std::cout << "\n";
    for (size_t K : Ks) {
        dudux::core::metrics::reset();
        auto r0 = clk::now();
        size_t rhits = 0;
        for (size_t b = 0; b < QBATCH; ++b) {
            for (auto& qstr : queries) {
                auto qv = encode_string_bloom(qstr, NBIT);
                auto r = router.query_topk(qv, K);
                if (!r.empty() && r[0].first == qstr) ++rhits;
            }
        }
        auto r1 = clk::now();
        double r_ms = std::chrono::duration<double, std::milli>(r1 - r0).count();
    const uint64_t popcnt_calls = dudux::core::metrics::popcount_calls();
    const double bitops_g = static_cast<double>(popcnt_calls) * 64.0 / 1e9; // GBitOPs approx
        std::cout << "Router M=" << M << ", K=" << K
          << ": Queried " << (Q*QBATCH) << " items in " << r_ms << " ms (" << ((Q*QBATCH) / (r_ms/1000.0)) << "/s), "
                  << "Top-1 exact hits: " << rhits << "/" << (Q*QBATCH)
          << ", popcount_calls: " << popcnt_calls
          << ", est_BitOPs_G: " << std::fixed << std::setprecision(3) << bitops_g << "\n";
    }

    // === Attention binaire (clé=BitVector NBIT, valeur=BitVector VBIT) ===
    {
        NanoAttention1b att(NBIT, VBIT);
        // Construire K/V et stocker les clés pour un routeur de candidats léger
        std::vector<BitVector> keys_vec; keys_vec.reserve(NITEMS);
        for (size_t i = 0; i < NITEMS; ++i) {
            auto k = encode_string_bloom(corpus[i], NBIT);
            auto v = encode_string_bloom(std::string("val_") + corpus[i], VBIT);
            att.add(k, v);
            keys_vec.push_back(k);
        }

        // Index inversé postings: pour chaque bit, la liste des items qui l'ont à 1
        std::vector<std::vector<size_t>> postings(NBIT);
        {
            const size_t words = (NBIT + 63) / 64;
            for (size_t i = 0; i < keys_vec.size(); ++i) {
                const uint64_t* w = keys_vec[i].data();
                for (size_t wi = 0; wi < words; ++wi) {
                    uint64_t x = w[wi];
                    while (x) {
                        unsigned long long t = x & -x;
                        int b = __builtin_ctzll(x);
                        size_t pos = wi * 64 + (size_t)b;
                        if (pos < NBIT) postings[pos].push_back(i);
                        x ^= t;
                    }
                }
            }
        }

        auto route_candidates = [&](const BitVector& q, size_t max_cand) {
            std::vector<size_t> cand; cand.reserve(max_cand);
            std::vector<uint8_t> seen(NITEMS, 0);
            const size_t words = (NBIT + 63) / 64;
            const uint64_t* w = q.data();
            size_t bits_used = 0, max_bits = 8; // échantillonner 8 bits actifs
            for (size_t wi = 0; wi < words && bits_used < max_bits; ++wi) {
                uint64_t x = w[wi];
                while (x && bits_used < max_bits) {
                    unsigned long long t = x & -x;
                    int b = __builtin_ctzll(x);
                    size_t pos = wi * 64 + (size_t)b;
                    if (pos < postings.size()) {
                        const auto& lst = postings[pos];
                        for (size_t idx : lst) { if (!seen[idx]) { seen[idx] = 1; cand.push_back(idx); if (cand.size() >= max_cand) break; } }
                        bits_used++;
                    }
                    if (cand.size() >= max_cand) break;
                    x ^= t;
                }
            }
            // Compléter si trop court
            for (size_t i = 0; cand.size() < max_cand && i < NITEMS; ++i) { if (!seen[i]) { seen[i] = 1; cand.push_back(i); } }
            if (cand.size() > max_cand) cand.resize(max_cand);
            return cand;
        };
        std::vector<std::pair<size_t, uint32_t>> scratch;
        BitVector out(VBIT);
        for (size_t K : Ks) {
            dudux::core::metrics::reset();
            auto a0 = clk::now();
            for (size_t b = 0; b < QBATCH; ++b) {
                for (auto& qstr : queries) {
                    auto qv = encode_string_bloom(qstr, NBIT);
                    const uint32_t tau = static_cast<uint32_t>((K + 1) / 2); // majorité
                    if (CAND > 0 && CAND < NITEMS) {
                        auto cand = route_candidates(qv, std::min(CAND, NITEMS));
                        scratch.clear();
                        att.topk_into_candidates(qv, K, cand, scratch);
                        att.attend_with_topk(scratch, tau, out);
                    } else {
                        att.attend_into(qv, K, tau, out, scratch);
                    }
                }
            }
            auto a1 = clk::now();
            double a_ms = std::chrono::duration<double, std::milli>(a1 - a0).count();
            const uint64_t popcnt_calls = dudux::core::metrics::popcount_calls();
            const double bitops_g = static_cast<double>(popcnt_calls) * 64.0 / 1e9; // GBitOPs approx
            std::cout << "Attention VBIT=" << VBIT << ", K=" << K
                      << ": Queried " << (Q*QBATCH) << " items in " << a_ms << " ms (" << ((Q*QBATCH) / (a_ms/1000.0)) << "/s), "
                      << "popcount_calls: " << popcnt_calls
                      << ", est_BitOPs_G: " << std::fixed << std::setprecision(3) << bitops_g << "\n";
        }
        // Variante pondérée (même K sweep)
        for (size_t K : Ks) {
            dudux::core::metrics::reset();
            auto a0 = clk::now();
            for (size_t b = 0; b < QBATCH; ++b) {
                for (auto& qstr : queries) {
                    auto qv = encode_string_bloom(qstr, NBIT);
                    const uint32_t tau_weight = static_cast<uint32_t>((K + 1) / 2);
                    scratch.clear();
                    att.topk_into(qv, K, scratch);
                    att.attend_weighted_with_topk(scratch, tau_weight, out);
                }
            }
            auto a1 = clk::now();
            double a_ms = std::chrono::duration<double, std::milli>(a1 - a0).count();
            const uint64_t popcnt_calls = dudux::core::metrics::popcount_calls();
            const double bitops_g = static_cast<double>(popcnt_calls) * 64.0 / 1e9; // GBitOPs approx
            std::cout << "AttentionWeighted VBIT=" << VBIT << ", K=" << K
                      << ": Queried " << (Q*QBATCH) << " items in " << a_ms << " ms (" << ((Q*QBATCH) / (a_ms/1000.0)) << "/s), "
                      << "popcount_calls: " << popcnt_calls
                      << ", est_BitOPs_G: " << std::fixed << std::setprecision(3) << bitops_g << "\n";
        }

        // === Multi-Head Attention (H têtes, K/V partagés) ===
        {
            NanoMultiHeadAttention1b mha(HEADS, NBIT, VBIT);
            // Réutiliser les mêmes K/V (clé depuis keys_vec)
            for (size_t i = 0; i < NITEMS; ++i) {
                auto v = encode_string_bloom(std::string("val_") + corpus[i], VBIT);
                mha.add(keys_vec[i], v);
            }
            std::vector<BitVector> q_heads(HEADS, BitVector(NBIT));
            std::vector<BitVector> out_heads;
            // Préparer des requêtes différentes par tête (simple rotation du hash)
            for (size_t K : Ks) {
                dudux::core::metrics::reset();
                auto m0 = clk::now();
                for (size_t b = 0; b < QBATCH; ++b) {
                    for (auto& qstr : queries) {
                        auto base = encode_string_bloom(qstr, NBIT);
                        for (size_t h = 0; h < HEADS; ++h) {
                            q_heads[h] = base; // ici identique; on pourrait appliquer une permutation par tête
                        }
                        const uint32_t tau = static_cast<uint32_t>((K + 1) / 2);
                        if (CAND > 0 && CAND < NITEMS) {
                            auto cand = route_candidates(base, std::min(CAND, NITEMS));
                            mha.attend_candidates(q_heads, K, tau, cand, out_heads);
                        } else {
                            mha.attend(q_heads, K, tau, out_heads);
                        }
                    }
                }
                auto m1 = clk::now();
                double m_ms = std::chrono::duration<double, std::milli>(m1 - m0).count();
                const uint64_t popcnt_calls = dudux::core::metrics::popcount_calls();
                const double bitops_g = static_cast<double>(popcnt_calls) * 64.0 / 1e9;
                std::cout << "MHA H=" << HEADS << ", VBIT=" << VBIT << ", K=" << K
                          << ": Queried " << (Q*QBATCH) << " items in " << m_ms << " ms (" << ((Q*QBATCH) / (m_ms/1000.0)) << "/s), "
                          << "popcount_calls: " << popcnt_calls
                          << ", est_BitOPs_G: " << std::fixed << std::setprecision(3) << bitops_g << "\n";

#ifdef DUDUX_ENABLE_CUDA
                // Variante multi-stream (un flux par tête) utilisant les candidats
                if (CAND > 0 && CAND < NITEMS) {
                    dudux::core::metrics::reset();
                    auto ms0 = clk::now();
                    for (size_t b = 0; b < QBATCH; ++b) {
                        for (auto& qstr : queries) {
                            auto base = encode_string_bloom(qstr, NBIT);
                            for (size_t h = 0; h < HEADS; ++h) q_heads[h] = base;
                            const uint32_t tau = static_cast<uint32_t>((K + 1) / 2);
                            auto cand = route_candidates(base, std::min(CAND, NITEMS));
                            mha.attend_candidates_multistream(q_heads, K, tau, cand, out_heads);
                        }
                    }
                    auto ms1 = clk::now();
                    double ms_ms = std::chrono::duration<double, std::milli>(ms1 - ms0).count();
                    const uint64_t popcnt_calls2 = dudux::core::metrics::popcount_calls();
                    const double bitops_g2 = static_cast<double>(popcnt_calls2) * 64.0 / 1e9;
                    std::cout << "MHA_multistream H=" << HEADS << ", VBIT=" << VBIT << ", K=" << K
                              << ": Queried " << (Q*QBATCH) << " items in " << ms_ms << " ms (" << ((Q*QBATCH) / (ms_ms/1000.0)) << "/s), "
                              << "popcount_calls: " << popcnt_calls2
                              << ", est_BitOPs_G: " << std::fixed << std::setprecision(3) << bitops_g2 << "\n";
                }
#endif
            }

            // MHA masquée + K par tête (via attention masquée)
            {
                NanoAttention1b att_mask(NBIT, VBIT);
                for (size_t i = 0; i < NITEMS; ++i) {
                    auto k = encode_string_bloom(corpus[i], NBIT);
                    auto v = encode_string_bloom(std::string("val_") + corpus[i], VBIT);
                    att_mask.add(k, v);
                }
                std::vector<uint8_t> base_mask(NITEMS, 1);
                for (size_t i = 0; i < NITEMS; ++i) if ((i % 10) == 0) base_mask[i] = 0; // invalide 1/10
                BitVector out(VBIT);
                std::vector<std::pair<size_t,uint32_t>> scratch;
                for (size_t K : Ks) {
                    dudux::core::metrics::reset();
                    auto mm0 = clk::now();
                    for (size_t b = 0; b < QBATCH; ++b) {
                        for (size_t qi = 0; qi < Q; ++qi) {
                            auto qv = encode_string_bloom(queries[qi], NBIT);
                            size_t valid_upto = query_ids[qi];
                            for (size_t h = 0; h < HEADS; ++h) {
                                const size_t Kh = (!KHs.empty() && h < KHs.size()) ? KHs[h] : K;
                                const uint32_t tau = static_cast<uint32_t>((Kh + 1) / 2);
                                scratch.clear();
                                att_mask.topk_into_masked(qv, Kh, &base_mask, valid_upto, scratch);
                                att_mask.attend_with_topk(scratch, tau, out);
                            }
                        }
                    }
                    auto mm1 = clk::now();
                    double mm_ms = std::chrono::duration<double, std::milli>(mm1 - mm0).count();
                    const uint64_t popcnt_calls = dudux::core::metrics::popcount_calls();
                    const double bitops_g = static_cast<double>(popcnt_calls) * 64.0 / 1e9;
                    std::cout << "MHA_masked H=" << HEADS << ", VBIT=" << VBIT << ", K(heads)=";
                    if (!KHs.empty()) {
                        std::cout << "[";
                        for (size_t h = 0; h < HEADS; ++h) { if (h) std::cout << ","; std::cout << ((h < KHs.size())?KHs[h]:K); }
                        std::cout << "]";
                    } else {
                        std::cout << K;
                    }
                    std::cout << ": Queried " << (Q*QBATCH) << " items in " << mm_ms << " ms (" << ((Q*QBATCH) / (mm_ms/1000.0)) << "/s), "
                              << "popcount_calls: " << popcnt_calls
                              << ", est_BitOPs_G: " << std::fixed << std::setprecision(3) << bitops_g << "\n";
                }

                // Variante pondérée masquée
                for (size_t K : Ks) {
                    dudux::core::metrics::reset();
                    auto mm0 = clk::now();
                    for (size_t b = 0; b < QBATCH; ++b) {
                        for (size_t qi = 0; qi < Q; ++qi) {
                            auto qv = encode_string_bloom(queries[qi], NBIT);
                            size_t valid_upto = query_ids[qi];
                            for (size_t h = 0; h < HEADS; ++h) {
                                const size_t Kh = (!KHs.empty() && h < KHs.size()) ? KHs[h] : K;
                                const uint32_t tauw = static_cast<uint32_t>((Kh + 1) / 2);
                                scratch.clear();
                                att_mask.topk_into_masked(qv, Kh, &base_mask, valid_upto, scratch);
                                att_mask.attend_weighted_with_topk(scratch, tauw, out);
                            }
                        }
                    }
                    auto mm1 = clk::now();
                    double mm_ms = std::chrono::duration<double, std::milli>(mm1 - mm0).count();
                    const uint64_t popcnt_calls = dudux::core::metrics::popcount_calls();
                    const double bitops_g = static_cast<double>(popcnt_calls) * 64.0 / 1e9;
                    std::cout << "MHA_masked_weighted H=" << HEADS << ", VBIT=" << VBIT << ", K(heads)=";
                    if (!KHs.empty()) {
                        std::cout << "[";
                        for (size_t h = 0; h < HEADS; ++h) { if (h) std::cout << ","; std::cout << ((h < KHs.size())?KHs[h]:K); }
                        std::cout << "]";
                    } else {
                        std::cout << K;
                    }
                    std::cout << ": Queried " << (Q*QBATCH) << " items in " << mm_ms << " ms (" << ((Q*QBATCH) / (mm_ms/1000.0)) << "/s), "
                              << "popcount_calls: " << popcnt_calls
                              << ", est_BitOPs_G: " << std::fixed << std::setprecision(3) << bitops_g << "\n";
                }
            }
        }
    }
    return 0;
}
