#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include "dudux/core/bitvector.hpp"
#include "dudux/core/encoder.hpp"
#include "dudux/core/metrics.hpp"
#include "dudux/memory/associative_memory.hpp"
#include "dudux/memory/router.hpp"
#include "dudux/nn/attention1b.hpp"

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
    size_t NBIT = DUDUX_BENCHDEF_NBIT;
    size_t NITEMS = DUDUX_BENCHDEF_NITEMS;
    size_t Q = DUDUX_BENCHDEF_Q;
    size_t M = DUDUX_BENCHDEF_M;
    std::vector<size_t> Ks = parse_klist(DUDUX_BENCHDEF_KLIST_STR);
    size_t VBIT = DUDUX_BENCHDEF_VBIT;
    size_t QBATCH = DUDUX_BENCHDEF_QBATCH;

    // Optional CLI overrides: NBIT NITEMS Q M KLIST (comma-separated) VBIT METRICS(0/1) QBATCH
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
    for (size_t i = 0; i < Q; ++i) queries.emplace_back("item_" + std::to_string(i * 7 % NITEMS));

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
    dudux::core::metrics::enable(metrics_on);
    std::cout << "metrics_enabled: " << (metrics_on ? 1 : 0)
#ifdef DUDUX_ENABLE_METRICS
              << " (compile-time ON)\n";
#else
              << " (compile-time OFF)\n";
#endif
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
        for (size_t i = 0; i < NITEMS; ++i) {
            auto k = encode_string_bloom(corpus[i], NBIT);
            auto v = encode_string_bloom(std::string("val_") + corpus[i], VBIT);
            att.add(k, v);
        }
        std::vector<std::pair<size_t, uint32_t>> scratch;
        BitVector out(VBIT);
        for (size_t K : Ks) {
            dudux::core::metrics::reset();
            auto a0 = clk::now();
            for (size_t b = 0; b < QBATCH; ++b) {
                for (auto& qstr : queries) {
                    auto qv = encode_string_bloom(qstr, NBIT);
                    const uint32_t tau = static_cast<uint32_t>((K + 1) / 2); // majorité
                    att.attend_into(qv, K, tau, out, scratch);
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
    }
    return 0;
}
