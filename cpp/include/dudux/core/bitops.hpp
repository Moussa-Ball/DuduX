#pragma once
/**
 * @file bitops.hpp
 * @brief Opérations bitwise de base pour Dudux Core (popcount, AND+popcount).
 */

#include <cstdint>
#include <cstddef>
#include "metrics.hpp"

namespace dudux { namespace core {

static inline uint32_t popcount_u64(uint64_t x) noexcept {
    metrics::on_popcount();
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcountll(static_cast<unsigned long long>(x)));
#else
    uint32_t c = 0;
    while (x) { x &= (x - 1); ++c; }
    return c;
#endif
}

static inline uint32_t and_popcnt(uint32_t acc, uint64_t a, uint64_t b) noexcept {
    return acc + popcount_u64(a & b);
}

}} // namespace dudux::core
