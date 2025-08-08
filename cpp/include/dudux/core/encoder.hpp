#pragma once
/**
 * @file encoder.hpp
 * @brief Encodage texte -> vecteur binaire via hachage (type Bloom) pour Dudux Core.
 */

#include <cstdint>
#include <string>
#include <vector>
#include "bitvector.hpp"

namespace dudux { namespace core {

static inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x = x ^ (x >> 31);
    return x;
}

inline BitVector encode_string_bloom(const std::string& s, size_t nbits, uint64_t seed=0xA5A5A5A5A5A5A5A5ull, int k=3) {
    BitVector v(nbits);
    uint64_t h = seed;
    for (unsigned char c : s) {
        h = splitmix64(h ^ c);
        uint64_t x = h;
        for (int i = 0; i < k; ++i) {
            x = splitmix64(x);
            size_t idx = static_cast<size_t>(x % nbits);
            v.set(idx, true);
        }
    }
    return v;
}

}} // namespace dudux::core
