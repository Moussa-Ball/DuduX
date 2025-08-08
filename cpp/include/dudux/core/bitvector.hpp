#pragma once
/**
 * @file bitvector.hpp
 * @brief Vecteur binaire packé (mots 64 bits) avec opérations de base.
 */

#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <algorithm>

#include "bitops.hpp"

namespace dudux { namespace core {

class BitVector {
public:
    explicit BitVector(size_t nbits = 0)
        : nbits_(nbits), words_((nbits + 63) / 64, 0ull) {}

    size_t size() const noexcept { return nbits_; }
    size_t words() const noexcept { return words_.size(); }

    void resize(size_t nbits) {
        nbits_ = nbits;
        words_.assign((nbits + 63) / 64, 0ull);
    }

    void clear() noexcept { std::fill(words_.begin(), words_.end(), 0ull); }

    void set(size_t i, bool v=true) {
        if (i >= nbits_) throw std::out_of_range("BitVector::set index");
        size_t w = i >> 6; // i / 64
        uint64_t m = 1ull << (i & 63ull);
        if (v) words_[w] |= m; else words_[w] &= ~m;
    }

    bool get(size_t i) const {
        if (i >= nbits_) throw std::out_of_range("BitVector::get index");
        size_t w = i >> 6; uint64_t m = 1ull << (i & 63ull);
        return (words_[w] & m) != 0ull;
    }

    const uint64_t* data() const noexcept { return words_.data(); }
    uint64_t* data() noexcept { return words_.data(); }

    void bit_and(const BitVector& other) {
        check_compat(other);
        for (size_t i = 0; i < words_.size(); ++i) words_[i] &= other.words_[i];
        mask_tail();
    }
    void bit_or(const BitVector& other) {
        check_compat(other);
        for (size_t i = 0; i < words_.size(); ++i) words_[i] |= other.words_[i];
        mask_tail();
    }
    void bit_xor(const BitVector& other) {
        check_compat(other);
        for (size_t i = 0; i < words_.size(); ++i) words_[i] ^= other.words_[i];
        mask_tail();
    }

    uint32_t popcount() const noexcept {
        uint32_t acc = 0;
        for (auto w : words_) acc += popcount_u64(w);
        return acc;
    }

    uint32_t hamming(const BitVector& other) const {
        check_compat(other);
        uint32_t acc = 0;
        for (size_t i = 0; i < words_.size(); ++i) acc += popcount_u64(words_[i] ^ other.words_[i]);
        return acc;
    }

    static BitVector from01(const std::string& s) {
        BitVector v(s.size());
        for (size_t i = 0; i < s.size(); ++i) if (s[i] == '1') v.set(i, true);
        return v;
    }
    std::string to01() const {
        std::string s; s.resize(nbits_);
        for (size_t i = 0; i < nbits_; ++i) s[i] = get(i) ? '1' : '0';
        return s;
    }

private:
    void check_compat(const BitVector& other) const {
        if (nbits_ != other.nbits_) throw std::invalid_argument("BitVector size mismatch");
    }
    void mask_tail() {
        size_t valid = nbits_ & 63ull;
        if (valid && !words_.empty()) {
            uint64_t mask = (valid == 64 ? ~0ull : ((1ull << valid) - 1ull));
            words_.back() &= mask;
        }
    }

    size_t nbits_;
    std::vector<uint64_t> words_;
};

inline BitVector bit_and(BitVector a, const BitVector& b) { a.bit_and(b); return a; }
inline BitVector bit_or (BitVector a, const BitVector& b) { a.bit_or(b);  return a; }
inline BitVector bit_xor(BitVector a, const BitVector& b) { a.bit_xor(b); return a; }

}} // namespace dudux::core
