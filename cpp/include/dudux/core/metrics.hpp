#pragma once
/**
 * @file metrics.hpp
 * @brief Instrumentation légère pour compter les popcounts réels.
 */

#include <cstdint>
#include <atomic>

namespace dudux { namespace core { namespace metrics {

#if defined(DUDUX_ENABLE_METRICS)
inline std::atomic<uint64_t> g_popcount_calls{0};
inline std::atomic<bool>     g_enabled{true};

inline void reset() noexcept { g_popcount_calls.store(0, std::memory_order_relaxed); }
inline void enable(bool on) noexcept { g_enabled.store(on, std::memory_order_relaxed); }
inline uint64_t popcount_calls() noexcept { return g_popcount_calls.load(std::memory_order_relaxed); }
inline void on_popcount() noexcept {
    if (g_enabled.load(std::memory_order_relaxed)) g_popcount_calls.fetch_add(1, std::memory_order_relaxed);
}
#else
inline void reset() noexcept {}
inline void enable(bool) noexcept {}
inline uint64_t popcount_calls() noexcept { return 0; }
inline void on_popcount() noexcept {}
#endif

}}} // namespace dudux::core::metrics
