#pragma once
/**
 * @file router.hpp
 * @brief Routage top-k sur plusieurs mémoires associatives.
 */

#include <vector>
#include <string>
#include <queue>
#include <functional>
#include <algorithm>
#include <utility>
#include <cstdint>
#include "associative_memory.hpp"

namespace dudux { namespace memory {

class RouterTopK {
public:
    explicit RouterTopK(std::vector<const AssociativeMemory*> memories)
        : memories_(std::move(memories)) {}

    // Top-k streaming O(k) mémoire, tie-break (distance, label)
    std::vector<std::pair<std::string, uint32_t>> query_topk(const dudux::core::BitVector& q, size_t k) const {
        std::vector<std::pair<std::string, uint32_t>> result;
        if (k == 0 || memories_.empty()) return result;

        struct Entry { uint32_t d; std::string label; };
        auto better = [](const Entry& a, const Entry& b){
            if (a.d != b.d) return a.d < b.d; // smaller distance is better
            return a.label < b.label;         // tie-break by label
        };
        std::priority_queue<Entry, std::vector<Entry>, decltype(better)> heap(better);

        // Buffer réutilisable pour les sorties partielles (évite allocations récurrentes)
        std::vector<std::pair<std::string, uint32_t>> tmp;
        tmp.reserve(k);

        for (auto* m : memories_) {
            tmp.clear();
            m->query_topk_into(q, k, tmp);
            for (auto& p : tmp) {
                Entry e{p.second, std::move(p.first)}; // move label pour éviter copie
                if (heap.size() < k) {
                    heap.push(std::move(e));
                } else if (better(e, heap.top())) {
                    heap.pop();
                    heap.push(std::move(e));
                }
            }
        }

        result.resize(heap.size());
        for (size_t i = result.size(); i-- > 0;) {
            Entry e = heap.top(); heap.pop();
            result[i] = { std::move(e.label), e.d };
        }
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b){
            if (a.second != b.second) return a.second < b.second;
            return a.first < b.first;
        });
        return result;
    }

    // Variante out-buffer sans allocations récurrentes côté appelant
    void query_topk_into(const dudux::core::BitVector& q, size_t k, std::vector<std::pair<std::string, uint32_t>>& out) const {
        out.clear();
        if (k == 0 || memories_.empty()) return;

        struct Entry { uint32_t d; std::string label; };
        auto better = [](const Entry& a, const Entry& b){
            if (a.d != b.d) return a.d < b.d;
            return a.label < b.label;
        };
        std::priority_queue<Entry, std::vector<Entry>, decltype(better)> heap(better);

        std::vector<std::pair<std::string, uint32_t>> tmp;
        tmp.reserve(k);
        for (auto* m : memories_) {
            tmp.clear();
            m->query_topk_into(q, k, tmp);
            for (auto& p : tmp) {
                Entry e{p.second, std::move(p.first)};
                if (heap.size() < k) {
                    heap.push(std::move(e));
                } else if (better(e, heap.top())) {
                    heap.pop();
                    heap.push(std::move(e));
                }
            }
        }

        const size_t msize = heap.size();
        // Laisser la capacité de 'out' gérée par l'appelant (pré-allocation possible)
        for (size_t i = 0; i < msize; ++i) {
            Entry e = heap.top(); heap.pop();
            out.emplace_back(std::move(e.label), e.d);
        }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){
            if (a.second != b.second) return a.second < b.second;
            return a.first < b.first;
        });
    }

private:
    std::vector<const AssociativeMemory*> memories_;
};

}} // namespace dudux::memory
