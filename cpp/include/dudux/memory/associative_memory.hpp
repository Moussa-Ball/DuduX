#pragma once
/**
 * @file associative_memory.hpp
 * @brief Mémoire associative binaire (distance de Hamming) pour Dudux Memory.
 */

#include <vector>
#include <string>
#include <queue>
#include <functional>
#include <limits>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include "../core/bitvector.hpp"

namespace dudux { namespace memory {

using dudux::core::BitVector;

class AssociativeMemory {
public:
    explicit AssociativeMemory(size_t nbits) : nbits_(nbits) {}

    void add(BitVector key, std::string label) {
        if (key.size() != nbits_) throw std::invalid_argument("AssociativeMemory: key size mismatch");
        items_.emplace_back(std::move(key), std::move(label));
    }

    /**
     * @brief Retourne les k meilleurs (label, distance) par distance de Hamming en O(N log k) et O(k) mémoire.
     *        Tie-break stable: distance croissante puis label croissant pour reproductibilité.
     */
    std::vector<std::pair<std::string, uint32_t>> query_topk(const BitVector& q, size_t k=1) const {
        if (q.size() != nbits_) throw std::invalid_argument("AssociativeMemory: query size mismatch");
        std::vector<std::pair<std::string, uint32_t>> result;
        if (k == 0 || items_.empty()) return result;

        // Entrée dans le tas: (distance, label*) — on évite les copies de string jusqu'à la phase finale.
        struct Entry { uint32_t d; const std::string* label; };
        // Compare "better": distance plus petite d'abord, puis label plus petit.
        // Utilisé comme comparateur du tas pour construire un max-heap selon "better" -> top = pire élément.
        auto better = [](const Entry& a, const Entry& b){
            if (a.d != b.d) return a.d < b.d;            // plus petit = meilleur
            return *a.label < *b.label;                  // tie-break stable
        };
        std::priority_queue<Entry, std::vector<Entry>, decltype(better)> heap(better);

        for (const auto& it : items_) {
            const uint32_t d = q.hamming(it.first);
            const Entry e{d, &it.second};
            if (heap.size() < k) {
                heap.push(e);
            } else if (better(e, heap.top())) {
                // e est meilleur que le pire en tête -> remplace
                heap.pop();
                heap.push(e);
            }
        }

        // Extraction et tri final (distance croissante puis label croissant)
        result.resize(heap.size());
        for (size_t i = result.size(); i-- > 0;) {
            const Entry e = heap.top(); heap.pop();
            result[i] = { *e.label, e.d };
        }
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b){
            if (a.second != b.second) return a.second < b.second;
            return a.first < b.first;
        });
        return result;
    }

    /**
     * @brief Variante sans allocation récurrente côté appelant: remplit un buffer fourni.
     * @details Le buffer `out` est vidé puis rempli avec jusqu'à k éléments triés (distance croissante, tie-break label).
     *          Pour éviter toute allocation, l'appelant peut pré-allouer `out` avec out.reserve(k).
     */
    void query_topk_into(const BitVector& q, size_t k, std::vector<std::pair<std::string, uint32_t>>& out) const {
        if (q.size() != nbits_) throw std::invalid_argument("AssociativeMemory: query size mismatch");
        out.clear();
        if (k == 0 || items_.empty()) return;

        struct Entry { uint32_t d; const std::string* label; };
        auto better = [](const Entry& a, const Entry& b){
            if (a.d != b.d) return a.d < b.d;
            return *a.label < *b.label;
        };
        std::priority_queue<Entry, std::vector<Entry>, decltype(better)> heap(better);

        for (const auto& it : items_) {
            const uint32_t d = q.hamming(it.first);
            const Entry e{d, &it.second};
            if (heap.size() < k) {
                heap.push(e);
            } else if (better(e, heap.top())) {
                heap.pop();
                heap.push(e);
            }
        }

        // Transfert vers out puis tri stable selon critères
        const size_t m = heap.size();
        // Ne pas réserver ici pour laisser le contrôle à l'appelant (pré-allocation possible)
        for (size_t i = 0; i < m; ++i) {
            const Entry e = heap.top(); heap.pop();
            out.emplace_back(*e.label, e.d);
        }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){
            if (a.second != b.second) return a.second < b.second;
            return a.first < b.first;
        });
    }

    size_t size() const noexcept { return items_.size(); }

private:
    size_t nbits_;
    std::vector<std::pair<BitVector, std::string>> items_;
};

}} // namespace dudux::memory
