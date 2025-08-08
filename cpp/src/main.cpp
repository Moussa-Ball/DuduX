/**
 * @file main.cpp
 * @brief Point d'entrée Dudux: encodage binaire, mémoires associatives, routage top-k.
 */

#include <iostream>
#include <string>
#include <vector>
#include "dudux/core/bitvector.hpp"
#include "dudux/core/encoder.hpp"
#include "dudux/memory/associative_memory.hpp"
#include "dudux/memory/router.hpp"

int main(int argc, char** argv) {
    using dudux::core::BitVector;
    using namespace dudux::core;
    using namespace dudux::memory;
    const size_t NBIT = 4096; // Nombre de neurones binaires

    std::string input = (argc > 1 ? std::string(argv[1]) : std::string("bonjour"));

    // Encodage: texte -> vecteur binaire (neurones 0/1 activés)
    BitVector q = encode_string_bloom(input, NBIT);

    // Plusieurs mémoires spécialisées
    AssociativeMemory mem_greetings(NBIT);
    mem_greetings.add(encode_string_bloom("bonjour", NBIT), "salut!\n");
    mem_greetings.add(encode_string_bloom("bonsoir", NBIT), "bonsoir a toi.\n");

    AssociativeMemory mem_smalltalk(NBIT);
    mem_smalltalk.add(encode_string_bloom("comment ca va", NBIT), "je vais bien.\n");
    mem_smalltalk.add(encode_string_bloom("ca va?", NBIT), "oui, et toi?\n");

    AssociativeMemory mem_identity(NBIT);
    mem_identity.add(encode_string_bloom("qui es-tu", NBIT), "Je suis Dudux binaire (0/1).\n");

    RouterTopK router({ &mem_greetings, &mem_smalltalk, &mem_identity });

    const size_t K = 3;
    auto top = router.query_topk(q, K);

    // Estimation BitOPs: chaque distance ≈ NBIT bits XOR+popcount; on fait size() requêtes.
    size_t total_items = mem_greetings.size() + mem_smalltalk.size() + mem_identity.size();
    uint64_t bitops = static_cast<uint64_t>(total_items) * static_cast<uint64_t>(NBIT);

    std::cout << "=== Dudux CLI ===\n";
    std::cout << "Entrée: " << input << "\n";
    std::cout << "Neurones (NBIT): " << NBIT << ", bases: " << total_items << ", K=" << K << "\n";
    std::cout << "Estimation BitOPs (XOR+popcount): ~" << bitops << "\n";
    std::cout << "Top-" << K << " réponses (distance de Hamming minimale):\n";
    for (auto& p : top) std::cout << "  d=" << p.second << " -> " << p.first;

    return 0;
}
