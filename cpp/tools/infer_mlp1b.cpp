#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <random>
#include <cstdlib>

#include "dudux/nn/mlp1b.hpp"
#include "dudux/core/bitvector.hpp"

using dudux::nn::NanoMLP1b;
using dudux::core::BitVector;

struct LoadedModel {
    NanoMLP1b mlp;
    uint32_t tau_out;
    LoadedModel(size_t in, size_t hidden, size_t out, uint32_t tau_hidden, uint32_t tau_out_)
        : mlp(in, hidden, out, tau_hidden), tau_out(tau_out_) {}
};

static bool load_packed_mlp(const std::string &path, uint32_t tau_hidden, uint32_t tau_out, LoadedModel &out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { std::cerr << "Cannot open model: " << path << "\n"; return false; }
    char magic[8] = {0}; ifs.read(magic, 8);
    const char ref[8] = {'D','X','1','B','M','L','P','\0'};
    if (std::memcmp(magic, ref, 8) != 0) { std::cerr << "Bad magic\n"; return false; }
    uint64_t in=0, hidden=0, outc=0, pack=0;
    ifs.read(reinterpret_cast<char*>(&in), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&hidden), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&outc), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&pack), sizeof(uint64_t));
    if (pack != 64) { std::cerr << "Unsupported pack_bits ("<<pack<<")\n"; return false; }
    LoadedModel tmp(in, hidden, outc, tau_hidden, tau_out);
    // Load W1 rows
    size_t w1_words = (in + 63) / 64;
    BitVector row1(in);
    for (size_t h=0; h<hidden; ++h) {
        ifs.read(reinterpret_cast<char*>(row1.data()), w1_words * sizeof(uint64_t));
        tmp.mlp.lin1().set_weight_row(h, row1);
    }
    // Load W2 rows
    size_t w2_words = (hidden + 63) / 64;
    BitVector row2(hidden);
    for (size_t o=0; o<outc; ++o) {
        ifs.read(reinterpret_cast<char*>(row2.data()), w2_words * sizeof(uint64_t));
        tmp.mlp.lin2().set_weight_row(o, row2);
    }
    out = std::move(tmp);
    return true;
}

static BitVector parse_input01(const std::string &s) {
    if (!s.empty() && s[0]=='@') {
        std::ifstream ifs(s.substr(1));
        if (!ifs) throw std::runtime_error("cannot open input file");
        std::string line; std::getline(ifs, line);
        return BitVector::from01(line);
    }
    return BitVector::from01(s);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: "<<argv[0]<<" <model.dx1bmlp> [--tau-hidden TH] [--tau-out TO] [--input 01STR|@path] [--random N] [--p SPARSITY] [--seed S]\n";
        return 2;
    }
    std::string model = argv[1];
    uint32_t tau_hidden = 1, tau_out = 1;
    std::string inputArg; int randomN = 0; double p=0.03; unsigned seed=123;
    for (int i=2;i<argc;++i) {
        std::string a = argv[i];
        if (a=="--tau-hidden" && i+1<argc) { tau_hidden = (uint32_t)std::stoul(argv[++i]); }
        else if (a=="--tau-out" && i+1<argc) { tau_out = (uint32_t)std::stoul(argv[++i]); }
        else if (a=="--input" && i+1<argc) { inputArg = argv[++i]; }
        else if (a=="--random" && i+1<argc) { randomN = std::stoi(argv[++i]); }
        else if (a=="--p" && i+1<argc) { p = std::stod(argv[++i]); }
        else if (a=="--seed" && i+1<argc) { seed = (unsigned)std::stoul(argv[++i]); }
    }
    LoadedModel lm(1,1,1,1,1); // placeholder, will be overwritten by loader
    if (!load_packed_mlp(model, tau_hidden, tau_out, lm)) return 3;
    size_t IN = lm.mlp.in_features();

    auto run_one = [&](const BitVector &x){
        std::vector<uint32_t> scores; lm.mlp.forward_scores(x, scores);
        bool yhat = scores[0] >= lm.tau_out;
        std::cout << "score="<<scores[0]<<" yhat="<<(yhat?1:0)<<"\n";
    };

    if (!inputArg.empty()) {
        BitVector x = parse_input01(inputArg);
        if (x.size() != IN) { std::cerr << "input size ("<<x.size()<<") != model in ("<<IN<<")\n"; return 4; }
        run_one(x);
        return 0;
    }

    if (randomN <= 0) randomN = 1;
    std::mt19937 rng(seed);
    std::bernoulli_distribution bern(p);
    for (int i=0;i<randomN;++i) {
        BitVector x(IN); x.clear();
        for (size_t b=0;b<IN;++b) if (bern(rng)) x.set(b,true);
        run_one(x);
    }
    return 0;
}
