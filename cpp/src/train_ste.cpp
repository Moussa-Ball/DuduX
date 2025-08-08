#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "dudux/nn/linear1b.hpp"
#include "dudux/core/bitvector.hpp"

using dudux::nn::NanoLinear1b;
using dudux::core::BitVector;

struct STEPerceptron1b {
    NanoLinear1b lin;                 // binaire packé (utilisé en forward)
    std::vector<float> w_real;        // poids "shadow" float32 (mise à jour)
    BitVector w_bin_bits;             // buffer binarisé pour set_weight_row
    size_t in_features;

    explicit STEPerceptron1b(size_t in_features)
        : lin(in_features, 1), w_real(in_features, 0.0f), w_bin_bits(in_features), in_features(in_features) {
        // init: w_real ~ N(0, 0.01)
        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0.f, 0.01f);
        for (auto &w : w_real) w = nd(rng);
        rebin_and_upload_();
    }

    // Binarise w_real (seuil 0) -> Maj NanoLinear1b
    void rebin_and_upload_() {
        for (size_t i=0;i<in_features;++i) w_bin_bits.set(i, w_real[i] >= 0.0f);
        lin.set_weight_row(0, w_bin_bits);
    }

    // Forward: score entier (= popcount(x & w_bin)), prédiction binaire via seuil tau
    uint32_t forward_score(const BitVector &x) { std::vector<uint32_t> s; lin.matvec_popcnt(x, s); return s[0]; }
    bool predict(const BitVector &x, uint32_t tau) { return forward_score(x) >= tau; }

    // Un pas d'entraînement STE: gradient approx sur w_real là où x_i=1
    void train_step(const BitVector &x, uint8_t y, float lr, uint32_t tau) {
        bool yhat = predict(x, tau);
        int err = static_cast<int>(y) - static_cast<int>(yhat); // {-1,0,+1}
        if (err != 0) {
            const size_t words = (in_features + 63) / 64;
            const uint64_t *w = x.data();
            for (size_t wi=0; wi<words; ++wi) {
                uint64_t xx = w[wi];
                while (xx) {
                    unsigned long long t = xx & -xx;
                    int b = __builtin_ctzll(xx);
                    size_t pos = wi*64 + (size_t)b;
                    if (pos < in_features) {
                        // STE update: w_real += lr * err (clip)
                        w_real[pos] += lr * static_cast<float>(err);
                        if (w_real[pos] > 1.f) w_real[pos] = 1.f; else if (w_real[pos] < -1.f) w_real[pos] = -1.f;
                    }
                    xx ^= t;
                }
            }
            rebin_and_upload_();
        }
    }
};

int main(int argc, char** argv) {
    size_t NBIT = 1024; // taille d'entrée
    size_t N_TRAIN = 200;
    size_t N_TEST = 100;
    size_t EPOCHS = 5;
    float LR = 0.1f;
    uint32_t TAU = 1; // seuil en votes
    if (argc>=2) NBIT = static_cast<size_t>(std::stoull(argv[1]));
    if (argc>=3) N_TRAIN = static_cast<size_t>(std::stoull(argv[2]));
    if (argc>=4) N_TEST = static_cast<size_t>(std::stoull(argv[3]));
    if (argc>=5) EPOCHS = static_cast<size_t>(std::stoull(argv[4]));
    if (argc>=6) LR = std::stof(argv[5]);
    if (argc>=7) TAU = static_cast<uint32_t>(std::stoul(argv[6]));

    const size_t SIG = 13 % NBIT; // bit signal
    std::mt19937 rng(123);
    std::bernoulli_distribution bern(0.05); // sparsité

    auto make_sample = [&](BitVector &x, uint8_t &y){
        x.resize(NBIT); x.clear();
        for (size_t i=0;i<NBIT;++i) if (bern(rng)) x.set(i,true);
        if (bern(rng)) x.set(SIG, true);
        y = x.get(SIG) ? 1 : 0;
    };

    std::vector<BitVector> Xtr(N_TRAIN, BitVector(NBIT));
    std::vector<uint8_t>   ytr(N_TRAIN, 0);
    for (size_t i=0;i<N_TRAIN;++i) make_sample(Xtr[i], ytr[i]);
    std::vector<BitVector> Xte(N_TEST, BitVector(NBIT));
    std::vector<uint8_t>   yte(N_TEST, 0);
    for (size_t i=0;i<N_TEST;++i) make_sample(Xte[i], yte[i]);

    STEPerceptron1b model(NBIT);

    for (size_t e=0;e<EPOCHS;++e) {
        size_t correct = 0;
        for (size_t i=0;i<N_TRAIN;++i) {
            model.train_step(Xtr[i], ytr[i], LR, TAU);
            correct += (model.predict(Xtr[i], TAU) == (ytr[i]!=0));
        }
        std::cout << "Epoch "<<e<<" train_acc="<< (double)correct/N_TRAIN <<"\n";
    }

    size_t correct = 0;
    for (size_t i=0;i<N_TEST;++i) correct += (model.predict(Xte[i], TAU) == (yte[i]!=0));
    std::cout << "Test acc="<< (double)correct/N_TEST <<"\n";
    return 0;
}
