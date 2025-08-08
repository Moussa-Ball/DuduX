#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "dudux/nn/mlp1b.hpp"
#include "dudux/core/bitvector.hpp"

using dudux::nn::NanoMLP1b;
using dudux::nn::NanoLinear1b;
using dudux::nn::NanoAct1b;
using dudux::core::BitVector;

struct STEMLP1bTrainer {
    NanoMLP1b mlp;
    // shadow weights
    std::vector<float> w1_real; // [hidden, in]
    std::vector<float> w2_real; // [out(=1), hidden]
    BitVector wrow1_bits;       // buffer pour set_weight_row
    BitVector wrow2_bits;
    size_t in, hidden, out;

    STEMLP1bTrainer(size_t in_features, size_t hidden_features, size_t out_features, uint32_t hidden_tau)
        : mlp(in_features, hidden_features, out_features, hidden_tau),
          w1_real(hidden_features * in_features, 0.0f),
          w2_real(out_features * hidden_features, 0.0f),
          wrow1_bits(in_features), wrow2_bits(hidden_features),
          in(in_features), hidden(hidden_features), out(out_features) {
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.f, 0.01f);
        for (auto &w : w1_real) w = nd(rng);
        for (auto &w : w2_real) w = nd(rng);
        rebin_upload_();
    }

    void rebin_upload_() {
        // W1: hidden rows
        for (size_t h=0; h<hidden; ++h) {
            for (size_t i=0;i<in;++i) wrow1_bits.set(i, w1_real[h*in + i] >= 0.0f);
            mlp.lin1().set_weight_row(h, wrow1_bits);
        }
        // W2: out rows (assume out=1 pour ce trainer)
        for (size_t o=0; o<out; ++o) {
            for (size_t h1=0; h1<hidden; ++h1) wrow2_bits.set(h1, w2_real[o*hidden + h1] >= 0.0f);
            mlp.lin2().set_weight_row(o, wrow2_bits);
        }
    }

    // Forward manuel pour récupérer hidden bits
    bool predict(const BitVector &x, uint32_t out_tau, BitVector* hidden_bits_out=nullptr) {
        // hidden scores
        std::vector<uint32_t> hid_scores; mlp.lin1().matvec_popcnt(x, hid_scores);
        BitVector hidden_bits(hidden);
        NanoAct1b act(mlp.act().threshold());
        act.forward(hid_scores, hidden_bits);
        // out scores
        std::vector<uint32_t> out_scores; mlp.lin2().matvec_popcnt(hidden_bits, out_scores);
        if (hidden_bits_out) *hidden_bits_out = hidden_bits;
        return out_scores[0] >= out_tau;
    }

    void train_step(const BitVector &x, uint8_t y, float lr, uint32_t out_tau) {
        BitVector hbits(hidden);
        bool yhat = predict(x, out_tau, &hbits);
        int err = static_cast<int>(y) - static_cast<int>(yhat);
        if (err == 0) return;
        // Output layer update: w2 += lr*err*hbits
        for (size_t h=0; h<hidden; ++h) if (hbits.get(h)) {
            w2_real[h] += lr * static_cast<float>(err);
            if (w2_real[h] > 1.f) w2_real[h] = 1.f; else if (w2_real[h] < -1.f) w2_real[h] = -1.f;
        }
        // First layer update (STE gating via hidden bits): for active hidden units only
        const size_t words = (in + 63) / 64;
        const uint64_t *xd = x.data();
        for (size_t h=0; h<hidden; ++h) if (hbits.get(h)) {
            for (size_t wi=0; wi<words; ++wi) {
                uint64_t xx = xd[wi];
                while (xx) {
                    unsigned long long t = xx & -xx; int b = __builtin_ctzll(xx);
                    size_t pos = wi*64 + (size_t)b;
                    if (pos < in) {
                        float &w = w1_real[h*in + pos];
                        w += lr * static_cast<float>(err);
                        if (w > 1.f) w = 1.f; else if (w < -1.f) w = -1.f;
                    }
                    xx ^= t;
                }
            }
        }
        rebin_upload_();
    }
};

int main(int argc, char** argv) {
    size_t NBIT=1024, H=64, EPOCHS=10, NTR=500, NTE=200;
    float LR=0.05f; uint32_t TAU_H=1, TAU_OUT=1;
    if (argc>=2) NBIT = static_cast<size_t>(std::stoull(argv[1]));
    if (argc>=3) H    = static_cast<size_t>(std::stoull(argv[2]));
    if (argc>=4) EPOCHS = static_cast<size_t>(std::stoull(argv[3]));
    if (argc>=5) NTR  = static_cast<size_t>(std::stoull(argv[4]));
    if (argc>=6) NTE  = static_cast<size_t>(std::stoull(argv[5]));
    if (argc>=7) LR   = std::stof(argv[6]);
    if (argc>=8) TAU_H = static_cast<uint32_t>(std::stoul(argv[7]));
    if (argc>=9) TAU_OUT = static_cast<uint32_t>(std::stoul(argv[8]));

    // Dataset: y=1 si exactement un des deux bits signal est actif (XOR light peut ne pas converger) => on choisit OR
    const size_t S1 = 7 % NBIT; const size_t S2 = 19 % NBIT;
    std::mt19937 rng(2025);
    std::bernoulli_distribution bern(0.03);
    auto make_sample = [&](BitVector &x, uint8_t &y){
        x.resize(NBIT); x.clear();
        for (size_t i=0;i<NBIT;++i) if (bern(rng)) x.set(i,true);
        if (bern(rng)) x.set(S1, true);
        if (bern(rng)) x.set(S2, true);
        y = (x.get(S1) || x.get(S2)) ? 1 : 0;
    };

    std::vector<BitVector> Xtr(NTR, BitVector(NBIT)); std::vector<uint8_t> ytr(NTR,0);
    for (size_t i=0;i<NTR;++i) make_sample(Xtr[i], ytr[i]);
    std::vector<BitVector> Xte(NTE, BitVector(NBIT)); std::vector<uint8_t> yte(NTE,0);
    for (size_t i=0;i<NTE;++i) make_sample(Xte[i], yte[i]);

    STEMLP1bTrainer trainer(NBIT, H, 1, TAU_H);
    for (size_t e=0;e<EPOCHS;++e) {
        size_t correct=0;
        for (size_t i=0;i<NTR;++i) {
            trainer.train_step(Xtr[i], ytr[i], LR, TAU_OUT);
            correct += (trainer.predict(Xtr[i], TAU_OUT) == (ytr[i]!=0));
        }
        std::cout << "Epoch "<<e<<" train_acc="<<(double)correct/NTR<<"\n";
    }
    size_t correct=0; for (size_t i=0;i<NTE;++i) correct += (trainer.predict(Xte[i], TAU_OUT) == (yte[i]!=0));
    std::cout << "Test acc="<<(double)correct/NTE<<"\n";
    return 0;
}
