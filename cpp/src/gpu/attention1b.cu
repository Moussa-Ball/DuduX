#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <stdint.h>
#include <algorithm>

extern "C" {
// Simple GPU kernel: AND+POPC for packed {0,1} bits
// q: [pack_words], keys: [N, pack_words], scores: [N]
__global__ void dudux_and_popcount_scores(const unsigned long long* __restrict__ q,
                                          const unsigned long long* __restrict__ keys,
                                          int N, int pack_words,
                                          unsigned int* __restrict__ scores) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const unsigned long long* k = keys + (size_t)i * pack_words;
    unsigned int acc = 0u;
    #pragma unroll
    for (int w=0; w<pack_words; ++w) {
        unsigned long long v = q[w] & k[w];
        acc += __popcll((unsigned long long)v);
    }
    scores[i] = acc;
}

// Top-k via Thrust sort_by_key on (score,index); caller will copy first k
void dudux_topk_scores(unsigned int* d_scores, int* d_indices, int N, int k, cudaStream_t stream) {
    thrust::device_ptr<unsigned int> s(d_scores);
    thrust::device_ptr<int> idx(d_indices);
    auto policy = thrust::cuda::par.on(stream);
    // initialize indices 0..N-1
    thrust::sequence(policy, idx, idx + N);
    // sort by score desc (tie-break handled on host by stable host sort/index)
    thrust::sort_by_key(policy, s, s + N, idx, thrust::greater<unsigned int>());
    (void)k; // suppress unused warning; caller copies top-k
}

__global__ void dudux_and_popcount_scores_indexed(const unsigned long long* __restrict__ q,
                                          const unsigned long long* __restrict__ keys_all,
                                          const int* __restrict__ cand_idx,
                                          int N_cand, int pack_words,
                                          unsigned int* __restrict__ scores) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_cand) return;
    int i = cand_idx[j];
    const unsigned long long* k = keys_all + (size_t)i * pack_words;
    unsigned int acc = 0u;
    #pragma unroll
    for (int w=0; w<pack_words; ++w) {
        unsigned long long v = q[w] & k[w];
        acc += __popcll((unsigned long long)v);
    }
    scores[j] = acc;
}
}
