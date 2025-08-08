#!/usr/bin/env python3
import argparse, math

# Rough memory estimator for binary LLM training ≤4GB

def bytes_h(n):
    units=["B","KB","MB","GB","TB"]
    i=0
    x=float(n)
    while x>=1024 and i<len(units)-1:
        x/=1024; i+=1
    return f"{x:.2f} {units[i]}"

parser=argparse.ArgumentParser(description="Rough memory estimator for binary LLM training (Dudux)")
parser.add_argument('--params', type=float, required=True, help='Total params (millions)')
parser.add_argument('--layers', type=int, default=16)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--seq', type=int, default=1024)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--optim', choices=['ste32','ste8','bop'], default='ste8')
parser.add_argument('--act-recompute', action='store_true')
parser.add_argument('--heads', type=int, default=8, help='Attention heads (for top-k bookkeeping)')
parser.add_argument('--topk', type=int, default=8, help='Top-k per head (training-time index buffer)')
parser.add_argument('--cand', type=int, default=None, help='Candidate keys per token per head for scoring (defaults to seq)')
parser.add_argument('--gbps', type=float, default=None, help='CPU throughput in Giga Bit-ops/s (for time estimate)')
args=parser.parse_args()

P = int(args.params*1e6)
# Weights
weights = P/8 # bytes (1-bit)
alpha_tau = args.layers*args.dim*16 # ~16B per channel

# Optim state
if args.optim=='ste32':
    optim = 4*P
elif args.optim=='ste8':
    optim = 1*P
else:
    optim = 1*P # momentum int8 for BOP

# Activations (binary, roughly batch * L * D/8 * seq). Using bytes.
acts = args.batch * args.layers * (args.dim/8) * args.seq
if args.act_recompute:
    acts *= 0.5

# Attention top-k bookkeeping: indices per (token, head, layer) during training
# Assume 4 bytes per index stored (int32). This is small but include it.
topk_idx = args.batch * args.seq * args.layers * args.heads * args.topk * 4

total = weights + alpha_tau + optim + acts + topk_idx

print(f"Weights:      {bytes_h(weights)}")
print(f"Alpha/Tau:    {bytes_h(alpha_tau)}")
print(f"Optim state:  {bytes_h(optim)} ({args.optim})")
print(f"Activations:  {bytes_h(acts)} (recompute={'on' if args.act_recompute else 'off'})")
print(f"Attn top-k:   {bytes_h(topk_idx)} (heads={args.heads}, k={args.topk})")
print('-'*40)
print(f"Total:        {bytes_h(total)}")
print(f"<= 4GB:       {'YES' if total<=4*1024**3 else 'NO'}")

# Rough compute estimate: popcount ops for attention scoring (per layer)
# candidates defaults to seq (full self-attention); use smaller cand if using router/gating
cand = args.seq if args.cand is None else args.cand
pack_words = max(1, int(math.ceil(args.dim/64)))
tokens = args.batch * args.seq
popcounts_per_layer = tokens * args.heads * cand * pack_words
total_popcounts = popcounts_per_layer * args.layers
print('-'*40)
print(f"Est popcounts: {total_popcounts/1e9:.3f} G (layers={args.layers}, heads={args.heads}, cand={cand}, words={pack_words})")
if args.gbps is not None and args.gbps>0:
    secs = total_popcounts / (args.gbps*1e9)
    print(f"Est time/loss-step (attn scoring only): {secs:.2f} s @ {args.gbps} GBitOP/s")
