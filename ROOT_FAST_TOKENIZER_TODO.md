# Root-FAST Tokenizer TODO

This is the cold-start TODO for the next compute window. Current state:

- Root/local factorization works. R3 is the quality reference, but its root
  latent is too large.
- Root-FAST continuous DCT coefficients are the strongest compact root
  representation so far.
- Coefficient discretization works with scalar quantization, but plain
  one-token-per-chunk vector VQ is too lossy.
- Product VQ is better than vector VQ, but still not good enough to be the
  final root tokenizer.
- Root-FAST RVQ has now been implemented and evaluated. It is the strongest
  token-like root representation so far.

## Current Artifacts

Code:

```text
src/motiongpt_m4human/factorized/root_fast_codec.py
src/motiongpt_m4human/factorized/root_fast_quantize.py
```

Experiment outputs:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_dct_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_quantized_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_vector_fixed_v2
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_scalar_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_product_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_rvq_v1
```

Main docs:

```text
ROOT_LATENT_COMPRESSION.md
FACTORIZED_TOKENIZER.md
M4HUMAN_EVAL_20260530.md
```

## Key Numbers

Root-only DCT continuous codec, using GT local pose:

| config | values/window | MPJPE | root xz mean | final xz |
| --- | ---: | ---: | ---: | ---: |
| chunk=16, K=4 | 208 | 1.20 mm | 1.12 mm | 0.42 mm |
| chunk=16, K=2 | 104 | 5.02 mm | 4.80 mm | 3.64 mm |
| chunk=32, K=4 | 112 | 4.16 mm | 3.96 mm | 2.94 mm |
| chunk=32, K=2 | 56 | 18.06 mm | 17.61 mm | 11.43 mm |

Product VQ, test196:

| root tokens | chunk | K | vocab/group | bits/window | MPJPE | root xz mean | final xz |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 2 | 256 | 128 | 82.61 mm | 80.09 mm | 106.65 mm |
| 28 | 32 | 2 | 256 | 224 | 53.67 mm | 51.57 mm | 80.14 mm |
| 52 | 16 | 2 | 256 | 416 | 41.20 mm | 39.68 mm | 63.61 mm |

Root-FAST RVQ, best test196 by token budget:

| root tokens | chunk | K | vocab | depth | bits/window | MPJPE | root xz mean | final xz |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 98 | 4 | 1024 | 4 | 80 | 89.97 mm | 87.11 mm | 128.46 mm |
| 16 | 98 | 4 | 512 | 8 | 144 | 54.57 mm | 53.09 mm | 67.76 mm |
| 28 | 32 | 2 | 1024 | 4 | 280 | 35.12 mm | 34.11 mm | 45.47 mm |
| 52 | 16 | 2 | 1024 | 4 | 520 | 23.23 mm | 22.52 mm | 34.07 mm |
| 56 | 32 | 2 | 512 | 8 | 504 | 20.15 mm | 19.64 mm | 16.55 mm |
| 104 | 16 | 2 | 1024 | 8 | 1040 | 7.03 mm | 6.76 mm | 7.47 mm |

Scalar quantization, test196:

| config | scalar codes | bits/window | MPJPE | root xz mean | final xz |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk=16, K=2, 8-bit | 104 | 832 | 9.94 mm | 9.48 mm | 11.98 mm |
| chunk=32, K=4, 8-bit | 112 | 896 | 9.36 mm | 8.91 mm | 12.32 mm |
| chunk=64, K=4, 8-bit | 64 | 512 | 19.08 mm | 18.53 mm | 18.64 mm |

## Interpretation

Token count and vocabulary size are separate controls:

```text
token count:     controlled mainly by chunk size and quantizer layout
vocabulary size: controlled by codebook size per token
capacity:        approximately token_count * log2(vocab_size)
```

Plain vector VQ uses very few tokens but asks one token to represent a full DCT
chunk. That is too hard at vocab sizes up to 1024. Product VQ uses more tokens
and smaller per-token targets, so it improves substantially. Scalar quantization
shows that the coefficients themselves are easy to discretize if enough scalar
codes are allowed.

The updated target is a middle ground:

```text
28-56 root tokens for balanced compression
104 root tokens for high-quality reconstruction
shared vocab size around 256-1024
```

## Completed Compute Items

1. Fixed full vector VQ sweep.

   The fixed sweep confirms the earlier conclusion: full-chunk vector VQ is too
   lossy. Best test196 results are:

   ```text
   13 tokens, vocab 1024: 128.28 mm
    7 tokens, vocab 1024: 139.29 mm
    4 tokens, vocab 1024: 164.43 mm
   ```

2. Root-FAST RVQ implementation and sweep.

   RVQ is implemented as:

   ```text
   coeff vector -> code_1 + residual -> code_2 + ... -> code_R
   ```

   Completed sweep:

   ```text
   chunk sizes: 16, 32, 64, 98, 196
   K:           2, 4
   vocab:       256, 512, 1024
   RVQ depth:   2, 4, 8
   ```

## Next Compute TODO

1. Full tokenizer evaluation with local VQ + Root-FAST RVQ.

   Current Root-FAST numbers use GT local pose and measure root codec error.
   The next full tokenizer evaluation should replace local pose with the trained
   local VQ decoder and root with the selected Root-FAST RVQ reconstruction.

   Evaluate these root operating points first:

   ```text
   aggressive:       chunk=32, K=2, vocab=1024, depth=4  -> 28 root tokens
   balanced:         chunk=16, K=2, vocab=1024, depth=4  -> 52 root tokens
   balanced/high-q:  chunk=32, K=2, vocab=512,  depth=8  -> 56 root tokens
   high-quality:     chunk=16, K=2, vocab=512,  depth=8  -> 104 root tokens
   ```

   Key metrics:

   ```text
   full MPJPE
   root-aligned MPJPE
   root gap
   root xz mean error
   final xz error
   path error
   speed bias
   ```

2. Decide the root-token operating point after full local+root eval.

   Current root-only interpretation:

   ```text
   <= 16 tokens: too lossy for faithful root trajectory
   28 tokens:    plausible aggressive setting
   52-56 tokens: likely balanced setting
   104 tokens:   near-scalar quality, longer sequence
   ```

3. Only after root tokens are stable, train predictors.

   Do not start from mmWave yet. First verify proxy tasks:

   ```text
   local VQ tokens -> Root-FAST tokens
   local continuous features -> Root-FAST tokens
   ```

   Then move to:

   ```text
   M4Human/mmWave features -> local VQ tokens + Root-FAST tokens
   ```

## Do Not Prioritize

- More R3 training. R3 is already a quality reference, not a compact tokenizer.
- Plain one-token-per-chunk vector VQ. The fixed sweep confirms it is too lossy.
- Larger product VQ vocab. RVQ is now a better token-like path.
- New root trajectory neural losses before the token representation is settled.
