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
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_scalar_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_product_v1
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

The next target should be a middle ground:

```text
few root tokens, but more than plain vector VQ
shared vocab size around 256-1024
root-only MPJPE closer to scalar 8-bit than product VQ
```

## Next Compute TODO

1. Rerun the fixed full vector VQ sweep.

   The first full vector sweep had an early-stop bug in k-means. The code is now
   fixed. A small post-fix check still looked weak, but the complete fixed sweep
   should be rerun before closing this branch.

   ```bash
   conda run -p /cpfs01/liangbo/data/conda_envs/mgpt \
     python -m src.motiongpt_m4human.factorized.root_fast_quantize \
     --out-dir /cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_vector_fixed_v2 \
     --mode vector \
     --eval-splits val test \
     --chunk-sizes 16 32 64 98 196 \
     --coeff-counts 2 4 \
     --codebook-sizes 64 128 256 512 1024 \
     --kmeans-iters 50 \
     --batch-size 512
   ```

2. Implement Root-FAST RVQ.

   This is the highest-priority next implementation. Fit residual codebooks over
   flattened DCT chunks:

   ```text
   coeff vector -> code_1 + residual -> code_2 + ... -> code_R
   ```

   Suggested first sweep:

   ```text
   chunk sizes: 16, 32
   K:           2, 4
   vocab:       256, 512, 1024
   RVQ depth:   2, 4, 8
   ```

   Expected token counts:

   ```text
   chunk=16, R=4 -> 13 * 4 = 52 root tokens
   chunk=32, R=4 ->  7 * 4 = 28 root tokens
   chunk=16, R=8 -> 13 * 8 = 104 root tokens
   ```

3. Implement Product-RVQ if plain RVQ is not enough.

   Product VQ already showed a useful direction. Product-RVQ can be applied per
   root command dimension:

   ```text
   per-dim DCT coeffs -> residual code_1 ... code_R
   ```

   This will use more tokens, but may approach scalar quantization quality with
   a token-like representation.

4. Combine Root-FAST root reconstruction with local VQ.

   Current Root-FAST numbers use GT local pose and measure root codec error.
   The next full tokenizer evaluation should replace local pose with the trained
   local VQ decoder and root with the Root-FAST reconstruction.

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

5. Decide the root-token operating point.

   Candidate targets:

   ```text
   aggressive:  28 root tokens, vocab 512/1024, root-only MPJPE <= 25 mm
   balanced:    52 root tokens, vocab 512/1024, root-only MPJPE <= 15 mm
   high-quality:104 root tokens, vocab 256/512, root-only MPJPE <= 10 mm
   ```

   A good practical target is likely the balanced setting. It is much smaller
   than R3 and not too long for upstream sequence models.

6. Only after root tokens are stable, train predictors.

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
- Plain one-token-per-chunk vector VQ unless the fixed sweep contradicts the
  current smoke check.
- Larger product VQ vocab without first trying RVQ. Product vocab 256 is still
  improving, but raising vocab alone will make the downstream classifier harder.
- New root trajectory neural losses before the token representation is settled.
