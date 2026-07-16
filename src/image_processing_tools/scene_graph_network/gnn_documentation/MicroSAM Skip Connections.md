# MicroSAM Skip Connections

This document explains how the `use_skip_connection` flag controls the UNETR decoder architecture in micro_sam, why true encoder skip connections are structurally absent from all deployed micro_sam models, and why this cannot be changed without retraining.

## Background: UNETR Skip Connections

In the original UNETR paper, the ViT encoder produces intermediate patch representations at multiple transformer depths (e.g. layers 3, 6, 9, 12 in a 12-layer ViT). These intermediate outputs are used as skip connections into the decoder — analogous to skip connections in U-Net but originating from different depths of a transformer rather than a convolutional encoder.

The decoder processes these skip connections through the `deconv1`, `deconv2`, `deconv3` branches, each starting from the same spatial resolution as the final embedding (64×64) and progressively upsampling to the intermediate spatial targets (128×128, 256×256, 512×512).

## The `use_skip_connection` Flag

**Reference:** `torch_em/model/unetr.py` — `UNETR.__init__` (lines 148–204) and `UNETR.forward` (lines 303–357)

The flag controls two independent things: the **architecture** built at `__init__` time and the **forward pass** path.

### Architecture differences in `__init__`

When `use_skip_connection=True`, `deconv1`–`3` are built as **multi-step stacks** because each intermediate encoder feature starts at 64×64 and needs multiple upsamplings to reach its target scale:

```python
# use_skip_connection=True
deconv1 = Deconv2DBlock(embed_dim=256 → 512)              # 1 step: layer-9 feature → 128×128
deconv2 = Deconv2DBlock(256→512) → Deconv2DBlock(512→256) # 2 steps: layer-6 feature → 256×256
deconv3 = Deconv2DBlock(256→512) → (512→256) → (256→128)  # 3 steps: layer-3 feature → 512×512
deconv4 = ConvBlock2d(in_chans → 64)                      # processes raw input pixels
```

When `use_skip_connection=False`, `deconv1`–`4` are **single-step blocks** chained sequentially because they are not receiving independent encoder inputs — they are a chain processing z12:

```python
# use_skip_connection=False  (what micro_sam builds)
deconv1 = Deconv2DBlock(256 → 512)   # z12  → z9
deconv2 = Deconv2DBlock(512 → 256)   # z9   → z6
deconv3 = Deconv2DBlock(256 → 128)   # z6   → z3
deconv4 = Deconv2DBlock(128 → 64)    # z3   → z0
```

`deconv2` and `deconv3` have entirely different parameter counts between the two modes. The pretrained checkpoint weights are therefore incompatible with the `use_skip_connection=True` architecture — switching the flag without retraining produces a weight shape mismatch.

### Forward pass differences

```python
# torch_em/model/unetr.py:320–341
encoder_outputs = self.encoder(x)

if isinstance(encoder_outputs[-1], list):
    z12, from_encoder = encoder_outputs   # encoder returned (final, [intermediates])
else:
    z12 = encoder_outputs                 # encoder returned only the final embedding

if use_skip_connection:
    from_encoder = from_encoder[::-1]
    z9 = self.deconv1(from_encoder[0])    # intermediate layer 9 output
    z6 = self.deconv2(from_encoder[1])    # intermediate layer 6 output
    z3 = self.deconv3(from_encoder[2])    # intermediate layer 3 output
    z0 = self.deconv4(x)                  # raw input pixels
else:
    z9 = self.deconv1(z12)               # chain from final embedding
    z6 = self.deconv2(z9)
    z3 = self.deconv3(z6)
    z0 = self.deconv4(z3)
```

## Why Encoder Skip Connections Can Never Be Used in MicroSAM

There are two independent blockers, either of which is sufficient on its own.

### Blocker 1: SAM's image encoder has no intermediate output API

**Reference:** `segment_anything/modeling/image_encoder.py:106–116` — `ImageEncoderViT.forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_embed(x)
    if self.pos_embed is not None:
        x = x + self.pos_embed

    for blk in self.blocks:   # runs all transformer blocks silently
        x = blk(x)

    x = self.neck(x.permute(0, 3, 1, 2))
    return x                   # single tensor — no intermediate outputs exposed
```

The encoder is a bare loop over all transformer blocks with a single return value. There is no mechanism to surface intermediate layer outputs. The UNETR skip-connection branch requires `encoder_outputs` to be a tuple `(z12, from_encoder)` where `from_encoder` is a list of intermediate tensors. SAM's encoder never produces this structure.

Forward hooks could capture intermediate block outputs at runtime, but those captured tensors have no path into the decoder — the `DecoderAdapter` receives only a single embedding tensor (see below).

### Blocker 2: `use_skip_connection=False` is hardcoded in `get_unetr`

**Reference:** `micro_sam/instance_segmentation.py:799–808` — `get_unetr`

```python
unetr = UNETR(
    backbone="sam",
    encoder=image_encoder,
    out_channels=out_channels,
    use_sam_stats=True,
    final_activation="Sigmoid",
    use_skip_connection=False,   # hardcoded
    resize_input=True,
    use_conv_transpose=use_conv_transpose,
)
```

This is the only place in the entire micro_sam library where `use_skip_connection` is set. There is no configuration path to override it.

## The DecoderAdapter: Only the Final Embedding

**Reference:** `micro_sam/instance_segmentation.py:715–762` — `DecoderAdapter`

The `DecoderAdapter` is the module used for AIS inference with precomputed embeddings. Its `_forward_impl` receives only `input_` — the final SAM embedding — and internally reproduces the `use_skip_connection=False` chain:

```python
def _forward_impl(self, input_):
    z12 = input_               # (B, 256, 64×64) — the only input

    z9 = self.deconv1(z12)
    z6 = self.deconv2(z9)
    z3 = self.deconv3(z6)
    z0 = self.deconv4(z3)

    updated_from_encoder = [z9, z6, z3]   # decoder-computed, not from encoder

    x = self.base(z12)
    x = self.decoder(x, encoder_inputs=updated_from_encoder)
    x = self.deconv_out(x)

    x = torch.cat([x, z0], dim=1)         # see MicroSAM Decoder Architecture.md
    x = self.decoder_head(x)
    x = self.out_conv(x)
    if self.final_activation is not None:
        x = self.final_activation(x)
    return x
```

The variable names `z9`, `z6`, `z3` are borrowed from UNETR naming conventions (where the number refers to the transformer layer depth), but the values here are entirely decoder-computed from z12, not encoder outputs.

## Summary

| Condition | Status |
| --- | --- |
| `use_skip_connection` flag in micro_sam | Always `False` — hardcoded in `get_unetr` |
| SAM encoder intermediate output API | Does not exist — `ImageEncoderViT.forward` returns a single tensor |
| Pretrained checkpoint compatibility with `use_skip_connection=True` | Incompatible — `deconv2`/`deconv3` have different architectures and parameter counts |
| z9, z6, z3 in `DecoderAdapter` | Decoder-computed proxies, not encoder features |

Enabling true encoder skip connections would require modifying SAM's encoder to return intermediate layer outputs, building the UNETR with `use_skip_connection=True` (a different architecture), and retraining the decoder from scratch.