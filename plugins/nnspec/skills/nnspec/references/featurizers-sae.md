# Sparse autoencoders

An `sae` featurizer is a dictionary at one site. `encode` reads its latents, laid out `[rows, pos,
features]`. An `encode`, edits, `decode` bracket changes latents and keeps the error term, what the dictionary
does not explain, exactly. `apply` replaces the value with the reconstruction, the error term dropped. The fit
step and bundles are in [featurizers](featurizers.md).

Every document here uses one dictionary on GPT-2's layer 5 output, the one `sae_train.json` trains
[below](#train-a-dictionary) on 16 sentences: enough to show the mechanics on a real model, far too little to
name features by.

## What breaks

- **A published file says neither how it encodes nor where it was trained, and a wrong statement of either
  runs.** State `activation`, `centered` and the layer from the release, and compare mean L0 with the
  release's before using a latent.
- **A latent's id belongs to its file.** It means nothing in another dictionary, or in the same release at
  another layer.
- **An index along a `select`ed axis counts within the select list.** With `"select": {"features": [497,
  376]}`, `features` 0 and 1 in a table are latents 497 and 376.
- **A latent that is zero at a position changes nothing there.** Rank the latents that fire before ablating
  one, and keep one that never fires as the null.
- **A splice is not an edit.** An effect measured through `apply` against the clean forward includes dropping
  the error term; a latent edit inside the bracket is the edit alone.
- **GPT-2's first position is an outlier in norm.** Harvest and score from position 1, `{"index": "1:"}`.

## Splice in a published dictionary

SAELens and Gemma Scope publish a dictionary as a file of tensors with no nnspec header: `W_enc` `[width,
d_sae]`, `W_dec` `[d_sae, width]`, `b_enc` `[d_sae]`, `b_dec` `[width]`, and in a JumpReLU release
`threshold` `[d_sae]`. nnspec reads safetensors; convert an `.npz` with `numpy.load` and
`safetensors.torch.save_file`. The document states the three things the file cannot. Saved as
`documents/sae_published.json`, it needs a release's file at `artifacts/sae_release.safetensors` under the
project folder:

```json
{
  "description": "Splice a published JumpReLU SAE into GPT-2. The file holds W_enc, W_dec, b_enc, b_dec and threshold and no nnspec header. The document states what the release says: its activation, whether its encoder subtracts b_dec (centered), and its layer, inline. mean_l0 over every token after the first is the number to compare with the release's; kl is KL(clean || spliced) at the last token.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"text": {"rows": [
    {"input": "The cat sat on the mat and looked out of the window."},
    {"input": "Stock markets fell sharply after the bank raised interest rates."},
    {"input": "She poured the coffee and opened the morning newspaper."},
    {"input": "The train to London leaves the station at nine."}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 5}, "head": {"component": "lm_head_output"}},
  "featurizers": {
    "dictionary": {"kind": "sae", "file_path": "artifacts/sae_release.safetensors",
                   "activation": "jumprelu", "centered": false, "site": {"component": "layer_output", "layers": 5}}
  },
  "steps": {
    "base": {"kind": "forward", "input": "text.input", "interventions": {
      "latents": {"site": "target", "pos": {"index": "1:"}, "ops": [{"op": "encode", "featurizer": "dictionary"}]},
      "logits": {"site": "head", "pos": -1}}},
    "spliced": {"kind": "forward", "input": "text.input", "interventions": {
      "reconstruct": {"site": "target", "pos": {"index": "1:"}, "ops": [{"op": "apply", "featurizer": "dictionary"}]},
      "logits": {"site": "head", "pos": -1}}},
    "kl": {"kind": "op", "op": "kl", "of": "base.logits", "against": "spliced.logits"},
    "active": {"kind": "op", "op": "gt", "of": "base.latents", "with": 0.0},
    "l0": {"kind": "op", "op": "sum", "of": "active", "over": ["features"]},
    "mean_l0": {"kind": "op", "op": "mean", "of": "l0"}
  },
  "saves": ["mean_l0", "kl"]
}
```

- **`activation`**: `jumprelu` when the file holds a `threshold`; `topk` with the release's `k`; else `relu`.
- **`centered`**: `true` when the release's encoder subtracts `b_dec` first, `(h − b_dec) W_enc + b_enc`;
  `false` for `h W_enc + b_enc`, as Gemma Scope's encoders are.
- **`site`**: where the release was trained, written inline as `{"component": ..., "layers": <int>}`. A
  SAELens hook name maps by the side of the block: `blocks.L.hook_resid_post` is `layer_output` layer L, and
  `blocks.L.hook_resid_pre` is `layer_input` layer L.
- **`mean_l0`** is the release's L0: nonzero latents per token, over every token after the first. Compare it
  with the number the release reports.
- **`kl`** is KL(clean ‖ spliced) at the last token: what the splice costs.

Refused before the model loads: a file with one of the three fields left out, `relu` on a file with a
`threshold`, `jumprelu` on one without. Refused once the model loads: a use at another layer than `site`.
These run, and give wrong numbers:

- **A wrong `centered`.** Stated `centered: false`, the stand-in file below gives a `mean_l0` of
  21.700000762939453; stated `true`, as that dictionary was trained, 24.625, its nnspec bundle's own number
  on these rows.
- **A TopK file stated as `relu`**, which keeps every positive pre-activation. The TopK dictionary below,
  resaved the same way, gives 15.024999618530273 stated `relu` and 14.574999809265137 stated `topk` with `k`
  16. An L0 near `k` does not prove the activation right: take it from the release.
- **A wrong layer in `site`**, or `site` written as the name of the entries' own site, which follows them to
  any layer.

With no release at hand, the dictionary `sae_train.json` trains stands in, resaved in the published layout,
with no header and a zero `threshold`; the numbers above come from it:

```python
import torch
from safetensors.torch import load_file, save_file

tensors = load_file("out/sae_train/fit.dictionary.safetensors")      # W_enc, b_enc, W_dec, b_dec
tensors["threshold"] = torch.zeros(tensors["b_enc"].shape[0])
save_file(tensors, "artifacts/sae_release.safetensors")              # no metadata: a published file
```

## Train a dictionary

The harvest runs once, before the fit, and the body only encodes it, so the fit makes no model call. `mse`
and `mean_l0` are body ops, so the `train` eval reports them. Saved as `documents/sae_train.json`:

```json
{
  "description": "Train a 768 -> 512 sparse autoencoder on GPT-2's layer 5 output, every token after the first (an outlier in norm). The harvest runs once; the body only encodes it, so the fit makes no model call. mse and mean_l0 are body ops, so the eval reports them for the trained dictionary and an epochs: 0 run reports them for its draw.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"text": {"rows": [
    {"input": "If today is Monday, tomorrow is Tuesday."},
    {"input": "If today is Tuesday, tomorrow is Wednesday."},
    {"input": "If today is Wednesday, tomorrow is Thursday."},
    {"input": "If today is Thursday, tomorrow is Friday."},
    {"input": "If today is Friday, tomorrow is Saturday."},
    {"input": "If today is Saturday, tomorrow is Sunday."},
    {"input": "If today is Sunday, tomorrow is Monday."},
    {"input": "The cat sat on the mat and looked out of the window."},
    {"input": "Stock markets fell sharply after the bank raised interest rates."},
    {"input": "She poured the coffee and opened the morning newspaper."},
    {"input": "He scored twice in the second half to win the match."},
    {"input": "The recipe calls for two cups of flour and an egg."},
    {"input": "The train to London leaves the station at nine."},
    {"input": "Please send the report to the office by Friday."},
    {"input": "Heavy rain caused flooding in several towns overnight."},
    {"input": "My grandmother baked bread every Sunday morning."}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 5}},
  "featurizers": {"dictionary": {"kind": "sae", "latents": 512}},
  "steps": {
    "harvest": {"kind": "forward", "input": "text.input", "interventions": {"acts": {"site": "target", "pos": {"index": "1:"}}}},
    "fit": {
      "kind": "fit", "train": ["dictionary"], "objective": {"dictionary.reconstruction": 1.0, "dictionary.l1": 5.0},
      "epochs": 50, "batch_size": 8, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01},
      "eval": {"train": {}},
      "steps": {
        "latents": {"kind": "op", "op": "encode", "featurizer": "fit.dictionary", "of": "harvest.acts"},
        "active": {"kind": "op", "op": "gt", "of": "latents", "with": 0.0},
        "l0": {"kind": "op", "op": "sum", "of": "active", "over": ["features"]},
        "mean_l0": {"kind": "op", "op": "mean", "of": "l0"},
        "recon": {"kind": "op", "op": "apply", "featurizer": "fit.dictionary", "of": "harvest.acts"},
        "err": {"kind": "op", "op": "sub", "of": "recon", "with": "harvest.acts"},
        "sq": {"kind": "op", "op": "square", "of": "err"},
        "sse": {"kind": "op", "op": "sum", "of": "sq", "over": ["hidden_size"]},
        "mse": {"kind": "op", "op": "mean", "of": "sse"}
      }
    }
  },
  "saves": ["fit.train.mse", "fit.train.mean_l0", "fit.history.dictionary.reconstruction", "fit.dictionary"]
}
```

| | `fit.train.mse` | `fit.train.mean_l0` |
|---|---|---|
| trained, 50 epochs | 173.5391845703125 | 18.985916137695312 |
| the same document at `"epochs": 0`, the draw | 8549.650390625 | 255.07041931152344 |

- **`dictionary.reconstruction` and `dictionary.l1` are the dictionary's own scalars**, over the vectors the
  body's `encode` read. Their history, `fit.history.dictionary.reconstruction`, falls from 5208 after the
  first epoch to 173.5 after the fiftieth, the eval's `mse`.
- **The decoder rows are put back to unit norm** after every optimizer step.
- **A TopK dictionary** is `{"kind": "sae", "latents": 512, "activation": "topk", "k": 16}`: `fit.train.mse`
  168.27603149414062 and `fit.train.mean_l0` 14.13380241394043 here. A JumpReLU cannot be drawn: its
  threshold is a published file's.
- **The saved bundle**, `out/sae_train/fit.dictionary.safetensors`, records `activation`, `centered` and
  `latents` in its header, so a document loading a ReLU one writes `{"kind": "sae", "file_path": ...}` alone.
  A TopK bundle is loaded with `"activation": "topk"` and its `k` stated; without them it is refused.

Two dictionaries compare in one document by a sweep over their names in the `encode` and the `apply`
([a sweep over featurizers](featurizers.md#a-sweep-over-featurizers)): `sae_published.json` with the
featurizers `relu` and `topk` loaded from `out/sae_train/` and `out/sae_topk/`, `{"sweep": ["relu", "topk"],
"label": "sae"}` as both `featurizer`s, and a `mean` step over `kl`, gives a mean L0 of 24.625 and
14.574999809265137, and a mean `kl` of 0.07636677473783493 and 0.06824039667844772.

## Rank latents, then ablate one

Rank where the variable is. `days` holds the seven prompts `If today is <day>, tomorrow is`, with the next day
as `answer`; these steps read the day token, `pos: -4`, through the trained dictionary:

```json
{
  "featurizers": {"dictionary": {"kind": "sae", "file_path": "out/sae_train/fit.dictionary.safetensors"}},
  "steps": {
    "base": {"kind": "forward", "input": "days.input", "interventions": {
      "latents": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "dictionary"}]}}},
    "top": {"kind": "op", "op": "top_k", "of": "base.latents", "k": 3, "over": "features"},
    "active": {"kind": "op", "op": "gt", "of": "base.latents", "with": 0.0},
    "fires": {"kind": "op", "op": "sum", "of": "active", "over": ["rows", "pos"]}
  },
  "saves": ["top", "fires"]
}
```

```text
$ nnspec table out/sae_rank top
row  pos  pos_token   input                               value
0    3    ĠMonday     If today is Monday, tomorrow is     {"ids":[497,376,195],"values":[64.8859558105468…
2    3    ĠWednesday  If today is Wednesday, tomorrow is  {"ids":[497,376,54],"values":[63.70417785644531…
6    3    ĠSunday     If today is Sunday, tomorrow is     {"ids":[497,408,38],"values":[63.27565383911133…
```

`top_k` over `features` gives each row's largest latents as `ids` and `values`. `fires` counts the rows each
latent fires on: 15 of the 512 fire at the day token, latents 497 and 376 on all seven rows, latent 0 on
none. Run it before choosing a null latent, in a published file too.

Then zero one latent per sweep point inside the bracket, and score against the clean forward. These steps
replace the ones above, with `head` (`lm_head_output`) among the sites:

```json
{
  "steps": {
    "clean": {"kind": "forward", "input": "days.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
    "p_clean": {"kind": "op", "op": "token_prob", "of": "clean.logits", "token": "days.answer"},
    "zeroed": {"kind": "forward", "input": "days.input", "interventions": {
      "ablate": {"site": "target", "pos": -4, "ops": [
        {"op": "encode", "featurizer": "dictionary", "select": {"features": {"sweep": [497, 376, 0]}}},
        {"op": "swap", "with": 0.0},
        {"op": "decode", "featurizer": "dictionary"}]},
      "logits": {"site": "head", "pos": -1}}},
    "p_zeroed": {"kind": "op", "op": "token_prob", "of": "zeroed.logits", "token": "days.answer"},
    "delta": {"kind": "op", "op": "sub", "of": "p_zeroed", "with": "p_clean"},
    "mean_delta": {"kind": "op", "op": "mean", "of": "delta"},
    "mean_clean": {"kind": "op", "op": "mean", "of": "p_clean"},
    "kl": {"kind": "op", "op": "kl", "of": "clean.logits", "against": "zeroed.logits"},
    "mean_kl": {"kind": "op", "op": "mean", "of": "kl"}
  },
  "saves": ["mean_clean", "mean_delta", "mean_kl"]
}
```

```text
$ nnspec table out/sae_ablation mean_delta mean_kl
at                mean_delta            mean_kl
{"features":497}  0.0915023684501648    0.4922812581062317
{"features":376}  -0.13322243094444275  0.20359280705451965
{"features":0}    0.0                   0.0
```

- **The null is the latent that never fires**: zeroing latent 0 changes every score by exactly 0.
- **Zeroing latent 376 lowers the answer's probability by 0.133**, from a `mean_clean` of 0.226; zeroing 497
  raises it by 0.092.
- **`kl` needs no answer column** and ranks the two the other way: 497 moves the whole distribution more
  while it moves this answer less.
- **Other edits take the `swap`'s place**: `{"op": "scale", "factor": 5.0}` amplifies a latent, `{"op": "add",
  "with": 1.0}` adds to it, and a list, `{"features": [376, 497]}`, edits several together.

## Find a latent's max-activating examples

A `top_k` over `["rows", "pos"]` of a latent's activations, read at every token after the first of
`sae_train.json`'s `text` rows:

```json
{
  "steps": {
    "base": {"kind": "forward", "input": "text.input", "interventions": {
      "latents": {"site": "target", "pos": {"index": "1:"},
                  "ops": [{"op": "encode", "featurizer": "dictionary", "select": {"features": [497, 376]}}]}}},
    "top": {"kind": "op", "op": "top_k", "of": "base.latents", "k": 3, "over": ["rows", "pos"]}
  },
  "saves": ["top"]
}
```

```text
$ nnspec table out/sae_maxact top
features  k  row  pos  pos_token   input                                         value
0         0  4    5    Ġtomorrow   If today is Friday, tomorrow is Saturday.     74.52745056152344
0         1  5    5    Ġtomorrow   If today is Saturday, tomorrow is Sunday.     74.2275161743164
0         2  3    5    Ġtomorrow   If today is Thursday, tomorrow is Friday.     74.0833511352539
1         0  2    7    ĠThursday   If today is Wednesday, tomorrow is Thursday.  44.977413177490234
1         1  1    7    ĠWednesday  If today is Tuesday, tomorrow is Wednesday.   40.321510314941406
1         2  3    3    ĠThursday   If today is Thursday, tomorrow is Friday.     37.815608978271484
```

Each pick is a row and a position, labelled with its token. `features` 0 is latent 497 and `features` 1 latent
376, the select list's order. A position past a row's own is missing, and `top_k` ranks it below every entry
that is there.
