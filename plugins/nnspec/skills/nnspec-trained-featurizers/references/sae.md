# Sparse autoencoders

An `sae` featurizer is a dictionary at one site. `encode` reads its latents, laid out `[rows, pos, features]`.
An `encode`, edits, `decode` bracket changes latents and keeps the error term (what the dictionary does not
explain) exactly. `apply` replaces the value with the reconstruction, the error term dropped. Each kind's
fields are the `nnspec` skill's [featurizers-and-fit reference](../../nnspec/references/featurizers-and-fit.md);
`top_k` and the reductions are its [ops reference](../../nnspec/references/ops.md).

Every document here runs on GPT-2's layer 5 output. The dictionary is the one `sae_train.json` trains below,
on 16 sentences: its latents show the mechanics on a real model, from a training set far too small to name
features by.

## What breaks here

- **A published file says nothing about how it encodes or where it was trained.** State `activation`,
  `centered` and the layer from the release, and compare mean L0 with the release's before using a latent.
- **A latent's id belongs to its file.** An id read from one dictionary means nothing in another, or in the
  same release at another layer.
- **`select` takes ids written in the document.** A value a step computed is refused (`'top.ids' is not a
  selector: an int (2, -1), a slice ... or a list of picks`); rank in one run, write the ids into the next.
- **An index along a `select`ed axis counts within the select list.** With `"select": {"features": [497, 376]}`,
  `features` 0 and 1 in a table are latents 497 and 376.
- **A latent that is zero at a position changes nothing there.** Its ablation is the bracket's null, exactly
  zero; save the plain `encode` read beside an ablation to see which latents were active.
- **A splice is not an edit.** An effect measured through `apply` against the clean forward includes dropping
  the error term; a latent edit inside the bracket is the edit alone.

## Splice in a published dictionary

SAELens and Gemma Scope publish a dictionary as a safetensors file with no nnspec header. For a 768-wide site
and `d_sae` latents it holds:

| tensor | shape | in |
|---|---|---|
| `W_enc` | `[768, d_sae]` | every release |
| `W_dec` | `[d_sae, 768]` | every release |
| `b_enc` | `[d_sae]` | every release |
| `b_dec` | `[768]` | every release |
| `threshold` | `[d_sae]` | a JumpReLU release only |

A Gemma Scope `.npz` is not read; `numpy.load` it and write the arrays with `safetensors.torch.save_file`.
The document states the three fields the file cannot: `activation`, `centered` and `site`, the layer inline.

<!-- test: skip -->
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
    "dictionary": {"kind": "sae", "file_path": "documents/artifacts/sae_release.safetensors",
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
- **`site`**: where the release was trained, `{"component": ..., "layers": <int>}`. A SAELens hook name maps
  by the residual stream's side of the block: `blocks.L.hook_resid_post` is `layer_output` layer L, and
  `blocks.L.hook_resid_pre` is `layer_input` layer L.
- **`mean_l0`** is the release's L0: nonzero latents per token over every token after the first, positions
  past a row's own left out. Compare it with the number the release reports.
- **`kl`** is KL(clean ‖ spliced) at the last token, the direction SAEBench reports.

Refused before the model loads, by name:

- a field left out (`... is a published sae, which says neither how it encodes nor where it was trained`);
- `relu` on a file with a `threshold` (`... holds a JumpReLU threshold; its activation is 'jumprelu', not
  'relu'`), and `jumprelu` on one without;
- shapes that disagree with each other (`W_dec is [500, 768]; with W_enc [768, 512] it is [latents 512, width
  768]`).

Refused once the weights load:

- a use at another layer than `site` (`featurizer 'dictionary' was trained at layer_output layer 5 (its
  site); this use is layer_output layer 6`);
- a file whose width is not the site's (`has no nnspec header and is 700 wide; this document uses 'dictionary'
  at layer_output, whose width is 768`).

Run, and give wrong numbers:

- **A wrong `centered`.** `sae_train.json`'s dictionary, resaved with `save_file` and no metadata plus a
  zero `threshold`, loads as this document's file. It gives mean L0 21.7 stated `centered: false` and 24.625
  stated `true`, as it was trained; the `true` run equals the nnspec bundle's on the same rows to the bit.
- **A TopK file stated as `relu`.** It keeps every positive pre-activation. The TopK dictionary below, resaved
  with no metadata, gives mean L0 15.0 stated `relu` and 14.6 stated `topk` over this document's four
  sentences. An L0 at or below `k` does not prove the activation right: read it from the release.
- **A wrong layer in `site`**, or `site` written as the name of the entries' site, which moves with it.

## Train a dictionary

The harvest runs once, before the fit; the body only encodes it, so the fit makes no model call. `mse` and
`mean_l0` are body ops, so the eval reports them for the trained dictionary and an `epochs: 0` run for its
draw:

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

```text
nnspec run documents/sae_train.json
nnspec table out/sae_train fit.train.mse
value
173.5391845703125
nnspec table out/sae_train fit.train.mean_l0
value
18.985916137695312
```

| | `fit.train.mse` | `fit.train.mean_l0` |
|---|---|---|
| trained, 50 epochs | 173.5 | 19.0 |
| the same document at `"epochs": 0` | 8549.7 | 255.1 |

- **`dictionary.reconstruction` and `dictionary.l1` are the dictionary's own scalars**, over the vectors the
  body's `encode` read. Their history, `fit.history.dictionary.reconstruction`, falls from 5208 after the
  first epoch to 173.5 after the fiftieth, which equals the eval's `mse`.
- **The decoder rows are put back to unit norm** after every optimizer step.
- **A TopK dictionary** is `{"kind": "sae", "latents": 512, "activation": "topk", "k": 16}`: `fit.train.mean_l0`
  14.1 and `fit.train.mse` 168.3, over the 16 training sentences. A drawn `jumprelu` is refused (`a
  'jumprelu' sae's threshold is a published file's; load one with 'file_path'`).
- **The trained file**, `out/sae_train/fit.dictionary.safetensors`, holds `W_enc` `[768, 512]`, `b_enc`
  `[512]`, `W_dec` `[512, 768]` and `b_dec` `[768]`, with a header recording `activation`, `centered` and
  `latents`. A document loading it writes `{"kind": "sae", "file_path": ...}` alone.

## Compare dictionaries in one document

An `encode`'s or an `apply`'s `featurizer` may be a sweep over featurizer names. This document loads
`out/sae_train/` and `out/sae_topk/`, `sae_train.json` with the TopK dictionary above, from the corpus root:

<!-- test: skip -->
```json
{
  "description": "The ReLU dictionary sae_train.json trained and the TopK one (k 16) the same document trains with that activation, compared on four of their training sentences in one document; a TopK bundle states its activation and k, which the header checks. One sweep over the two names, sae, runs in the encode that counts each token's active latents and in the apply that splices the reconstruction in; kl is KL(clean || spliced) at the last token.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"text": {"rows": [
    {"input": "The cat sat on the mat and looked out of the window."}, {"input": "Stock markets fell sharply after the bank raised interest rates."},
    {"input": "She poured the coffee and opened the morning newspaper."}, {"input": "The train to London leaves the station at nine."}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 5}, "head": {"component": "lm_head_output"}},
  "featurizers": {"relu": {"kind": "sae", "file_path": "out/sae_train/fit.dictionary.safetensors"},
                  "topk": {"kind": "sae", "file_path": "out/sae_topk/fit.dictionary.safetensors", "activation": "topk", "k": 16}},
  "steps": {
    "base": {"kind": "forward", "input": "text.input", "interventions": {
      "latents": {"site": "target", "pos": {"index": "1:"}, "ops": [{"op": "encode", "featurizer": {"sweep": ["relu", "topk"], "label": "sae"}}]},
      "logits": {"site": "head", "pos": -1}}},
    "spliced": {"kind": "forward", "input": "text.input", "interventions": {
      "splice": {"site": "target", "pos": {"index": "1:"}, "ops": [{"op": "apply", "featurizer": {"sweep": ["relu", "topk"], "label": "sae"}}]},
      "logits": {"site": "head", "pos": -1}}},
    "active": {"kind": "op", "op": "gt", "of": "base.latents", "with": 0.0},
    "l0": {"kind": "op", "op": "sum", "of": "active", "over": ["features"]}, "mean_l0": {"kind": "op", "op": "mean", "of": "l0"},
    "kl": {"kind": "op", "op": "kl", "of": "base.logits", "against": "spliced.logits"}, "mean_kl": {"kind": "op", "op": "mean", "of": "kl"}
  },
  "saves": ["mean_l0", "mean_kl"]
}
```

```text
nnspec run documents/sae_compare.json
nnspec table out/sae_compare mean_l0 mean_kl
at              mean_l0             mean_kl
{"sae":"relu"}  24.625              0.07636677473783493
{"sae":"topk"}  14.574999809265137  0.06824039667844772
```

- **Each point is one dictionary's use**, keyed by its name: TopK keeps 14.6 latents a token against 24.6, and
  its splice moves the last token's distribution less, 0.068 nats against 0.076.
- **`base` only encodes**, so one forward serves both points; `spliced` writes, and runs one per dictionary.
- **A TopK bundle states `activation` and `k`**; with `file_path` alone the unstated `relu` is refused
  (`header 'activation' is 'topk'; this document expects 'relu'`).

## Rank latents, then ablate one

Rank the latents where the variable is, here at the day token of the seven weekday prompts:

<!-- test: skip -->
```json
{
  "description": "Rank the trained dictionary's latents at the day token of the seven weekday prompts, and count how many rows each fires on.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"days": {"rows": [
    {"input": "If today is Monday, tomorrow is", "answer": " Tuesday"},
    {"input": "If today is Tuesday, tomorrow is", "answer": " Wednesday"},
    {"input": "If today is Wednesday, tomorrow is", "answer": " Thursday"},
    {"input": "If today is Thursday, tomorrow is", "answer": " Friday"},
    {"input": "If today is Friday, tomorrow is", "answer": " Saturday"},
    {"input": "If today is Saturday, tomorrow is", "answer": " Sunday"},
    {"input": "If today is Sunday, tomorrow is", "answer": " Monday"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 5}},
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
nnspec table out/sae_rank top
row  pos  pos_token   input                               value
0    3    ĠMonday     If today is Monday, tomorrow is     {"ids":[497,376,195],"values":[64.8859558105468…
2    3    ĠWednesday  If today is Wednesday, tomorrow is  {"ids":[497,376,54],"values":[63.70417785644531…
6    3    ĠSunday     If today is Sunday, tomorrow is     {"ids":[497,408,38],"values":[63.27565383911133…
```

`nnspec table out/sae_rank fires` lists each latent's count of rows: 15 of the 512 fire at the day token,
latents 497 and 376 on all seven rows, and latent 0 on none. A `top_k` over a value with no rows axis does not
table (`has no rows to table: it is laid out ['k'] ...; read it from plan.results`).

- **Run `fires` before choosing a null latent**, in a published file too: pick one whose count is 0 at the
  positions the ablation writes. Where every latent fires on your prompts, a dead one needs other prompts.

Zero one latent per sweep point inside the bracket. Score the answer against the clean forward, and, with no
answer column needed, the whole distribution:

<!-- test: skip -->
```json
{
  "description": "Zero one latent of the trained dictionary at the day token, one per sweep point, inside an encode and its decode, which keep the error term. Score the answer's probability against the clean forward's, the KL from the clean logits and the largest logit change. Latent 0 never fires there: its point must equal the clean run.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"days": {"rows": [
    {"input": "If today is Monday, tomorrow is", "answer": " Tuesday"},
    {"input": "If today is Tuesday, tomorrow is", "answer": " Wednesday"},
    {"input": "If today is Wednesday, tomorrow is", "answer": " Thursday"},
    {"input": "If today is Thursday, tomorrow is", "answer": " Friday"},
    {"input": "If today is Friday, tomorrow is", "answer": " Saturday"},
    {"input": "If today is Saturday, tomorrow is", "answer": " Sunday"},
    {"input": "If today is Sunday, tomorrow is", "answer": " Monday"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 5}, "head": {"component": "lm_head_output"}},
  "featurizers": {"dictionary": {"kind": "sae", "file_path": "out/sae_train/fit.dictionary.safetensors"}},
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
    "mean_kl": {"kind": "op", "op": "mean", "of": "kl"},
    "shift": {"kind": "op", "op": "sub", "of": "zeroed.logits", "with": "clean.logits"},
    "abs_shift": {"kind": "op", "op": "abs", "of": "shift"},
    "max_shift": {"kind": "op", "op": "max", "of": "abs_shift", "over": ["rows", "pos", "vocab_size"]}
  },
  "saves": ["mean_clean", "mean_delta", "mean_kl", "max_shift"]
}
```

```text
nnspec run documents/sae_ablation.json
nnspec table out/sae_ablation mean_clean
value
0.22559413313865662
nnspec table out/sae_ablation mean_delta mean_kl max_shift
at                mean_delta            mean_kl              max_shift
{"features":497}  0.0915023684501648    0.4922812581062317   33.44483947753906
{"features":376}  -0.13322243094444275  0.20359280705451965  4.266571044921875
{"features":0}    0.0                   0.0                  0.0
```

Zeroing latent 376 lowers the answer's probability by 0.133 on average, from 0.226; zeroing 497 raises it by
0.092. Latent 0, which never fires there, changes every score by exactly 0: the bracket's null.

- **`kl` needs no answer column.** KL(clean ‖ ablated) at the last token ranks 497 above 376 (0.49 against
  0.20 nats): 497 moves the whole distribution more while it moves this answer less.
- **A reduction of logits names `vocab_size` in its `over`.** By default `max` removes rows and positions only,
  and `max_shift` without it is one value per token of the vocabulary, `[50257]` at each point.
`{"op": "scale", "factor": 5.0}` in place of the `swap` amplifies a latent; `{"op": "add", "with": 1.0}`
adds to it. A list, `{"features": [376, 497]}`, zeroes several together.

## Find a latent's max-activating examples

A latent's max-activating examples are a `top_k` over `["rows", "pos"]` of its latents, read at every token
after the first:

<!-- test: skip -->
```json
{
  "description": "Max-activating examples of latents 497 and 376 of the dictionary sae_train.json trained: each latent's three largest activations over every row and every position after the first.",
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
  "featurizers": {"dictionary": {"kind": "sae", "file_path": "out/sae_train/fit.dictionary.safetensors"}},
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
nnspec run documents/sae_maxact.json
nnspec table out/sae_maxact top
features  k  row  pos  pos_token   input                                         value
0         0  4    5    Ġtomorrow   If today is Friday, tomorrow is Saturday.     74.52745056152344
0         1  5    5    Ġtomorrow   If today is Saturday, tomorrow is Sunday.     74.2275161743164
0         2  3    5    Ġtomorrow   If today is Thursday, tomorrow is Friday.     74.0833511352539
1         0  2    7    ĠThursday   If today is Wednesday, tomorrow is Thursday.  44.977413177490234
1         1  1    7    ĠWednesday  If today is Tuesday, tomorrow is Wednesday.   40.321510314941406
1         2  3    3    ĠThursday   If today is Thursday, tomorrow is Friday.     37.815608978271484
```

`features` 0 is latent 497, which peaks on " tomorrow" in the weekday template; `features` 1 is latent 376,
which peaks on day names. Each pick is a row and a position, labelled with its token. A position past a row's
own is missing, and `top_k` ranks it below every entry that is there.
