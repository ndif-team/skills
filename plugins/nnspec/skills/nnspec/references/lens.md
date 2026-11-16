# The logit lens and the tuned lens

What each layer predicts, read through the model's own head or through a trained translator. The basic lens,
a `readout` step over every layer's residual stream with its wiring check, is `ioi_lens.json` in
[SKILL.md](../SKILL.md). This guide adds what the model itself says at a position, the lens's variations, and
the tuned lens. Nothing here writes into the forward.

## What breaks

- **Reading only the output.** A prediction can form in the middle layers and fade: on "The Colosseum is in
  the city of", ` Rome` is 0.84 under the layer-9 lens and 0.014 at the output (`p_tuned` at `epochs: 0` and
  `p_final` [below](#the-tuned-lens)). Score the answer at every layer.
- **An early layer's lens read as what the layer knows.** The final norm and head were trained on the last
  layer's stream. Compare layers with each other and with their KL to the output, or train a tuned lens.
- **A lens compared across forms.** Inside an entry `readout` runs on each batch, and as a step on the whole
  value. Over SKILL.md's four rows the two give the same score to the bit in one batch, and differ by up to
  6.6e-06 at `--batch-size 2`. Write the wiring check, and both sides of a null such as the tuned lens's, in
  one form ([debugging](debugging.md#reproducing-a-number)).
- **`apply` inside an entry writes.** A translator applied in an entry is written back, and every later layer
  sees it. A tuned lens's `apply` and `readout` are op steps over a plain read.
- **A tuned lens without its `epochs: 0` point.** Every translator starts at zero, so the untrained tuned lens
  is the logit lens and its gain is exactly 0.0. A gain means something only beside that null, on held-out rows.

## What the model says at a position

The model's own next tokens are a `top_k` step over `logits`. With `ioi_lens.json`'s `model` and `data`, the
rest of `ioi_top.json` is:

```json
{
  "steps": {
    "run": {"kind": "forward", "input": "pairs.input", "interventions": {"logits": {"site": {"component": "logits"}, "pos": -1}}},
    "top5": {"kind": "op", "op": "top_k", "of": "run.logits", "k": 5},
    "p_answer": {"kind": "op", "op": "token_prob", "of": "run.logits", "token": "pairs.base_answer"}
  },
  "saves": ["top5", "p_answer"]
}
```

`nnspec table --csv out/ioi_top top5 p_answer` prints each cell whole, where the aligned table cuts cells at 48
characters. Row 0's `top5` cell, with `p_answer` 0.4460211992263794 beside it:

```text
{"ids":[5335,606,262,1757,607],"probs":[0.4460211992263794,0.21155205368995667,0.07107384502887726,0.05997771397233009,0.045011114329099655],"tokens":["ĠMary","Ġthem","Ġthe","ĠJohn","Ġher"]}
```

- **A `top_k` step has members**: `top5.ids`, `top5.probs` and `top5.tokens` are named, saved and tabled on
  their own. `probs` is a float32 softmax over the whole vocabulary, so it equals `token_prob` of the same id.
- **`tokens` are the tokenizer's pieces**, spelled as `pos_token` is: `Ġ` is GPT-2's leading space.
- **A `top_k` inside an entry binds `ids` and `probs` only**: no `tokens`, and no step can name its members.
- **A different answer at each position**, such as each next token of a repeat, is a shifted column
  ([attention](attention.md#the-correct-next-token-at-each-position)). A generation's text is `detokenize`
  ([generation](generation.md)).

## Variations on the lens

Each is a change to `ioi_lens.json`:

- **Keep only the score.** An entry `{"site": "resid", "pos": -1, "ops": [{"op": "readout"}, {"op":
  "token_prob", "token": "pairs.base_answer"}]}` holds one number per layer, row and position, `[12, 4, 1]`
  here, and the vocabulary never leaves the model. Use it on large runs.
- **The last few tokens.** `"pos": {"index": "-3:"}` on both `h` and `final` lenses each row's last three
  tokens. `kl` pairs position with position, and is 0.0 at layer 11 at each.
- **A sublayer's lens.** `attention_output` or `mlp_output` in place of `layer_output` lenses what one
  sublayer adds to the stream. On the IOI rows attention 9's lens puts 0.98 on the answer, averaged over the
  rows, and no MLP's exceeds 0.002. Only `layer_output`'s last layer is the model's logits, so a sublayer's
  lens has no wiring check of its own: keep the stream's beside it.
- **Two lenses compared.** `kl` of a lens read at `"layers": [9, 3]` against one at `":"` pairs layer 9 with
  9 and 3 with 3, by label (0.0 for two reads of one forward). The side with fewer layers is `of`; an `against`
  that lacks one of `of`'s layers is refused.
- **What `readout` takes.** A component `nnspec describe components` marks `stream` (`layer_output`,
  `layer_input`, `layer_mid`, `attention_output`, `mlp_output`, `token_embeddings`), whole along `hidden_size`.
  Narrow the logits after it, never the stream before.

## The tuned lens

A tuned lens trains one affine translator per layer, `h + h W + b`, so that each layer's lens matches the
model's own next-token distribution. One `affine` featurizer over `layers` holds all of them, one fit trains
them, and the `epochs: 0` point beside it is the logit lens. The fit step's fields and members are in
[featurizers](featurizers.md). Saved as `documents/tuned_lens.json`:

```json
{
  "description": "A tuned lens on GPT-2: one affine translator per layer for layers 0 to 10, in one featurizer over the layers axis, trained on KL(final || lens) at the first 16 positions of 12 sentences. wiring is the logit lens's wiring check. gain is the tuned lens's held-out KL less the logit lens's; at epochs 0 it must be exactly 0. p_tuned and top_tuned read the trained lens on two fact prompts, beside the output's own p_final.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "train": {"rows": [
      {"input": "The river flows north through the valley before it reaches the sea near the old fishing town."},
      {"input": "She opened the window, looked at the rain, and decided to stay at home with a book."},
      {"input": "Most of the students in the class finished the exam early and went outside to play football."},
      {"input": "The company announced on Monday that it would open three new factories in the south of the country."},
      {"input": "When the train finally arrived, the passengers on the platform pushed forward to find a seat."},
      {"input": "He spent the summer working on a farm, where he learned to drive a tractor and milk the cows."},
      {"input": "The museum holds a large collection of paintings, sculptures and coins from the ancient world."},
      {"input": "After the storm, the streets were covered with branches, and many houses had no electricity."},
      {"input": "The new law requires every restaurant in the city to display its prices clearly near the door."},
      {"input": "Scientists have found that the bird can remember the location of thousands of hidden seeds."},
      {"input": "Our neighbours bought a small dog last year, and it barks every time someone walks past the house."},
      {"input": "The team won the final match by two goals, and the fans celebrated in the streets all night."}
    ]},
    "test": {"rows": [
      {"input": "In the morning the market is busy with people buying fresh bread, fruit and vegetables."},
      {"input": "The doctor told him to rest for a week, drink plenty of water and avoid heavy lifting."},
      {"input": "Two hundred years ago the town was a small village with a church, a mill and a few farms."},
      {"input": "The book tells the story of a young woman who leaves her family to travel around the world."}
    ]},
    "facts": {"rows": [
      {"example_id": "colosseum", "input": "The Colosseum is in the city of", "answer": " Rome"},
      {"example_id": "eiffel", "input": "The Eiffel Tower is in the city of", "answer": " Paris"}
    ]}
  },
  "sites": {
    "resid": {"component": "layer_output", "layers": "0:11"},
    "every": {"component": "layer_output", "layers": ":"},
    "head": {"component": "logits"}
  },
  "featurizers": {"lens": {"kind": "affine", "over": ["layers"]}},
  "steps": {
    "lfw": {"kind": "forward", "input": "test.input", "interventions": {
      "h": {"site": "resid", "pos": {"index": "0:16"}},
      "wire": {"site": "every", "pos": {"index": "0:16"}},
      "final": {"site": "head", "pos": {"index": "0:16"}}}},
    "wire_ll": {"kind": "op", "op": "readout", "of": "lfw.wire"},
    "wire_kl": {"kind": "op", "op": "kl", "of": "wire_ll", "against": "lfw.final"},
    "wiring": {"kind": "op", "op": "max", "of": "wire_kl", "over": ["rows", "pos"]},
    "ll": {"kind": "op", "op": "readout", "of": "lfw.h"},
    "ll_kl": {"kind": "op", "op": "kl", "of": "ll", "against": "lfw.final", "reverse": true},
    "ll_mkl": {"kind": "op", "op": "mean", "of": "ll_kl"},
    "fit": {
      "kind": "fit", "train": ["lens"], "objective": {"kl": 1.0},
      "epochs": {"sweep": [0, 4]}, "batch_size": 8, "seed": 0, "optimizer": {"name": "adam", "lr": 0.001},
      "eval": {"held_out": {"train": "test"}, "seen": {}},
      "steps": {
        "run": {"kind": "forward", "input": "train.input", "interventions": {
          "h": {"site": "resid", "pos": {"index": "0:16"}},
          "final": {"site": "head", "pos": {"index": "0:16"}}}},
        "lensed": {"kind": "op", "op": "apply", "featurizer": "fit.lens", "of": "run.h"},
        "tl": {"kind": "op", "op": "readout", "of": "lensed"},
        "kl": {"kind": "op", "op": "kl", "of": "tl", "against": "run.final", "reverse": true},
        "mkl": {"kind": "op", "op": "mean", "of": "kl"}
      }
    },
    "gain": {"kind": "op", "op": "sub", "of": "fit.held_out.mkl", "with": "ll_mkl"},
    "ask": {"kind": "forward", "input": "facts.input", "interventions": {
      "h": {"site": "resid", "pos": -1}, "final": {"site": "head", "pos": -1}}},
    "p_final": {"kind": "op", "op": "token_prob", "of": "ask.final", "token": "facts.answer"},
    "tuned": {"kind": "op", "op": "apply", "featurizer": "fit.lens", "of": "ask.h"},
    "tuned_logits": {"kind": "op", "op": "readout", "of": "tuned"},
    "p_tuned": {"kind": "op", "op": "token_prob", "of": "tuned_logits", "token": "facts.answer"},
    "top_tuned": {"kind": "op", "op": "top_k", "of": "tuned_logits", "k": 3}
  },
  "saves": ["wiring", "ll_mkl", "fit.held_out.mkl", "fit.seen.mkl", "gain", "fit.history.mkl", "p_tuned", "top_tuned.tokens", "p_final"]
}
```

- **`{"kind": "affine", "over": ["layers"]}`** holds one map per layer of its site: `W` `[11, 768, 768]` and
  `b` `[11, 768]`, both zero at the draw, so it starts as the identity exactly. `apply` takes each layer's map
  by the layer's id. A sweep of the site's `layers` instead would run a fit, and its forwards, per layer.
- **The fit's site stops before the last layer** (`"0:11"` is layers 0 to 10), whose lens is the model's own
  logits. `wiring`, the largest KL(lens ‖ final) per layer at every layer, proves it.
- **The objective is KL(final ‖ lens).** The lens is `of`, since an operand cannot carry a `layers` axis that
  `of` lacks, and `"reverse": true` sets the direction. `wiring` is the other direction, as SKILL.md's is.
- **`lfw` scores the logit lens on the held-out rows exactly as the body scores the tuned lens**: the same 16
  positions, which every sentence holds, as a `readout` step. `gain` subtracts the two layer by layer.
- **`eval` runs the body again after the fit**: `{"held_out": {"train": "test"}}` reads `train` as `test`, and
  `{"seen": {}}` reruns the fit's own rows. Their values are `fit.held_out.<op>` and `fit.seen.<op>`.
- **Steps after the fit name the trained lens `fit.lens`**; the bare `lens` there is the untrained draw. The
  logit lens's step is `ll` because steps and featurizers share one namespace, and `lens` is taken.

```text
$ nnspec table out/tuned_lens wiring
layer  value
0      12.636137008666992
10     2.5567445755004883
11     0.0
$ nnspec table out/tuned_lens ll_mkl fit.held_out.mkl fit.seen.mkl gain
at            layer  ll_mkl              fit.held_out.mkl    fit.seen.mkl         gain
{"epochs":0}  0      5.468942642211914   5.468942642211914   5.612302780151367    0.0
{"epochs":0}  10     0.9010816812515259  0.9010816812515259  0.8214883208274841   0.0
{"epochs":4}  0      5.468942642211914   2.9686098098754883  1.621287226676941    -2.500332832336426
{"epochs":4}  5      3.5412392616271973  1.668989658355713   1.126927375793457    -1.8722496032714844
{"epochs":4}  10     0.9010816812515259  0.571384072303772   0.44277918338775635  -0.3296976089477539
$ nnspec table out/tuned_lens p_tuned top_tuned.tokens
at            example_id  layer  row  pos  pos_token  input                               answer  p_tuned                 top_tuned.tokens
{"epochs":0}  colosseum   9      0    9    Ġof        The Colosseum is in the city of      Rome   0.8437438607215881      ["ĠRome","ĠNaples","ĠAthens"]
{"epochs":4}  colosseum   9      0    9    Ġof        The Colosseum is in the city of      Rome   0.25413525104522705     ["ĠRome","ĠLa","ĠMarse"]
```

- **The wiring check holds**: `wiring` is 0.0 at layer 11. Read nothing else until it is.
- **The null holds**: at `{"epochs":0}` `gain` is 0.0 at all eleven layers, and `p_tuned` is the logit lens.
- **Four epochs, eight Adam steps, take layer 0's held-out KL from 5.47 nats to 2.97** and layer 10's from 0.90
  to 0.57. `fit.history.mkl` is the training rows' mean KL after each epoch, 1.82 falling to 1.12.
- **The fit's own rows flatter it**: `fit.seen.mkl` is 1.62 at layer 0 against 2.97 held out. Report the
  held-out gain.
- **A tuned lens predicts the model's output, not the answer.** It moves ` Rome` at layer 9 from 0.84 to 0.25,
  toward the output's 0.014 (`p_final`).
- **Twelve sentences show the mechanics.** Train on far more text before reading a layer's tuned lens.

**Keeping the lens.** A swept fit's bundle file holds one bundle per point, which no featurizer loads. To use
the trained lens in another document, run the fit unswept (`"epochs": 4`) and save `fit.lens`
([featurizers](featurizers.md) has loading).

**Cost.** After each epoch the body runs over every training row with no gradients, and its `readout` step
binds `[layers, rows, positions, vocab_size]` at once, whatever the fit's `batch_size`: 11 × 12 × 16 × 50257
float32 values here, about 425 MB. Scale the rows, positions or layers down before a larger model.
