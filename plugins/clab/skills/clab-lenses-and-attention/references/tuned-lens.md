# The tuned lens

An early layer's logit lens reads that layer's stream in the last layer's basis. A tuned lens first maps each
layer's stream with a translator of its own, `h + h W + b`, trained so the lens's next-token distribution matches
the model's. The document below runs on the CPU in about 20 seconds on `openai-community/gpt2`. The fit step's
fields (`train`, `objective`, `epochs`, `batch_size`, `seed`, `optimizer`, `eval`, `steps`) and its members are
the `clab` skill's [featurizers-and-fit.md](../../clab/references/featurizers-and-fit.md); this page is the recipe.

## What breaks

- **`apply` inside an entry writes.** Each translated layer is written back, and every later layer sees it. Write
  `apply` and `readout` as op steps over a plain read: the forward stays the model's own.
- **The bare featurizer name after the fit.** `lens` is the untrained draw, the identity; the trained one is
  `fit.lens`.
- **A fit without its `epochs: 0` point.** Every translator starts at zero, so at `epochs: 0` the tuned lens is the
  logit lens and `gain` is exactly 0.0 at every layer. A gain means something only beside that null.
- **A null compared across forms or batch shapes.** `gain` is 0.0 to the bit only where both lenses compute alike.
  Write the logit lens as a `readout` step, as the tuned lens's is. Keep `--batch-size` equal to the fit's
  `batch_size` (8 here, the run's default): `lfw` runs in batches of the one, the fit's eval in batches of the
  other. On this document the null also held at `--batch-size` 2 and 3, and with an entry-form logit lens; the
  entry form does move the wiring check's last bits ([SKILL.md](../SKILL.md)).
- **The direction of the KL.** The objective is KL(final ‖ lens): the lens is `of`, since an operand cannot carry
  a `layers` axis that `of` lacks, and `"reverse": true` sets the direction. The logit lens's `to_final`
  ([SKILL.md](../SKILL.md)) is the other direction, KL(lens ‖ final), and gives other numbers.
- **A per-layer fit by sweep.** A sweep of the site's `layers` makes a fit, and its forwards, per layer. One
  `affine` with `"over": ["layers"]` trains every layer's map in one forward per minibatch.

## The document

```json
{
  "description": "A tuned lens on GPT-2 small: one affine translator `h + h W + b` per layer for layers 0 to 10 (every layer but the last, whose lens is the model's own logits), all eleven in one featurizer over the layers axis, trained in one fit on KL(final || lens) at the first 16 positions of 12 sentences. `ll_mkl` is the logit lens's KL per layer on 4 held-out sentences, scored the same way; `gain` is the tuned lens's held-out mean KL less the logit lens's. The epochs sweep puts the null beside the fit: at epochs 0 every translator is the identity and `gain` is exactly 0. After the fit, `p_tuned` and `top_tuned` read the trained lens (`fit.lens`) on two fact prompts.",
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
    "head": {"component": "logits"}
  },
  "featurizers": {"lens": {"kind": "affine", "over": ["layers"]}},
  "steps": {
    "lfw": {
      "kind": "forward",
      "input": "test.input",
      "interventions": {
        "h": {"site": "resid", "pos": {"index": "0:16"}},
        "final": {"site": "head", "pos": {"index": "0:16"}}
      }
    },
    "ll": {"kind": "op", "op": "readout", "of": "lfw.h"},
    "ll_kl": {"kind": "op", "op": "kl", "of": "ll", "against": "lfw.final", "reverse": true},
    "ll_mkl": {"kind": "op", "op": "mean", "of": "ll_kl"},
    "fit": {
      "kind": "fit",
      "train": ["lens"],
      "objective": {"kl": 1.0},
      "epochs": {"sweep": [0, 4]},
      "batch_size": 8,
      "seed": 0,
      "optimizer": {"name": "adam", "lr": 0.001},
      "eval": {"held_out": {"train": "test"}},
      "steps": {
        "run": {
          "kind": "forward",
          "input": "train.input",
          "interventions": {
            "h": {"site": "resid", "pos": {"index": "0:16"}},
            "final": {"site": "head", "pos": {"index": "0:16"}}
          }
        },
        "lensed": {"kind": "op", "op": "apply", "featurizer": "fit.lens", "of": "run.h"},
        "tl": {"kind": "op", "op": "readout", "of": "lensed"},
        "kl": {"kind": "op", "op": "kl", "of": "tl", "against": "run.final", "reverse": true},
        "mkl": {"kind": "op", "op": "mean", "of": "kl"}
      }
    },
    "gain": {"kind": "op", "op": "sub", "of": "fit.held_out.mkl", "with": "ll_mkl"},
    "ask": {"kind": "forward", "input": "facts.input", "interventions": {"h": {"site": "resid", "pos": -1}}},
    "tuned": {"kind": "op", "op": "apply", "featurizer": "fit.lens", "of": "ask.h"},
    "tuned_logits": {"kind": "op", "op": "readout", "of": "tuned"},
    "p_tuned": {"kind": "op", "op": "token_prob", "of": "tuned_logits", "token": "facts.answer"},
    "top_tuned": {"kind": "op", "op": "top_k", "of": "tuned_logits", "k": 3}
  },
  "saves": ["ll_mkl", "fit.held_out.mkl", "gain", "fit.history.mkl", "p_tuned", "top_tuned.tokens"]
}
```

- **`{"kind": "affine", "over": ["layers"]}`** holds one map per layer of its site's selection: `W`
  `[11, 768, 768]` and `b` `[11, 768]`, both zero at the draw, so it starts as the identity exactly. `apply`
  takes each layer's map by the layer's id. `clab describe featurizers` lists the kinds.
- **The site stops before the last layer** (`"0:11"` is layers 0 to 10). The last layer's lens is the model's own
  logits, which the logit lens's wiring check proves; run that check on the model first.
- **Every row holds the 16 positions** `"0:16"` reads (the shortest sentence is 16 tokens), so `kl` pairs the lens
  and the logits position by position at a static width.
- **`lfw` scores the logit lens on the held-out rows** exactly as the body scores the tuned lens, and `gain`
  subtracts the two layer by layer: below 0 is a tuned lens closer to the model's output.
- **The `eval` substitution** `{"held_out": {"train": "test"}}` runs the body once more after the fit with
  `train` read as `test`; its values are `fit.held_out.<op>`.
- **`ask` and the steps after it** read the trained lens on other prompts: `apply` of `fit.lens` over a plain read
  at the same site, then `readout`. They are outside the fit, so they run once per sweep point.

## Run it and read it back

```bash
clab run documents/tuned_lens.json
clab table out/tuned_lens ll_mkl fit.held_out.mkl gain
```

```text
at            layer  ll_mkl              fit.held_out.mkl    gain
{"epochs":0}  0      5.468942642211914   5.468942642211914   0.0
{"epochs":0}  5      3.5412392616271973  3.5412392616271973  0.0
{"epochs":0}  9      2.358225107192993   2.358225107192993   0.0
{"epochs":0}  10     0.9010816812515259  0.9010816812515259  0.0
{"epochs":4}  0      5.468942642211914   2.9686098098754883  -2.500332832336426
{"epochs":4}  5      3.5412392616271973  1.668989658355713   -1.8722496032714844
{"epochs":4}  9      2.358225107192993   1.4495793581008911  -0.908645749092102
{"epochs":4}  10     0.9010816812515259  0.571384072303772   -0.3296976089477539
```

- **The null holds:** at `{"epochs":0}`, `gain` is 0.0 at all eleven layers.
- **Four epochs** (eight Adam steps on twelve sentences) take layer 0's held-out KL from 5.47 nats to 2.97, and
  layer 10's from 0.90 to 0.57. The gain shrinks with depth, as the logit lens's own KL does.
- **`at`** is the sweep point. In Python a swept result is keyed by it (`Plan.load(out).results["gain"]`); the
  `clab` skill's [sweeps.md](../../clab/references/sweeps.md) has the forms.

`clab table out/tuned_lens fit.history.mkl` is the training rows' mean KL after each epoch, over every layer and
position. The `epochs: 0` point has an empty history, so only the trained point prints:

```text
at            epochs  value
{"epochs":4}  0       1.821115493774414
{"epochs":4}  1       1.43071711063385
{"epochs":4}  2       1.2815499305725098
{"epochs":4}  3       1.1246228218078613
```

`clab table out/tuned_lens p_tuned top_tuned.tokens`, at layer 9:

```text
at            example_id  layer  row  pos  pos_token  input                               answer  p_tuned                 top_tuned.tokens
{"epochs":0}  colosseum   9      0    9    Ġof        The Colosseum is in the city of      Rome   0.8437438607215881      ["ĠRome","ĠNaples","ĠAthens"]
{"epochs":0}  eiffel      9      1    9    Ġof        The Eiffel Tower is in the city of   Paris  0.24841895699501038     ["ĠLondon","ĠParis","ĠAmsterdam"]
{"epochs":4}  colosseum   9      0    9    Ġof        The Colosseum is in the city of      Rome   0.25413525104522705     ["ĠRome","ĠLa","ĠMarse"]
{"epochs":4}  eiffel      9      1    9    Ġof        The Eiffel Tower is in the city of   Paris  0.08908192068338394     ["ĠAmsterdam","ĠParis","ĠLondon"]
```

- **At `epochs: 0` the tuned lens is the logit lens to the bit:** 0.8437438607215881 is the logit lens's
  ` Rome` at layer 9 on the same prompt.
- **A tuned lens predicts the model's output, not the answer.** Trained toward the output distribution, it
  moves ` Rome` at layer 9 from 0.84 to 0.25, toward the output's 0.014.
- **Twelve sentences show the mechanics.** Train on far more text before reading a layer's tuned lens, and judge
  it by `gain` on rows the fit never saw.

## Keeping the lens

A swept fit's bundle file holds one bundle per point, which no featurizer loads. To use the trained lens in
another document, run the fit unswept (`"epochs": 4`) beside a separate `epochs: 0` run, and save `fit.lens`.
Saving and loading a bundle is the `clab` skill's [featurizers-and-fit.md](../../clab/references/featurizers-and-fit.md);
reusing one across documents is the `clab-trained-featurizers` skill's.

## Cost

After each epoch the body runs over every training row with no gradients, and its `readout` step binds
`[layers, rows, positions, vocab_size]` at once, whatever the fit's `batch_size`. Here that is 11 × 12 × 16 ×
50257 float32 values, about 425 MB; scale the rows, positions or layers down before a larger model.
