# Other Tasks and Chunked Tasks

Any task the `transformers` pipeline API supports works through
`TransformersModel` with the right `task=`. This page covers which preprocessor
each task brings, and the tasks that split one input into several forward rows.
Module paths below were measured on transformers 5.17.

<!-- test: setup -->
```python
from nnsight import TransformersModel
from PIL import Image

image = Image.new("RGB", (224, 224), (120, 30, 200))       # a solid purple square
```

## Which preprocessor is populated

Which preprocessor is populated depends on the task, and the others are `None`:

| task | populated |
|---|---|
| `image-text-to-text` | `processor`, `tokenizer` (from the processor) |
| `image-classification`, `image-feature-extraction` | `image_processor` |
| `automatic-speech-recognition` | `feature_extractor`, `tokenizer` |
| `audio-classification` | `feature_extractor` |
| `fill-mask`, `text-classification`, `token-classification` | `tokenizer` |

`model.tokenizer` is populated on a VLM. It is `None` for the tasks with no text
side, which is where a `tokenizer.decode` call falls over.

```python
vision = TransformersModel("google/vit-base-patch16-224",
                           task="image-feature-extraction", dispatch=True)

with vision.trace(image):
    patches = vision.layers[6].output.save()

print(patches.shape, "| tokenizer:", vision.tokenizer)
assert patches.shape == (1, 197, 768)          # 196 patches + the CLS token
assert vision.tokenizer is None
```

```
torch.Size([1, 197, 768]) | tokenizer: None
```

The root envoy here is the `ViTModel` itself, whose blocks are `layers`. A path
copied from another vision family — `vision.encoder.layer[i]`, `vision.model...` —
raises `AttributeError: 'TransformersModel' object (nor its module) has attribute
'encoder'`. Print the model before indexing.

## Chunked tasks

Some tasks split one input into several encodings and forward each separately:
token windows past the model's length limit, one entailment pair per candidate
label, a long recording's windows. Those become **rows of the trace's single
forward**, in the order the task yields them, so a read inside the block sees one
row per chunk. How many chunks there are is the task's preprocessing's decision,
not something to compute from the text length — count them the way the task does:

```python
ner = TransformersModel("hf-internal-testing/tiny-random-BertForTokenClassification",
                        task="token-classification", dispatch=True)
long_text = " ".join(["John lives in Paris"] * 200)

with ner.trace("John lives in Paris"):
    short = ner.output.logits.save()

with ner.trace(long_text, stride=16):
    windows = ner.output.logits.save()

# the windows the task's tokenizer call makes, with the same stride
chunks = ner.tokenizer(long_text, truncation=True, return_overflowing_tokens=True,
                       stride=16)["input_ids"]

print(short.shape, windows.shape, len(chunks))
assert short.shape[0] == 1
assert windows.shape[0] == len(chunks) > 1      # one row per window
```

Six things to know about these tasks:

- A chunked invoke is the whole batch. The row count belongs to the task, so
  putting one next to another invoke is refused with `task=... splits this invoke
  into N forward rows`.
- `document-question-answering` and `zero-shot-object-detection` take the task's
  own input dict (`{"image": ..., "question": ...}`), which goes through the
  task's preprocessing rather than being read as a model encoding.
- An encoder-decoder task needs the decoder's side of the forward:
  `whisper` traces with `decoder_input_ids=` and gives one row per audio chunk
  (70 s at `chunk_length_s=30` is 3 rows of `[1500, 384]` at the encoder).
- **Whisper runs its encoder twice.** The first pass is language detection, the
  second is the transcription. A block that does not loop binds to the *first*
  one, so an edit written to change the transcript changes the language guess
  instead and the text comes back unaltered. Wrap it: `for _ in tracer.iter[:2]:`,
  and act on the second iteration.
- `mask-generation` is refused. Its preprocessing runs the model to embed the
  image before yielding one input per batch of candidate points, so there is no
  single forward to assemble. Use `model.pipe(image)`, or build an encoding with
  `model.image_processor` and pass the points as `input_points=`.
- `keypoint-matching` is refused too, for a different reason: its unit input is
  a *pair* of images, which a trace's list convention (one prompt per element)
  would split. Use `model.pipe([image_a, image_b])`, or trace an encoding you
  build with `model.image_processor(images=[image_a, image_b],
  return_tensors='pt')`.

```python
zero_shot = TransformersModel("hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
                              task="zero-shot-classification", dispatch=True)

with zero_shot.trace("one day I will see the world",
                     candidate_labels=["travel", "cooking", "dancing"]):
    per_label = zero_shot.output.logits.save()

print(per_label.shape)         # one row per candidate label
assert per_label.shape[0] == 3
```
