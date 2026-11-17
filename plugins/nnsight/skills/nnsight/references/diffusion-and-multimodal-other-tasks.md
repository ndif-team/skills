# Other pipeline tasks: speech, vision encoders, classifiers

`TransformersModel` loads any `transformers` pipeline task by its `task=` name,
and reading, editing and gradients work as in [SKILL.md](../SKILL.md). What
changes with the task is the input, the module paths, which run method gives the
task's answer, and how many forward passes or batch rows one input becomes.
Module paths are different for every model class on this page: `print(model)`
before indexing.

## Which preprocessor a task has

Only the preprocessors the task uses are populated. The others are `None`, which
is where a `model.tokenizer.decode(...)` fails on a task with no text side.

| task | populated |
|---|---|
| `automatic-speech-recognition` | `feature_extractor`, `tokenizer` |
| `audio-classification` | `feature_extractor` |
| `image-classification`, `image-feature-extraction` | `image_processor` |
| `image-text-to-text` | `processor`, with `tokenizer` and `image_processor` taken from it |
| `fill-mask`, `text-classification`, `token-classification`, `zero-shot-classification` | `tokenizer` |

## Whisper: speech recognition

The input is a mono waveform as a 1-D float `numpy` array sampled at
`whisper.feature_extractor.sampling_rate`, or `{"raw": array, "sampling_rate":
rate}`. A float `torch.Tensor` is taken as log-mel features that were already
extracted, not as audio.

```python
import numpy as np
import torch
import nnsight
from nnsight import TransformersModel

whisper = TransformersModel("openai/whisper-tiny", task="automatic-speech-recognition", dispatch=True)
encoder = whisper.model.encoder                    # 4 layers on whisper-tiny; whisper.model.decoder beside it

rate = whisper.feature_extractor.sampling_rate     # 16000
clip = (0.3 * np.sin(2 * np.pi * 440 * np.arange(3 * rate) / rate)).astype(np.float32)   # 3 s tone in place of speech

settings = dict(language="en", max_new_tokens=12)  # language= makes the encoder run once

with whisper.generate(clip, **settings) as tracer:
    hidden = encoder.layers[2].output.save()       # an encoder layer's hidden state
    ids = tracer.result.save()                     # the transcript's token ids

with whisper.generate(clip, **settings) as tracer:
    encoder.layers[3].output[:] = 0                # zero one encoder layer
    ablated_ids = tracer.result.save()

text = whisper.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
ablated = whisper.tokenizer.batch_decode(ablated_ids, skip_special_tokens=True)[0]
print(tuple(hidden.shape), repr(text), "->", repr(ablated))

assert tuple(hidden.shape) == (1, whisper.feature_extractor.nb_max_frames // 2, 384)
assert text != ablated
```

```
(1, 1500, 384) ' You' -> ' abouts'
```

A tone has no words in it, so the transcript is a stock word. The edit changing
it is the point. The encoder always returns 1500 positions, because the feature
extractor pads or cuts every window to 30 s and one position covers 20 ms: the
3 s clip is `hidden[0, :150]` and the rest encodes padding.

| Run | What runs | `tracer.result` |
|---|---|---|
| `whisper.generate(clip, ...)` | the model's greedy decoding | transcript token ids, `(rows, new_tokens)` |
| `whisper.pipe(clip, ...)` | the ASR pipeline, which decodes with 5 beams | `{"text": ...}` |
| `whisper.trace(clip, decoder_input_ids=ids)` | one encoder pass and one decoder pass | the forward's output; read `whisper.output.logits` |

`pipe` and `generate` decode differently (on this clip `pipe` returns
`' Thank you very much.'`), so compare a clean and an edited run under the same
method.

**Without `language=` the encoder runs twice**: once to detect the language,
then to transcribe. Both passes compute the same hidden states, so a read needs
no care, but an edit written without a loop lands on the detection pass only.
Zeroing layer 3 that way leaves the transcript unchanged, and zeroing layer 0
changes the detected language (the transcript comes back in Japanese). Pass
`language=`, or address the second pass:

```python
with whisper.generate(clip, max_new_tokens=12) as tracer:     # no language=
    for step in tracer.iter[1]:                               # pass 0 detects, pass 1 transcribes
        encoder.layers[3].output[:] = 0
    second_pass = tracer.result.save()

assert whisper.tokenizer.batch_decode(second_pass, skip_special_tokens=True)[0] == ablated
```

The decoder's first call belongs to the detection pass too, so with no
`language=` decoding step *k* of a decoder module is `tracer.iter[k + 1]`.

**One forward with logits and gradients** is `trace`, which needs the decoder's
input. Without `decoder_input_ids=` it raises `ValueError: You cannot specify
both decoder_input_ids and decoder_inputs_embeds at the same time`.

```python
whisper.tokenizer.set_prefix_tokens(language="en", task="transcribe")
start = torch.tensor([whisper.tokenizer.prefix_tokens])   # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>

with whisper.trace(clip, decoder_input_ids=start):
    clean = whisper.output.logits[0, -1].detach().save()  # logits for the first transcript token

with whisper.trace(clip, decoder_input_ids=start):
    encoder.layers[3].output[:] = 0
    edited = whisper.output.logits[0, -1].detach().save()

first = ids[0, 0]                                          # the token generate emitted first
print(f"{float(clean.softmax(-1)[first]):.3f} -> {float(edited.softmax(-1)[first]):.3f}")
assert clean.softmax(-1)[first] > 10 * edited.softmax(-1)[first]
```

```
0.017 -> 0.000
```

These are raw logits. `generate` suppresses a list of tokens before it picks, so
the argmax here (`' ('`) is not the token it emits (`' You'`).

**A clip longer than 30 s** needs `chunk_length_s=30`; without it `generate`
and `pipe` raise `ValueError: You have passed more than 3000 mel input
features`. Under `generate` and `trace` the windows are rows of one forward: a
70 s clip is `(3, 1500, 384)` at the encoder, and `trace` takes one row of
`decoder_input_ids` per window. Under `pipe` each window is a forward of its
own, so a block without a loop sees only the first; with `language=` set,
window *k* is `tracer.iter[k]`.

## A vision encoder

Image tasks need Pillow installed. The image goes in as a `PIL.Image`, and the
root envoy is the bare encoder:

```python
from PIL import Image

image = Image.new("RGB", (224, 224), (120, 30, 200))
vision = TransformersModel("google/vit-base-patch16-224",
                           task="image-feature-extraction", dispatch=True)

with vision.trace(image):
    patches = vision.layers[6].output.save()

assert tuple(patches.shape) == (1, 197, 768)       # 196 patches and the CLS token
assert vision.tokenizer is None
```

## Classifiers and chunked tasks

An encoder classifier pads a batch on the **right**, so position 0 (`[CLS]`) is
the same token on every row and `[:, -1]` is padding on every row but the
longest.

Some tasks split one input into several encodings: token windows past the
model's length limit, one entailment pair per candidate label, the windows of a
long recording. Those become **rows of the trace's single forward**, in the
order the task yields them. Count them the way the task does, not from the text
length:

```python
ner = TransformersModel("hf-internal-testing/tiny-random-BertForTokenClassification",
                        task="token-classification", dispatch=True)
long_text = " ".join(["John lives in Paris"] * 200)

with ner.trace("John lives in Paris"):
    short = ner.output.logits.save()

with ner.trace(long_text, stride=16):
    hidden = ner.bert.encoder.layer[0].output.save()
    windows = ner.output.logits.save()

with ner.trace(long_text):                        # no stride=
    cut = ner.output.logits.save()

chunks = ner.tokenizer(long_text, truncation=True, return_overflowing_tokens=True,
                       stride=16)["input_ids"]    # the windows the task makes

assert ner.tokenizer.padding_side == "right"
assert short.shape[0] == 1
assert windows.shape[0] == hidden.shape[0] == len(chunks) == 7     # one row per window
assert tuple(cut.shape) == (1, 512, 2)            # truncated to one window, nothing raised
```

```python
zero_shot = TransformersModel("hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
                              task="zero-shot-classification", dispatch=True)
labels = ["travel", "cooking", "dancing"]

with zero_shot.trace("one day I will see the world", candidate_labels=labels):
    per_label = zero_shot.output.logits.save()

assert per_label.shape[0] == 3                    # one entailment row per candidate label
print(zero_shot.pipe("one day I will see the world", candidate_labels=labels)["labels"])
```

- **A chunked invoke is the whole batch.** Its row count belongs to the task, so
  a second invoke beside it raises `NotImplementedError: task='zero-shot-classification'
  splits this invoke into 3 forward rows ...`. Give each chunked input its own trace.
- `model.pipe(...)` returns the task's records: `{"sequence", "labels", "scores"}`
  here, `[{"label", "score"}]` for `text-classification`, one dict per token for
  `token-classification`.
- `document-question-answering` and `zero-shot-object-detection` take the task's
  own dict (`{"image": ..., "question": ...}`), which goes through the task's
  preprocessing in place of being read as model inputs.
- `mask-generation` and `keypoint-matching` cannot be traced from an image. The
  `NotImplementedError` names the two ways round: `model.pipe(...)`, or a trace
  on an encoding you build with `model.image_processor`.
