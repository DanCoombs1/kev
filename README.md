# kev

**kev is a small vision model that answers typed questions about X-ray baggage scans, and gives a calibrated
probability for every answer, in one pass.** It reads the scan's pixels directly (no OCR, no object detector bolted
on) and was built from scratch on a laptop: tokenizer, text encoder, image encoder, pretraining, question decoder and
training data pipeline.

```
$ uv run python scripts/ask.py --image data/pidray/easy/xray_easy00000.png --like pidray
> all
  baton 100%   power bank 0%   knife 0%   lighter 0%   spray can 0%   pliers 0%   scissors 0%   gun 0% ...   (27 ms)
> is there a knife in this bag?
  yes 1%   no 99%   (57 ms)
```

It's a proof of concept, not a screening product. See [Results](#results) and [Limitations](#limitations) for how
well it actually works.

## Where the idea came from

Jev, from TypeSafe AI, is a "System One" model. You give it a situation and a list of typed
questions, and in one forward pass it returns a probability for every option, without generating any text. That makes
it fast and gives it honest, calibrated confidence. Jev works on text.

kev takes the same idea into vision. The situation is an X-ray scan of a bag, and the questions are the ones a
screener would ask: *is there a knife in this bag?*, *which of these are in it?*, *any weapon or explosive?* The
answer comes straight from the image, in tens of milliseconds, on a laptop. I built it to understand how this kind of
model works from the inside, so nothing is borrowed from existing replicas and no pretrained weights are used.

## How it works

```
scan ─→ 16×16 patches ─→ image encoder (9 blocks, MAE-pretrained) ─→ one vector per patch ──┐
                                                                                            │
question ─→ tokenizer ─→ text encoder (frozen) ─→ question reader ─┐                        │
                                                                   question decoder (3 blocks):
                                                                   questions look at each other,
                                                                   then at the scan's patches
                                                                                 │
each option ─→ tokenizer ─→ text encoder (frozen) ─→ option reader ──→ answer · option ─→ probabilities
```

**Question types.** Every question has a type, as in Jev:

| Type | Example | Output |
|---|---|---|
| yes/no | *is this bag free of knives?* | one of yes / no |
| one of | *which of these is in the bag?* knife, gun, scissors | one option, or "none of these" |
| any of | *select everything that is in this bag* | an independent probability per option |

Options are text, not fixed output slots, so the same model can be asked about different lists of items without
changing its architecture.

**Parts.**

| Part | What it is | Parameters |
|---|---|---|
| Tokenizer | byte-level BPE trained from scratch, 8,261 tokens | – |
| Text encoder | 4-block transformer, pretrained by filling in hidden words, then frozen | 10.3M |
| Image encoder | 9-block Vision Transformer over 16×16 patches with row, column and view positions; handles one or two views of a bag | 16.7M |
| Readers | small attention layers that turn a question's or option's words into one vector | 1.2M |
| Question decoder | 3 blocks with self-attention between questions and cross-attention to the scan | 7.1M |
| **Total** | | **35.4M** |

**Training, in four stages.**

1. **Data.** Four public X-ray datasets (STCray, PIDray, DvXray, IEDXray), about 128,000 scans after removing
   near-duplicates. Items from every dataset are mapped to one vocabulary of 35 items, each in a group: weapon or
   explosive, restricted, or allowed. Scans are cropped to the bag and rescaled so an item is the same size in pixels
   whatever the scanner. Splits are made by group (same bag, same packing) so near-copies never straddle train and
   test. I checked for shortcuts too: a tiny probe could tell clean bags from threat bags by their outline alone, so
   "is this bag clean?" is only trained on the *same* bag with and without a threat pasted in. The pasting uses the
   physics of X-ray transmission: the bag's image is multiplied by the threat's.
2. **Text pretraining.** The text encoder learns English by filling in hidden words across 269M tokens of Wikipedia,
   Simple English Wikipedia and SQuAD questions (80 minutes).
3. **Image pretraining.** The image encoder learns what scans look like as a masked autoencoder: 75% of each scan's
   patches are hidden and it learns to redraw them (38 passes over 74,000 scans, about 10.5 hours, with early
   stopping).
4. **Question training.** Questions are generated for every scan on the fly, at real-life answer rates. The items to
   ask about are chosen without looking at the bag, so most answers are "no", just like real bags. The generator knows
   that a 3D-printed gun *is* a gun, and skips questions whose answer is genuinely ambiguous (is a box cutter a
   knife?). About 8 hours for 10 passes.

**Calibration** is measured with reliability diagrams and expected calibration error (ECE), and can be tightened with
temperature scaling.

## Practical uses

A model like this fits anywhere a person looks at an image and makes a quick yes/no or which-one decision:

- **Airport security screening.** An assistant for the operator. Every bag gets a probability for each prohibited item
  in tens of milliseconds, the uncertain ones are flagged for a closer look or a hand search, and the operator stays
  in charge.
- **Parcel, mail and cargo screening**, where volumes are too high for a person to look closely at everything.
- **Venue and event security**, with smaller, cheaper scanners at the door.
- **Quality control and training.** Checking operators' decisions after the fact, or generating practice cases.

Because the questions and options are text, a change of policy ("now also flag lithium batteries over 100 Wh") is a
new question, not a new model, as long as the model has learned what the item looks like.

## Results

Measured on 1,024 validation scans (same datasets as training) and 1,024 test scans the model never saw. "Empty-bag
guess" is the accuracy of always answering as if the bag were empty. Because most real answers are "no", it's the bar
to beat.

| | Validation | Test |
|---|---|---|
| Accuracy | 96.1% | 78.9% |
| Empty-bag guess | 71.1% | 70.6% |
| Items in the bag that kev reports | 91.8% | 55.0% |
| False alarms | 0.7% | 4.8% |

| Test set | Items reported | Notes |
|---|---|---|
| DvXray | 76.5% | holds up: its test scans come from the same kind of split as validation |
| PIDray | 74.0% | includes deliberately hidden and cluttered items |
| IEDXray | 72.4% | |
| STCray | 20.6% | its test set uses physical objects never seen in training; kev mostly fails here |

Calibration on held-back validation scans: ECE 0.006 for "is this item here?" and 0.009 for one-answer questions,
0.003 and 0.002 after temperature scaling.

## Limitations

- **A small sample.** About 128,000 scans from four public datasets, with only a handful of physical objects for many
  items. kev learned the particular objects it saw rather than what a knife or a pair of pliers looks like in general:
  it reports 92% of items on familiar data but only 21% on STCray's unseen objects. It calls a pair of pliers
  scissors, and a bag holding an explosive and a battery "knife, 99%".
- **A laptop.** Everything was trained on a MacBook Pro (M4 Pro, 48 GB), which capped the model at 35M parameters,
  batches of 8–16 scans and roughly a day of total training time. Production vision models are far larger and train
  on far more data.
- **Validation was too kind.** The validation split shared physical objects with training, so early stopping and model
  choice rewarded memorising rather than generalising. The next step is a validation split that holds out whole
  objects, like the real test.
- **Wording.** The text encoder is frozen to protect its knowledge of English, but kev still keys on the exact item
  names it trained with. Held-out synonyms that share a word with a training name work ("rigged pager", "spray
  canister"). Most that don't, fail ("spanner", "beeper", "sidearm"), and a new negation pattern ("no knives in here,
  right?") gets answered backwards.
- **Scanners.** Images from other scanners, such as screenshots from the internet, have different colours and scales
  and give unreliable answers.
- **Not for real screening decisions.** Nothing here is certified, and two of the training datasets are licensed for
  academic use only.

## Why a small, local model is still the right shape for the job

The limits above are about the data and the compute, not the approach. The approach has real advantages for a
business like airport security:

- **Speed without a network.** One pass, 20–60 ms per scan on a laptop. A hosted model adds network latency to every
  bag. A local one doesn't, and keeps working if the connection drops.
- **Data never leaves the building.** Scans of passengers' belongings are sensitive. A model this size runs on the
  screening machine or a local server, air-gapped if needed.
- **Cheap to train and retrain.** The whole pipeline ran on one laptop in about a day of compute. An airport or
  scanner maker already holds millions of labelled scans from its own machines, which is exactly what kev lacked, and
  could train a far stronger version of this without a data centre.
- **Honest confidence.** Calibrated probabilities let a security team choose its own trade-off, such as "hand-search
  anything above 5% for a weapon", and know what that costs in false alarms.
- **Built on what airports already do.** Threat image projection (pasting fictional threats into real bags to keep
  operators alert) is already standard at checkpoints, and it's the same technique kev uses to learn clean-versus-threat
  from the same bag.
- **Understandable end to end.** Every stage is small enough to inspect, test and explain, which matters when the
  decision is a security one.

## Project layout

```
src/kev/data/        datasets, naming, duplicates, splits, scan preparation, threat insertion, loading
src/kev/tokenizer/   byte-level BPE
src/kev/questions/   question wording, and the generator that asks questions and knows the answers
src/kev/model/       transformer blocks, text encoder, image encoder, the full model
src/kev/pretrain/    masked-word text pretraining, masked-autoencoder image pretraining
src/kev/train/       question training
scripts/             downloads, checks, calibration, and ask.py for trying it out
tests/               unit tests (uv run pytest)
```

## Running it

Requires [uv](https://docs.astral.sh/uv/) and about 45 GB of disk. Datasets and trained weights are not included.

```
uv sync
scripts/download_xray_data.sh        # DvXray, COMPASS-XP, PIDray
scripts/download_stcray.sh           # needs a free Hugging Face account and accepting the dataset's terms
                                     # IEDXray: download Figshare file 60096533 and unzip it into data/iedxray/
scripts/download_text.sh             # WikiText-103, SQuAD, Wikipedia

uv run python -m kev.data.manifest   # every scan in one format
uv run python -m kev.data.duplicates
uv run python -m kev.data.curate     # merge duplicates, assign train/val/test
uv run python -m kev.data.loader     # threat library for insertion, patch counts
uv run python -m kev.tokenizer.bpe
uv run python -m kev.pretrain.text
uv run python -m kev.pretrain.image
uv run python -m kev.train.questions

uv run python scripts/ask.py                     # ask about test scans it has never seen
uv run python scripts/ask.py --image scan.png    # or your own X-ray image
```

## Data

| Dataset | Licence |
|---|---|
| [STCray](https://huggingface.co/datasets/Naoufel555/STCray-Dataset) | CC BY 4.0 |
| IEDXray (Khalifa University, Figshare) | CC BY 4.0 |
| [PIDray](https://github.com/lutao2021/PIDray) | academic use only |
| [DvXray](https://github.com/Mbwslib/DvXray) | academic use only |
| [COMPASS-XP](https://zenodo.org/records/2654887) | CC BY 4.0 (downloaded, not yet used) |
| WikiText-103, Wikipedia, Simple English Wikipedia | CC BY-SA 3.0 |
| SQuAD | CC BY-SA 4.0 |

## Licence

All rights reserved. See [LICENSE](LICENSE).
