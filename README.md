# AstroCLIMB QLoRA

QLoRA adaptation of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) for citation-relation classification between astrophysics figures and captions.

**Paper:** [PSGS@2026](https://aclanthology.org/2026.wasp-1.21/) · [PDF](paper/2026.wasp-1.21.pdf)  
**Adapter:** [ShivRamSaud/gemma4-e4b-qlora-astroclimb-v2-fresh](https://huggingface.co/ShivRamSaud/gemma4-e4b-qlora-astroclimb-v2-fresh)

## Results

Balanced 1,800-row validation split (450 per class, seed 42):

| System | Macro F1 | Accuracy |
| --- | ---: | ---: |
| Zero-shot Gemma-4-E4B | 0.4408 | 0.5206 |
| QLoRA `adapter_best` | 0.5347 | 0.5483 |

Per-class validation F1:

| Class | Zero-shot | QLoRA |
| --- | ---: | ---: |
| `same_figure` | 0.7760 | 0.5705 |
| `same_paper` | 0.4108 | 0.5311 |
| `related_papers` | 0.0172 | 0.3508 |
| `unrelated_papers` | 0.5593 | 0.6864 |

The official 10,000-row test set scores **0.64282 macro F1**. Predicted class counts are 684 `same_figure`, 3,005 `same_paper`, 2,058 `related_papers`, and 4,253 `unrelated_papers`. The released adapter is the best validation-loss checkpoint (0.3464 at step 1,000). Exact metrics are in [`results/`](results).

## Setup

```bash
pip install -r requirements.txt
```

Public models and datasets need no login. For gated or private assets, run `huggingface-cli login` or set `HF_TOKEN`.

## Data

The default source is the public `ShivRamSaud/astroclimb_train` dataset. Pass a local CSV with `--train-csv path/to/train.csv`.

The validation split samples 450 rows per class with seed 42. `same_figure` draws only TXT-IMG pairs; the other classes draw 150 rows each from TXT-IMG, IMG-IMG, and TXT-TXT. Training uses the remaining 8,200 rows.

## Training

```bash
python -m astroclimb.train \
  --train-csv ShivRamSaud/astroclimb_train \
  --output-dir outputs/astroclimb
```

The submitted run used an A100-80GB: QLoRA rank 64, alpha 128, dropout 0.05, 4-bit NF4 with double quantization, BF16, sequence length 1536, 2,560 steps, effective batch size 16, learning rate 1e-4, cosine schedule, and early stopping with patience 3 at 200-step evaluation intervals. Batch size is selected from available VRAM. Add `--hub-repo-id <user>/<repo>` to upload checkpoints.

## Evaluation

```bash
python -m astroclimb.evaluate \
  --adapter ShivRamSaud/gemma4-e4b-qlora-astroclimb-v2-fresh \
  --adapter-subfolder adapter_best \
  --output-dir outputs/astroclimb_eval
```

The script rebuilds the balanced 1,800-row validation split, runs greedy generation, and writes predictions and metrics.

## Test submission

```bash
python -m astroclimb.predict \
  --test-csv test.csv \
  --adapter ShivRamSaud/gemma4-e4b-qlora-astroclimb-v2-fresh \
  --adapter-subfolder adapter_best \
  --output submission.csv
```

The output is a one-hot CSV with `id` and the four label columns in paper order. The original test script was not released. By default the command uses the released inference prompt; pass `--prompt train` for the system-first training contract described in Appendix A.1.

## Model

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "ShivRamSaud/gemma4-e4b-qlora-astroclimb-v2-fresh",
    allow_patterns="adapter_best/*",
)
```

## Layout

```text
astroclimb/
  data.py       loading, labels, balanced split, examples
  image.py      base64 image decoding
  labels.py     label order and parsing
  model.py      processor, 4-bit base model, LoRA, adapter
  prompts.py    training and inference message builders
  train.py      QLoRA training and checkpointing
  evaluate.py   balanced 1,800-row evaluation
  predict.py    test submission generation
model/          released adapter configuration
results/        paper metrics and released test submission
paper/          WASP 2026 paper and BibTeX
```

## Citation

```bibtex
@inproceedings{ghimire-etal-2026-psgs,
    title = "{PSGS}@2026: From Figures to Citation Graphs: {QL}o{RA} Based Multimodal Linking for {A}stro{CLIMB}",
    author = "Ghimire, Prajwal  and
      Saud, Shiv Ram  and
      Mahato, Aashish  and
      Dawadi, Sundeep  and
      Saud, Ramesh  and
      Regmi, Sunil  and
      Shakya, Manoj",
    editor = "Accomazzi, Alberto  and
      Ghosal, Tirthankar  and
      Grezes, Felix  and
      Lockhart, Kelly  and
      Alkan, Atilla Kaan",
    booktitle = "Proceedings of the 4th Workshop on Artificial Intelligence for Scientific Publications",
    month = nov,
    year = "2026",
    address = "Hengqin, Zhuhai, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.wasp-1.21/",
    pages = "192--198",
    ISBN = "979-8-89176-456-9"
}
```

The modules in `astroclimb/` reproduce the submitted training and evaluation notebooks. Both prompt builders are preserved: `build_train_messages` follows the system-first training prompt, and `build_inference_messages` follows the released inference prompt. The hardcoded token in the evaluation notebook was removed.
