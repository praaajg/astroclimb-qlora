# Architecture

## Inference pipeline

```mermaid
flowchart LR
    A["Object A<br/>figure or caption"] --> P["Prompt builder<br/>object blocks + answer instruction"]
    B["Object B<br/>figure or caption"] --> P
    P --> T["Processor chat template"]
    T --> M["Gemma-4-E4B-it<br/>4-bit NF4 + LoRA adapter"]
    M --> G["Greedy decoding<br/>max 32 new tokens"]
    G --> R["Case-insensitive label parser"]
    R --> L["same_figure<br/>same_paper<br/>related_papers<br/>unrelated_papers"]
```

The prompt is identical for training, validation, and test. Images are resized
to 448×448, captions are truncated to 2,000 characters, and no DOI or paper
metadata is provided at any point.

## Training pipeline

```mermaid
flowchart TD
    D["10,000 labelled competition pairs"] --> S["Seed-42 index split"]
    S --> TR["8,200 training rows"]
    S --> VA["1,800 validation rows<br/>450 per class"]
    TR --> Q["QLoRA<br/>r=64, alpha=128, dropout=0.05"]
    Q --> E["Evaluate every 200 steps<br/>early stopping patience 3"]
    E --> C["adapter_best<br/>step 1,000 / val loss 0.3464"]
    C --> I["Official 10,000-row test inference"]
    I --> O["Macro F1 0.64282"]
```

## Module map

| Module | Responsibility |
|---|---|
| `astroclimb.prompt` | Verbatim system prompt, object blocks, caption truncation. |
| `astroclimb.labels` | Four-label vocabulary, regex parsing, fallback. |
| `astroclimb.data` | Base64/path image decoding, 448×448 resize, type detection, CSV validation. |
| `astroclimb.split` | Balanced 450-per-class validation split with seed 42. |
| `astroclimb.model` | 4-bit NF4 loading, LoRA config, adapter attachment. |
| `astroclimb.train` | Reference `transformers.Trainer` pipeline and collator. |
| `astroclimb.infer` | Batched inference and one-hot submission writer. |
| `astroclimb.evaluate` | Macro F1, accuracy, per-class precision/recall/F1, confusion matrix. |

## Design notes

- **One prompt for all pair types.** `same_figure` is only possible for
  figure–caption pairs, but the prompt still instructs the model to judge from
  alignment instead of assuming the class is impossible. This matches the
  paper's single-prompt contract.
- **Macro F1 is the selection metric.** The competition metric weights all
  four classes equally, which is why `related_papers` matters far more than its
  10% frequency suggests.
- **Asymmetric errors.** The adapter is conservative about `same_figure`
  (validation precision 0.9945, recall 0.4000). Most missed figures are labelled
  `same_paper`. Any future calibration work should target this boundary first.
- **Why figure–figure pairs are hard.** `related_papers` recall is only 0.0867
  on validation figure–figure pairs; 104 of 150 were predicted as
  `same_paper`. Visual style similarity is an unusually strong distractor for
  citation inference.
- **No metadata shortcut.** Test pairs carry no DOI or paper identifiers, so
  the model cannot learn a citation-graph shortcut from document metadata.
