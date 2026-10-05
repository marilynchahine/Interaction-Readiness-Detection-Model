# Interaction Readiness Detection with Vision-Language Models

This repository contains the models, inference pipelines, fine-tuning
code, and evaluation framework developed for **interaction readiness
detection in Human-Robot Interaction (HRI)**.

The project investigates whether Vision-Language Models (VLMs) can
jointly **localize people** and determine whether each person is
currently available to interact with a robot. It compares frame-based
and temporal formulations, zero-shot inference and LoRA adaptation, and
several strategies for incorporating information from preceding frames.

> **M2 Machine Learning Internship Project --- Sorbonne Université /
> ISIR**\
> **Marilyn N. Chahine**, Hamed Rahimi, Mohamed Chetouani

[Dataset on Hugging
Face](https://huggingface.co/datasets/marilynchahine1/interaction-readiness-detection-dataset-frame-path-version)

------------------------------------------------------------------------

## Overview

For every relevant person in a scene, the model predicts:

-   a **person ID**;
-   a **normalized bounding box**;
-   one of three interaction-readiness states:
    -   `not_interaction_ready`
    -   `interaction_ready`
    -   `interaction_ongoing`

The task is formulated as structured multimodal generation. Instead of
explicitly extracting individual social cues such as gaze, body
orientation, or distance and feeding them to a separate classifier, the
VLM receives the visual scene directly and jointly performs person
localization and readiness classification.

A typical prediction has the form:

``` json
[
  {
    "person_ID": 1,
    "bbox": [x1, y1, x2, y2],
    "label": "interaction_ready"
  },
  {
    "person_ID": 2,
    "bbox": [x1, y1, x2, y2],
    "label": "not_interaction_ready"
  }
]
```

Bounding-box coordinates are normalized to the range expected by the
inference pipeline. Predictions are limited to the most relevant visible
people, with an empty list returned when no person should be predicted.

------------------------------------------------------------------------

## Motivation

A socially interactive robot should not only recognize *who* is present,
but also *when* approaching or initiating an interaction is appropriate.

Interaction readiness is inherently multimodal and context dependent.
Relevant cues can include:

-   whether a person is approaching or leaving;
-   body and head orientation;
-   attention toward the robot;
-   distance and position;
-   ongoing engagement with the robot;
-   temporal changes in these cues.

This project explores whether modern VLMs can reason over these signals
directly from images and video, while producing both spatial and
semantic predictions in a single model.

------------------------------------------------------------------------

## Model Formulations

### 1. Frame-Based Interaction Readiness

The frame-based formulation receives only the **current target frame**.
The VLM must identify relevant people, localize them, and assign an
interaction-readiness label using only instantaneous visual evidence.

```{=html}
<p align="center">
```
`<img src="figures/frame_based_architecture.png" alt="Frame-based interaction readiness architecture" width="850">`{=html}
```{=html}
</p>
```
This formulation is used as the main baseline for studying:

-   zero-shot VLM capabilities;
-   the effect of model scale;
-   the effect of task-specific LoRA fine-tuning.

### 2. Video-Based Interaction Readiness

The temporal formulation uses the **15 frames preceding the target
frame** as visual context.

```{=html}
<p align="center">
```
`<img src="figures/video_based_architecture.png" alt="Video-based interaction readiness architecture" width="850">`{=html}
```{=html}
</p>
```
Several temporal strategies are investigated.

#### Prediction-conditioned rollout

For target frame (t), the model receives frames
(t-15,`\ldots`{=tex},t-1), together with the available person-level
predictions associated with those frames.

Inference starts at frame 16:

``` text
predict frame 16 <- frames 1 ... 15
predict frame 17 <- frames 2 ... 16 + prediction for frame 16
predict frame 18 <- frames 3 ... 17 + predictions for frames 16-17
...
predict frame 31 <- frames 16 ... 30 + predictions for all 15 context frames
```

The formulation is intended to provide both temporal visual evidence and
explicit person continuity.

#### Temporal visual context without rollout

The model receives the same 15 preceding frames but **no previous
bounding boxes, labels, or person predictions**. This isolates the
contribution of visual temporal context from the contribution of
prediction feedback.

#### Oracle rollout

Ground-truth annotations from preceding frames are supplied at inference
time. This is **not a deployable setting**; it is used as a diagnostic
upper bound to test whether the model can exploit accurate temporal
person information.

------------------------------------------------------------------------

## Models

The main experiments use **Qwen3-VL** at multiple parameter scales:

-   Qwen3-VL 2B
-   Qwen3-VL 4B
-   Qwen3-VL 8B

The repository supports experiments in:

-   **zero-shot inference**, using the pretrained VLM without
    task-specific adaptation;
-   **LoRA fine-tuning**, for parameter-efficient adaptation to
    interaction-readiness detection;
-   **frame-based inference**;
-   **video-based inference**;
-   **prediction-conditioned rollout**.

Additional VLM families were also explored during benchmarking,
including **InternVL3.5** and **LLaVA-OneVision** variants.

------------------------------------------------------------------------

## Dataset

The aligned interaction-readiness dataset used in this project is
hosted separately on Hugging Face:

**[Interaction Readiness Detection
Dataset](https://huggingface.co/datasets/marilynchahine1/interaction-readiness-detection-dataset-frame-path-version)**

It combines and aligns annotations from:

-   **AVDIAR**
-   **JRDB**
-   **SSUP-HRI**

into a common three-class interaction-readiness taxonomy.

The released dataset contains both:

-   a **frame-based representation**, for independent-image experiments;
-   a **temporal representation**, preserving continuous sequences for
    video-based experiments.

The dataset is split at the **video level** to avoid leakage of
near-duplicate frames between train, validation, and test sets. The
final split is approximately **90 / 5 / 5**.

### Labels

  -----------------------------------------------------------------------
  Label                               Meaning
  ----------------------------------- -----------------------------------
  `not_interaction_ready`             The person is not currently
                                      available or oriented toward
                                      interaction with the robot.

  `interaction_ready`                 The person displays cues indicating
                                      availability or intention to
                                      interact.

  `interaction_ongoing`               The person is currently engaged in
                                      an interaction with the robot.
  -----------------------------------------------------------------------

See the Hugging Face dataset card for the full annotation alignment
procedure, file formats, split construction, and dataset statistics.

------------------------------------------------------------------------

## Using the Dataset

Access the dataset through Hugging Face:

``` bash
git clone https://huggingface.co/datasets/marilynchahine1/interaction-readiness-detection-dataset-frame-path-version
```

The dataset is gated. You may need to log in to Hugging Face and accept
its access conditions before downloading the files.

The repository includes separate frame-based and temporal annotation
formats. Refer to the dataset's own README for the authoritative
directory structure and schema.

------------------------------------------------------------------------

## Evaluation

The task contains two coupled components: **person localization** and
**interaction-readiness classification**.

### Localization

Predicted and ground-truth boxes are matched using IoU-based assignment.
Localization is evaluated with:

-   Precision
-   Recall
-   F1-score

### Classification

Classification metrics are computed over accepted person matches:

-   Accuracy
-   per-class Precision / Recall / F1
-   **Macro F1**
-   Weighted F1
-   Confusion matrix

**Macro F1 is treated as the primary classification metric**, because it
gives equal weight to the three readiness classes and is therefore less
dominated by the more frequent classes.

------------------------------------------------------------------------

## Fine-Tuning

LoRA is used for parameter-efficient task adaptation.

The experiments are designed to compare pretrained VLM behavior with
models adapted to the aligned interaction-readiness annotations while
avoiding full-model fine-tuning.

Conceptually:

``` text
Pretrained VLM
     |
     +-- Zero-shot evaluation
     |
     +-- LoRA adaptation
             |
             +-- Frame-based model
             |
             +-- Temporal model
                    |
                    +-- prediction-conditioned rollout
                    +-- visual context only
```

For rollout fine-tuning, teacher forcing supplies ground-truth
annotations for context frames while the loss is computed on the current
target prediction.

------------------------------------------------------------------------

## Output Format

The model is prompted to return machine-readable person-level
predictions. Each prediction contains:

``` json
{
  "person_ID": 1,
  "bbox": [x1, y1, x2, y2],
  "label": "interaction_ready"
}
```

The inference and evaluation pipelines parse this structured output
before matching predictions to ground truth.

------------------------------------------------------------------------

## Main Results

### Frame-Based Qwen3-VL

| Model | Localization Precision | Localization Recall | Localization F1 | Classification Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL 2B — Zero Shot | 27% | 92% | 41% | 47% | 23% | 32% |
| Qwen3-VL 4B — Zero Shot | 34% | 93% | 50% | 53% | 33% | 45% |
| Qwen3-VL 8B — Zero Shot | 21% | 55% | 31% | 54% | 30% | 43% |
| **Qwen3-VL 4B — Fine-Tuned** | **83%** | **82%** | **83%** | **51%** | **46%** | **51%** |
| Qwen3-VL 8B — Fine-Tuned | 80% | 81% | 81% | 48% | 44% | 48% |

Task-specific LoRA adaptation substantially improves localization. The
strongest frame-based result is obtained by the **Qwen3-VL 4B fine-tuned
model**, reaching **83% localization F1** and **46% Macro F1**.

### Video-Based and Diagnostic Pipelines

All models in this table use **Qwen3-VL 2B**.

| Setting | Localization Precision | Localization Recall | Localization F1 | Classification Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Upper-Bound Rollout — Zero Shot | 87% | 88% | 88% | 80% | 76% | 81% |
| Rollout — Zero Shot | 36% | 50% | 42% | 77% | 32% | 69% |
| Rollout — Fine-Tuned | 6% | 14% | 9% | — | — | — |
| No Rollout — Zero Shot | 23% | 68% | 34% | 46% | 31% | 45% |
| **No Rollout — Fine-Tuned, No-Rollout Inference** | **99%** | **81%** | **89%** | **72%** | **65%** | **63%** |
| No Rollout — Fine-Tuned, Rollout Inference | 87% | 69% | 77% | 60% | 30% | 46% |

Classification values are not reported for conditions with localization F1
below 10%, because the resulting classification subset is too small to support
a reliable interpretation.

The strongest learned temporal formulation is **No Rollout — Fine-Tuned,
No-Rollout Inference**, which reaches **89% localization F1** and **65% Macro
F1**. The upper-bound rollout reaches **88% localization F1** and **76% Macro
F1**, showing how strongly the model can benefit when accurate previous-frame
annotations are available.

## Key Findings

1.  **Task-specific adaptation is highly effective.**\
    LoRA substantially improves the frame-based models & the video-based model with no rollout, especially
    person localization.

2.  **More parameters do not automatically improve zero-shot
    performance.**\
    Qwen3-VL 4B outperforms the 8B model in the frame-based zero-shot
    setting.

3.  **Temporal visual information is useful without explicit
    prediction feedback.**\
    Fine-tuning with the preceding visual frames alone produces the
    strongest learned temporal result, rollout prediction remains to be fixed (5).

4.  **Accurate previous-frame information is highly effective.**\
    Upper-bound rollout reaches 88% localization F1 and 76% Macro F1, showing
    that the model can exploit correct temporal person information.

5.  **Prediction-conditioned rollout introduces error-propagation and
    shortcut-learning risks.**\
    After teacher-forced rollout fine-tuning, the model tends to
    reproduce similar bounding boxes across consecutive frames and
    localization performance collapses.

These experiments suggest that temporal context is valuable, but that
**how temporal information is represented and propagated is at least as
important as providing temporal context itself**.

------------------------------------------------------------------------

## Rollout Failure Potential Analysis

The rollout experiments revealed an important failure mode.

During training, ground-truth annotations from preceding frames were
provided as context through teacher forcing. Because people often occupy
similar positions in consecutive frames, the model could learn a
shortcut: reuse recent bounding boxes instead of re-localizing people
from the current visual evidence.

At inference time, ground truth is replaced by the model's own
predictions. Small errors can therefore be propagated and reinforced
over subsequent frames.

Three observations support this interpretation:

-   **fine-tuned self-rollout collapses**, with repeated or nearly
    repeated boxes across frames;
-   **upper-bound rollout performs strongly** when the propagated annotations
    are correct, which shows the pipeline itself is functional;
-   **fine-tuning on temporal visual context without predictions
    performs strongly**, showing that temporal modeling itself is not
    the main limitation & isolating the effects of teacher forcing.

This distinction is central to the project: temporal context helps, but
recursively feeding structured predictions (at least in the format used in the scope of this project) back into a generative VLM
can lead the model to learn faulty shortcuts.

**This is one possible interpretation of the failure we have seen in this project, other possible reasons may also contribute to the failure and need to be looked into in more detail.**

------------------------------------------------------------------------

## Reproducing Experiments

The experimental workflow is:

``` text
1. Download / request access to the aligned dataset
2. Configure local frame paths
3. Select a VLM and parameter scale
4. Run zero-shot inference or LoRA fine-tuning
5. Generate structured person-level predictions
6. Evaluate localization
7. Match accepted detections to ground truth
8. Evaluate interaction-readiness classification
```

Because the project contains multiple experimental formulations, use the
configuration and scripts corresponding to the desired setting rather
than mixing frame-based, no-rollout, and rollout checkpoints.

------------------------------------------------------------------------

## Research Context

This work was developed as part of an M2 Machine Learning internship at
**ISIR, Sorbonne Université**, on interaction readiness recognition for
Human-Robot Interaction.

The broader objective is to investigate whether general-purpose
multimodal foundation models can replace or complement pipelines based
on manually selected social cues, while supporting richer spatial and
temporal reasoning.

------------------------------------------------------------------------

## Limitations

-   Interaction readiness is context dependent and can be ambiguous even
    for human observers.
-   The three source datasets differ in environment, recording
    conditions, and original annotation objectives.
-   `interaction_ongoing` is less frequent than the two primary
    readiness classes.
-   Generative bounding-box prediction is sensitive to output formatting
    and localization errors.
-   Prediction-conditioned rollout can amplify errors over time.
-   Upper-bound rollout is diagnostic only and cannot be used in real-world
    deployment.
-   Results across different model sizes and training configurations
    should not be interpreted as fully controlled head-to-head
    comparisons unless all other conditions are held constant.

------------------------------------------------------------------------

## Citation

If you use this code or dataset in academic work, please cite the
repository and associated report/paper once a formal publication
reference is available.

``` bibtex
@misc{chahine2026interactionreadiness,
  title        = {Benchmarking Vision-Language Models for Interaction Readiness Detection in Human-Robot Interaction},
  author       = {Chahine, Marilyn N. and Rahimi, Hamed and Chetouani, Mohamed},
  year         = {2026},
  note         = {Sorbonne Université / ISIR}
}
```

------------------------------------------------------------------------

## Authors

**Marilyn N. Chahine**\
Hamed Rahimi\
Mohamed Chetouani

Sorbonne Université --- ISIR

------------------------------------------------------------------------

## Acknowledgements

This project builds on the pretrained multimodal models and open
research datasets used throughout the experiments. Please also cite the
original model and dataset publications when using their respective
resources.