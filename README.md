# FERMAT

**Foundation model for Exploring Real-world Multimodal health data using Autoregressive Trajectory modeling**

FERMAT is a clinical foundation model for patient representations and event/time generation. The current collaborator release is the **65.22M-parameter SNUH model**, with the validation-best stage-2 checkpoint at **93,000 updates** and corrected generation code.

## Current model and executable package

Start with **[the 2026-10-21 release](releases/fermat_snuh_20261021/README.md)**. It is self-contained and does not import the older root-level model or training scripts.

- [Model, checkpoint selection, and evaluation scope](releases/fermat_snuh_20261021/docs/MODEL_CARD.md)
- [Frozen vocabulary and preprocessing specification](releases/fermat_snuh_20261021/docs/INPUT_SPEC.md)
- [Completed experiments and project progress](releases/fermat_snuh_20261021/docs/PROGRESS.md)
- [Exact model configuration](releases/fermat_snuh_20261021/config/model_config.json)
- [Generation settings](releases/fermat_snuh_20261021/config/generation_config.json)
- [Executable verification](releases/fermat_snuh_20261021/config/verification.json)

The public repository contains code, configuration, synthetic examples, and aggregate results. The separately distributed collaborator ZIP also contains the model weights, frozen token registry, lab cutpoints, and optional next-record-day prediction head. GitHub code alone does not include those artifacts.

## Current evidence

The SNUH model was developed on a patient-level split of 3.77 million patients and 1.238 billion events. It has 10 layers, 10 attention heads, width 640, a 512-event context, and a 24,435-entry model vocabulary. Labs and the small genomics extension provide context rather than prediction targets.

The original manuscript studied both embedding-based disease prediction and generated next-record-day events. Later work expanded the frozen-embedding head to 100,000 training patients, corrected joint event/time generation, and completed a 1,999-patient, 50-path disease-risk evaluation. ETHOS is one comparator in that latter evaluation. See the progress report for version-specific results and termination handling; historical next-record-day metrics are not a new evaluation of the corrected generator.

## Legacy development files

The root-level model, training scripts, synthetic-data examples, and `docs/token_spec_v0.1.md` describe earlier development stages. They are preserved for history and are not the specification or runtime for the current SNUH handoff. The [previous README](docs/legacy/README_before_20261021.md) describes earlier project plans, including nationwide data integration; it should not be read as a claim that this release trained on those data or has external validation.
