# PRINGLE

**PRINGLE** (Perception-Reasoning INference with Graph-based Logical Evaluation) is a research framework for **ontology-guided semantic validation of vision–language perception in autonomous driving**, with a focus on **traffic-light perception under adversarial and ambiguous conditions**.

Unlike prior vision–language or ontology-based approaches that operate offline or at the planning layer, PRINGLE embeds **explicit symbolic reasoning directly into the perception pipeline**, enabling inference-time validation of visual interpretations against domain knowledge, traffic rules, and contextual constraints.

---

## Key idea

PRINGLE treats perception failures not only as statistical errors, but as **violations of semantic and physical consistency**.  
By combining vision–language models with a **domain-specific traffic-light ontology**, the framework detects visually plausible but semantically invalid observations caused by physical manipulation, environmental ambiguity, or long-tail edge cases.

---

## Scope and focus

- **Targeted domain:** traffic-light perception
- **Threat model:** physical-world manipulation and ambiguity (e.g., light injection, reflections)
- **Primary goal:** security-oriented semantic validation at inference time
- **Out of scope:** network attacks, digital adversarial examples, V2X compromise

---

## Repository contents

This repository provides **dataset manifests, splits, and preparation scripts** used in the PRINGLE evaluation.

## Datasets

PRINGLE experiments are based on:
- **DTLD (DriveU Traffic Light Dataset)** for training and validation
- **A 25-image traffic-light attack benchmark** combining real-world and manipulated samples for evaluation

Exact dataset composition and provenance are documented in `datasets/`.
