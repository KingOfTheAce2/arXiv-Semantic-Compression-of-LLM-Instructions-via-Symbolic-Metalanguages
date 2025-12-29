# MetaGlyph — Symbolic Metalanguages for LLM Prompting

This repository contains the research artifacts, datasets, and evaluation framework for studying **symbolic metalanguages for large language model (LLM) prompting**. The goal of this project is to evaluate whether mathematical and logical operators can be used to **semantically compress instruction language**, independent of context compression or learned prompt optimization.

The work is designed to be **fully automatic, reproducible, and long-context aware**, and supports experiments using open-weight models executed locally (e.g., via Ollama), with optional validation on API-based models.

---

## Project motivation

Natural-language prompts act as *instruction languages*, specifying how inputs should be selected, transformed, or constrained. While effective, natural language is verbose and ambiguous as a control channel. This project investigates whether **symbolic operators already internalized during model pretraining** (e.g., ∈, ¬, ∩, ⇒) can function as compact, reliable instruction-semantic primitives.

Unlike prompt compression systems that prune context, or constructed prompt languages that rely on system-level decoding schemes, this work focuses on **semantic compression of the instruction language itself**, under strict token control.

---

## What this repository contains

This repository supports the experiments described in the accompanying paper and includes:

* **Task definitions** for four task families:

  * Selection and classification
  * Structured extraction
  * Constraint composition
  * Conditional transformation
* **Prompt variants** for each task instance:

  * Natural language (NL)
  * MetaGlyph symbolic instructions (MG)
  * Symbol-shaped semantic controls (CTRL)
* **Gold labels and constraints** for automatic evaluation
* **Operator fidelity checks** that verify whether symbolic constraints are respected
* **Token accounting metadata** for instruction-level token control
* **Evaluation outputs and aggregation artifacts** used to generate paper tables

All tasks are designed for **automatic scoring**; no manual output inspection is required to reproduce results.

---

## Symbolic operator inventory

MetaGlyph uses a constrained set of high-frequency mathematical and logical operators whose semantics are reinforced across domains during pretraining. The core operator inventory includes:

* Transformation and rules: `→`, `⇒`, `∘`, `↦`
* Set and constraints: `∈`, `∉`, `⊆`, `∩`, `∪`
* Logical control: `¬`, `∀`, `∃`
* Scope restriction: `|`

Natural-language predicates (e.g., `mammal`, `pet`, `technical`) are combined with symbolic structure to express instruction semantics compactly.

---

## Experimental design (high level)

Each task instance is evaluated under three instruction conditions:

1. **NL** — a verbose natural-language instruction
2. **MG** — a compact MetaGlyph instruction with equivalent semantics
3. **CTRL** — a symbol-shaped control matching token length and structure but breaking semantics

Instruction token counts are matched across conditions to isolate **semantic effects** from length or formatting effects. Models are run deterministically, and outputs are evaluated using exact-match, F1, and operator-specific constraint checks.

---

## Reproducibility

The project is designed for reproducibility:

* All primary experiments use **open-weight instruction-tuned models**, executed locally
* Model versions, decoding parameters, and tokenizers are fixed and recorded
* Prompt templates, token counts, and evaluation scripts are released
* Results can be regenerated end-to-end without API access

Optional validation using API-based models is supported but treated as supplementary evidence.

---

## Scope and limitations

This repository focuses on **instruction semantics**, not system performance. It does **not** claim or measure:

* latency improvements
* memory usage
* attention matrix complexity
* throughput or cost savings

The experiments are single-turn and do not evaluate multi-turn dialogue or interactive prompting.

---

## Intended audience

This project is intended for:

* NLP and computational linguistics researchers
* Prompting and instruction-following researchers
* Practitioners interested in principled prompt design
* Reviewers and readers of the associated arXiv paper

---

## Citation

If you use or build on this work, please cite the accompanying paper:

```
@article{metaglyph2025,
  title={Semantic Compression of LLM Instructions via Symbolic Metalanguages},
  author={Ernst van Gassen},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

This repository is released for research and academic use. See the `LICENSE` file for details.
