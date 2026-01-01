# Methodology Review: Pipeline vs. Article Section 4

This document compares the implemented pipeline against the methodology described in Section 4 of Article.md, identifying what was implemented, what was not implemented, and any additional features not described in the original methodology.

---

## Section 4.1: Overview

### Described in Article
> "We evaluate MetaGlyph by comparing:
> 1. a natural-language instruction,
> 2. a MetaGlyph instruction that encodes the same constraints using symbolic operators, and
> 3. a symbol-shaped control that preserves the surface form of MetaGlyph while breaking its intended semantics."

### Implementation Status: **IMPLEMENTED**

The pipeline implements all three conditions:
- **NL**: `prompts/<family>/NL.txt` - Natural language instructions
- **MG**: `prompts/<family>/MG.txt` - MetaGlyph symbolic instructions
- **CTRL**: `prompts/<family>/CTRL.txt` - Control with nonsense symbols (⊙, ⊗ replacing ∩, ∈)

---

## Section 4.2: MetaGlyph Operator Inventory and Grammar

### Described in Article
> "MetaGlyph is operationalized as a minimal inventory of high-frequency mathematical and logical glyphs:
> - Transformation and rule structure: →, ⇒, ∘, ↦
> - Set and constraint operators: ∈, ∉, ⊆, ∩, ∪
> - Logical control: ¬, ∧, ∨"

### Implementation Status: **PARTIALLY IMPLEMENTED**

**Operators used in prompts:**
- ✅ ∈ (membership) - used in selection and extraction tasks
- ✅ ∩ (intersection) - used for conjunctive constraints
- ✅ ¬ (negation) - used for exclusion
- ✅ ⇒ (implication) - used in conditional transformation
- ✅ → (transformation) - used for mapping
- ✅ ∘ (composition) - used for sequential operations

**Operators NOT used:**
- ❌ ∉ (non-membership) - not found in prompt templates
- ❌ ⊆ (subset) - not found in prompt templates
- ❌ ∪ (union/disjunction) - not found in prompt templates
- ❌ ↦ (anonymous mapping) - not found in prompt templates
- ❌ ∧ (logical and) - not found in prompt templates
- ❌ ∨ (logical or) - not found in prompt templates

**Note:** The article mentions "edge-case groups in targeted ablations: quantifiers (∀, ∃) and 'soft constraint' operators (≈, =, ≠)". These are NOT implemented in the pipeline.

---

## Section 4.3: Prompt Variants

### Described in Article
> "For each test instance we construct three prompt variants that share the same input and differ only in the instruction segment."
>
> Two control families mentioned:
> - Permutation controls: swap a key operator with a different operator
> - Nonsense controls: replace operators with visually similar but semantically irrelevant glyphs

### Implementation Status: **PARTIALLY IMPLEMENTED**

- ✅ NL baseline implemented
- ✅ MG implemented
- ✅ CTRL with nonsense glyphs (⊙, ⊗, ⊕) implemented
- ❌ **Permutation controls NOT implemented** - Article mentions swapping ∩ with ∪, or ∈ with ∉, but this control family does not exist in the pipeline

**Discrepancy:** The pipeline only implements nonsense controls, not permutation controls.

---

## Section 4.4: Token Control Protocol

### Described in Article
> "Token accounting. We measure tokens using the official tokenizer for each model family."
> "Budget matching... we allow a tolerance of at most 1 token and record the residual difference."

### Implementation Status: **PARTIALLY IMPLEMENTED**

- ✅ Token counting implemented in `stage3_tokens.py`
- ✅ Token tolerance of ±1 implemented
- ⚠️ **Official tokenizers not used** - The pipeline uses `tiktoken` (OpenAI) or a simple word-based approximation. For Llama, Gemma, and OLMo models, the article specifies using "the official tokenizer for each model family" but the implementation falls back to tiktoken or simple tokenization.
- ❌ **NL paraphrase pool not implemented** - Article describes generating "10 to 30" NL paraphrases and selecting the one matching MG token count. The pipeline uses a single NL template without paraphrase generation.

**Discrepancy:** Token matching is based on a fixed NL template, not a pool of paraphrases matched to MG.

---

## Section 4.5: Tasks and Datasets

### Described in Article
> "Selection and classification... Inputs are small lists (8 to 20 items)"
> "Structured extraction... Inputs are short passages (150 to 400 words)"
> "Constraint composition... multiple constraints apply"
> "Conditional transformation... conditional rules using ⇒"
> "For each family we construct a benchmark of 100 to 300 instances"

### Implementation Status: **PARTIALLY IMPLEMENTED**

- ✅ Selection and Classification - implemented with pet/animal classification
- ✅ Structured Extraction - implemented with technical risk extraction
- ✅ Constraint Composition - implemented
- ✅ Conditional Transformation - implemented with account labeling

**Discrepancies:**
- ❌ **Instance count**: Article specifies 100-300 instances per family; pipeline uses 50 instances per family (configurable)
- ⚠️ Input diversity unclear - Current instances appear to be duplicated from instance_001 templates

---

## Section 4.6: Models and Decoding

### Described in Article
> "We evaluate across a heterogeneous set of instruction-tuned models: one API model and two open-weight baselines (for example, GPT-4o mini, Llama 3.1 8B Instruct, and Mistral 7B Instruct)."
> "temperature 0, top-p 1.0, and a fixed maximum output length per task"
> "Where a model remains nondeterministic at temperature 0, we repeat each prompt three times and aggregate by majority vote"

### Implementation Status: **PARTIALLY IMPLEMENTED**

- ✅ temperature 0 - implemented in config.json
- ✅ top-p 1.0 - implemented in config.json
- ✅ max_tokens 2048 - implemented in config.json
- ❌ **GPT-4o mini not used** - Pipeline uses Llama, Gemma, OLMo via OpenRouter, not GPT-4o mini
- ❌ **Majority vote for nondeterministic models NOT implemented** - No retry/vote aggregation logic exists

**Discrepancy:** Different model selection from article examples; no majority vote implementation.

---

## Section 4.7: Metrics

### Described in Article
> "Task performance. Standard accuracy, exact match, and F1"
> "Operator fidelity. For each symbol used in a task we define a concrete check"
> "Compression descriptors. We report instruction token count, character count, and compression ratio"

### Implementation Status: **IMPLEMENTED**

- ✅ Accuracy implemented in `stage5_evaluation.py`
- ✅ Exact match implemented
- ✅ F1 score implemented
- ✅ Operator fidelity checks implemented (OperatorFidelity dataclass)
- ✅ Token counts reported in `stage6_aggregation.py`
- ⚠️ Character count not explicitly reported (but trivially computable)

---

## Section 4.8: Statistical Analysis

### Described in Article
> "We compute 95% confidence intervals using bootstrap resampling over instances."
> "For binary outcomes we report paired significance tests (McNemar where appropriate)."
> "When comparing multiple operators or task families, we control the family-wise error rate using Holm correction."

### Implementation Status: **NOT IMPLEMENTED**

- ❌ **Bootstrap confidence intervals NOT implemented**
- ❌ **McNemar tests NOT implemented**
- ❌ **Holm correction NOT implemented**

**Major Gap:** The pipeline performs aggregation but does not compute statistical significance tests or confidence intervals.

---

## Section 4.9: Reproducibility

### Described in Article
> "We release prompt templates, the full set of instruction variants, tokenizer versions, and per-instance token counts."

### Implementation Status: **IMPLEMENTED**

- ✅ Prompt templates released in `prompts/`
- ✅ Token counts saved in `tokens/`
- ✅ Raw outputs saved in `outputs/`
- ✅ Evaluation results saved in `results/`

---

## Additional Features NOT in Article

The pipeline implements features not described in Section 4:

1. **Multiple backend support** - OllamaBackend, OpenAIBackend, AnthropicBackend, OpenRouterBackend (Article only mentions API models)

2. **skip_existing flag** - Allows resuming interrupted runs (not mentioned in article)

3. **Semantic equivalence metric** - `semantic_equivalence_pass` comparing NL==MG and both!=CTRL (implied but not explicitly defined in article)

4. **8 task families in code** - The pipeline references 8 families in execution output, but article describes 4. Investigation needed.

---

## Summary Table

| Methodology Element | Article Section | Implementation Status |
|---------------------|-----------------|----------------------|
| Three prompt conditions (NL, MG, CTRL) | 4.1 | ✅ Implemented |
| Full operator inventory | 4.2 | ⚠️ Partial (6/12 operators) |
| Quantifier ablations (∀, ∃) | 4.2 | ❌ Not implemented |
| Permutation controls | 4.3 | ❌ Not implemented |
| Nonsense controls | 4.3 | ✅ Implemented |
| Official model tokenizers | 4.4 | ❌ Uses tiktoken fallback |
| NL paraphrase pool | 4.4 | ❌ Not implemented |
| Token tolerance ±1 | 4.4 | ✅ Implemented |
| 100-300 instances | 4.5 | ❌ Uses 50 instances |
| Four task families | 4.5 | ✅ Implemented |
| Temperature 0 | 4.6 | ✅ Implemented |
| Majority vote aggregation | 4.6 | ❌ Not implemented |
| Accuracy, F1, exact match | 4.7 | ✅ Implemented |
| Operator fidelity checks | 4.7 | ✅ Implemented |
| Bootstrap CIs | 4.8 | ❌ Not implemented |
| McNemar tests | 4.8 | ❌ Not implemented |
| Holm correction | 4.8 | ❌ Not implemented |
| Reproducibility artifacts | 4.9 | ✅ Implemented |

---

## Recommendations

### Critical Gaps to Address
1. **Statistical analysis** - Add bootstrap CIs and significance tests (McNemar, Holm correction)
2. **Permutation controls** - Implement operator swapping (∩↔∪, ∈↔∉) as additional CTRL condition
3. **Instance count** - Increase to 100+ instances or justify reduced count

### Minor Gaps
1. Model tokenizers - Use model-specific tokenizers instead of tiktoken fallback
2. NL paraphrase pool - Implement paraphrase generation for exact token matching
3. Majority vote - Add retry logic for temperature-0 nondeterminism

### Documentation Updates Needed
1. Article should clarify which operators are actually tested (current 6 vs. described 12)
2. Article should document actual instance count used
3. Article should specify exact models used (Llama 3.2 3B, Gemma 3 12B, OLMo 3 7B)

---

## Conclusion

The pipeline implements the core experimental design from Section 4 but omits several important elements:
- Statistical rigor (no CIs or significance tests)
- Control condition diversity (no permutation controls)
- Full operator coverage (6/12 operators used)
- Instance scale (50 vs. 100-300)

These gaps should be addressed before the methodology section accurately describes the implemented experiments.
