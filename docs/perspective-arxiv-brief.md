# The Language We Train On: A Conjecture on Purpose-Built Substrate Languages for LLM Pre-Training

**Shaun Russell**
Independent Researcher
shaun.d.russell@gmail.com

*Preprint. Comments welcome.*

---

## Abstract

Large language models are trained almost exclusively on text produced for human readers — text shaped by millennia of communicative evolution, cultural accident, and biological constraint. We argue that this is a contingent, not necessary, choice. The substrate on which a model is trained is a design variable, and natural language is a poor default: it is morphologically irregular, syntactically ambiguous, and tokenizes with poor semantic alignment. We conjecture that a small constructed language designed from the ground up around the computational properties of transformer architectures and byte-pair encoding will yield measurably better training efficiency — lower bits-per-byte at equivalent compute and parameter count — than an English-trained baseline on the same semantic content. We describe the design of a candidate language (Loga), ground the conjecture in recent empirical literature, identify the gap in prior work, and propose a concrete experiment using freely available tools and consumer hardware. To our knowledge, the controlled pre-training comparison this conjecture requires has not yet been performed.

**Keywords**: language model pre-training, tokenization, constructed language, bits-per-byte, training efficiency

---

## 1. The Contingency of English

Every large language model trained today inherits the accidents of English. Its spelling system conflates letters and sounds without regularity: "though," "through," "tough," and "cough" share a graphemic suffix but no phoneme-orthography correspondence. Its morphology is partially fusional and partially analytic, mixing suffixed inflection with auxiliary constructions under no systematic rule. Its syntax permits movement, topicalization, and pragmatic reordering that introduce genuine attachment ambiguity resolvable only with world knowledge. Its vocabulary is dense with near-synonyms whose distributional differences are subtle and culturally contingent.

These properties are not defects in human language — they are features that emerged over millennia to serve human communicative needs: expressiveness, implicature, social signalling, poetic compression. But they are defects for a machine learner. A transformer model expends representational capacity learning that "ran," "run," and "running" share a root; that "do not" and "don't" are semantically identical; that "the bank" requires world-knowledge disambiguation. This capacity could instead be spent on the semantic content the model is trained to learn.

The field has responded to language's irregularities with a proliferation of engineering patches: byte-pair encoding (BPE) [1] to manage the long tail of surface forms; morpheme-aware tokenizer variants to better respect linguistic unit boundaries [2, 3]; byte-level architectures to sidestep tokenization entirely [4, 5]. These are patches on a substrate that was never designed for the task. We propose a more fundamental intervention: redesign the substrate itself.

---

## 2. Evidence That Language Structure Affects Learnability

The core claim — that a more regular training language would be easier for a neural network to learn — is not merely intuitive. It has direct empirical support.

Galke, Ram, and Raviv (2024) trained recurrent networks and a large language model on artificial languages varying systematically in compositional structure [6]. Languages in which meaning was expressed through regular, transparent combinations of primitives were learned faster, generalized better to held-out combinations, and produced more consistent representations across independently trained agents. The neural network behaves, in this respect, as a rational statistical learner — and rational learners benefit from regular substrates.

This finding provides adjacent empirical support for the conlang-for-LLM proposal. It is not direct evidence — Galke et al. tested compositional generalization on small artificial languages rather than next-token prediction on a translated Wikipedia-scale corpus — but it establishes that neural networks are sensitive to compositional regularity in a direction consistent with our hypothesis. The question shifts from pure speculation to a plausible conjecture: *how much* does regularity help at the scale of modern pre-training?

A concurrent information-theoretic analysis of tokenization [7, 8] finds that morphological alignment and compression efficiency must be jointly optimized; alignment alone is insufficient. This identifies the deeper design question: what properties of a language maximize the conditional predictability of its token stream without sacrificing expressive completeness? A purpose-built language can be designed to answer this question directly.

---

## 3. LLMs Develop an Internal Interlingua — and It Is Not Optimal

Schut et al. (2025) applied logit-lens probing to multilingual models processing French, German, Dutch, and Mandarin inputs [9]. They found that models make semantic decisions in a representational space closest to English, regardless of input or output language — an emergent interlingua, consistent with earlier observations in neural machine translation [10, 11, 12]. This interlingua is shaped by English, the dominant training language, rather than by any principled design.

If models develop an internal representational language anyway, the obvious question is whether training on a surface language designed to align with the architecture's inductive biases could cause the emergent interlingua to be *better* — more structured, more transferable, requiring fewer parameters to achieve equivalent compression.

Additional evidence that the reasoning substrate is a design variable comes from Coconut [13], which replaced discrete chain-of-thought tokens with continuous latent vectors and found that surface textual coherence constraints reduce inference efficiency. Natural language imposes costs on reasoning beyond what the reasoning itself requires.

---

## 4. What Has Been Done and What Has Not

Several adjacent research directions have approached this problem without converging on the specific intervention we propose.

**Morpheme-aware tokenization.** MorphBPE [2] and MorphPiece [3] improve alignment between tokenizer outputs and morphological units, but operate within the constraint of natural language. They cannot address the irregularities encoded in English morphology — only improve the tokenizer's response to them.

**Byte-level models.** ByT5 [4], MegaByte [14], and the Byte Latent Transformer [5] remove the tokenizer as a source of inefficiency but do not address the underlying irregularity of the training language. A byte-level model trained on English still learns irregular morphological alternations and resolves syntactic ambiguity from distributional evidence.

**AI-centric language design.** A 2025 preprint [15] proposes a framework for an AI-centric language system and argues that regularity, unambiguity, and conciseness would reduce computational overhead in transformer architectures. This is the closest conceptual precedent to the present work. However, it presents a theoretical framework only. No controlled experiment has been performed in which an LLM is trained from scratch on a purpose-designed constructed language and compared against an English baseline with identical architecture, parameter count, and compute budget.

**That experiment is what we propose.**

---

## 5. Loga: A Candidate LLM-Optimized Language

We describe a candidate constructed language, Loga, designed around the measurable properties identified above. The full specification is available at https://github.com/pragmaticsc/loga.

**Character set.** Loga uses the 95 printable ASCII characters (U+0020–U+007E), all 1 byte under UTF-8. Character ranges are partitioned by semantic role: lowercase first character marks noun roots; uppercase marks verb roots; punctuation characters `!`–`/` serve as case suffixes; characters `:`–`@` serve as tense markers. For lowercase-first tokens the first byte alone identifies noun roots; for uppercase-first tokens, verbs and proper nouns are distinguished by the third byte (tense marker vs. case suffix) — always local and deterministic, requiring no sentence-level context.

**Root structure.** All roots are exactly two characters, giving $26 \times 62 = 1{,}612$ noun roots and $1{,}612$ verb roots ($3{,}224$ usable) in 2 bytes. A core vocabulary of approximately 300 noun roots and 200 verb roots ensures every root appears thousands of times in a Wikipedia-scale corpus — a prerequisite for reliable BPE merging.

**Morphology.** Loga is agglutinative with invariant roots. An inflected word is exactly 3 bytes: $[\text{root}_1][\text{root}_2][\text{suffix}]$, where the suffix is a case marker for nouns or a tense marker for verbs. No word carries both simultaneously. English's fusional irregularities — go/went, is/was/be, good/better/best — are structurally impossible.

**Syntax.** Word order is strict SOV. The grammar is context-free. No topicalization, no wh-movement, no garden-path constructions.

**Worked example.** The English sentence "The person sees the thing." encodes as:

| Loga | Gloss | Bytes | BPE tokens (est.) |
|------|-------|-------|-------------------|
| `ka!` | person + nominative | 3 | 1 |
| `da"` | thing + accusative | 3 | 1 |
| `Se:` | see-verb + present | 3 | 1 |
| `.` | sentence boundary | 1 | 1 |

Total: 13 bytes (including 3 word-delimiter spaces), ~4 tokens. English "The person sees the thing." is 26 bytes and tokenises to ~6 BPE tokens. At BPE vocab=8,192 trained on Loga text, every inflected 3-byte word form appears thousands of times and merges to a single token, approaching the ideal of one token per semantic+grammatical unit.

---

## 6. The Conjecture and Proposed Experiment

> **Conjecture**: A transformer model trained from scratch on a Loga-encoded corpus will achieve lower bits-per-byte (val\_bpb) than an architecturally identical model trained on the English-encoded version of the same corpus, at equivalent parameter count and compute budget.

Bits-per-byte measures how efficiently a model compresses held-out text. Lower val\_bpb indicates more learned structure per parameter. We predict a measurable reduction relative to the English baseline, primarily driven by:

1. **Tokenization alignment**: BPE trained on Loga will reliably merge 3-byte inflected word forms into single tokens. English BPE produces inconsistent multi-token fragments.
2. **Conditional entropy reduction**: Loga's strict SOV syntax and agglutinative morphology reduce next-token entropy given context — the model has stronger priors about what class of token must follow.
3. **No irregular alternation**: the model need not allocate parameters to learning that go/went are the same verb, or that bank is ambiguous.

**Experiment.** Two cells, identical in all respects except training language:

| | English corpus | Loga corpus |
|--|---|---|
| **float16 weights** | A: reference | B: tests conjecture |

Both cells use the same model architecture (small-scale transformer, 10–50M parameters), the same compute budget (automated hyperparameter search via autoresearch-mlx [16] on Apple M4 Max), and BPE tokenizers of identical vocabulary size (8,192) trained on their respective corpora.

The training corpus for both conditions is Simple English Wikipedia (~160MB, ~250K articles). The Loga corpus is produced by LLM-assisted translation (claude-sonnet-4-6 with the full grammar specification as system context), with back-translation validation against a sentence embedding threshold to filter low-fidelity articles. Both cells train until a matched BPE token budget; results report both byte-normalized bpb (the primary metric) and token-normalized bpb, allowing tokenizer efficiency to be separated from model learning efficiency.

---

## 7. Limitations

The primary limitation is translation fidelity: systematic errors in the LLM-generated Loga corpus would mean the model learns noise rather than structure. Back-translation validation at 1% of articles provides a partial check. A second limitation is effective vocabulary size: Loga's ~500 core roots is smaller than English's open vocabulary, so lower val\_bpb is partly consistent with an "intrinsically easier prediction task" interpretation. Reporting the type-token ratio and effective vocabulary size of each corpus will characterise this confound. A third limitation is confounded design: Loga bundles compositional morphology, reduced lexical ambiguity, and BPE-aligned character partitioning; a difference in val\_bpb will not isolate which factor drives it.

---

## 8. Conclusion

The language a model is trained on is not a fixed background condition. It is an engineering choice, subject to optimization like any other. Natural language was not designed for machine learning; it evolved for human communication. The tokenization workarounds, architectural extensions, and post-hoc alignment procedures the field has developed are evidence of a mismatch, not a resolution of it.

A converging body of evidence — from compositional learning theory [6], tokenization information theory [7, 8], multilingual representation probing [9], and continuous reasoning research [13] — provides principled grounds for the conjecture. What remains is the experiment. The required tools are freely available, and the compute required is accessible on consumer hardware.

We offer this paper as a precise statement of the conjecture and a proposal for the experiment needed to test it. To our knowledge, the controlled comparison has not been performed. We invite the community to run it.

---

## Acknowledgements

The authors thank the developers of autoresearch-mlx, the MLX framework (Apple), and the Simple English Wikipedia community. Translation infrastructure uses the Anthropic Claude API.

---

## References

[1] Sennrich, R., Haddow, B. & Birch, A. (2016). Neural machine translation of rare words with subword units. *ACL 2016*. https://aclanthology.org/P16-1162/

[2] MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies. (2025). arXiv:2502.00894.

[3] Jabbar, H. et al. (2023). MorphPiece: A Linguistic Tokenizer for Large Language Models. arXiv:2307.07262.

[4] Xue, L. et al. (2022). ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models. *TACL* 10, 291–306. arXiv:2105.13626.

[5] Pagnoni, A. et al. (2024). Byte Latent Transformer: Patches Scale Better Than Tokens. arXiv:2412.09871.

[6] Galke, L., Ram, Y. & Raviv, L. (2024). What Makes a Language Easy to Deep-Learn? Deep Neural Networks and Humans Similarly Benefit from Compositional Structure. *Nature Communications*. arXiv:2302.12239.

[7] Evaluating Morphological Alignment of Tokenizers in 70 Languages. (2025). arXiv:2507.06378.

[8] Erdogan, M. et al. (2026). An Information-Theoretic Perspective on LLM Tokenizers. arXiv:2601.09039.

[9] Schut, L., Gal, Y. et al. (2025). Do Multilingual LLMs Think In English? *ICLR 2025*. arXiv:2502.15603.

[10] Johnson, M. et al. (2017). Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation. *TACL* 5, 339–351. arXiv:1611.04558.

[11] Lu, Y. et al. (2018). A Neural Interlingua for Multilingual Machine Translation. *WMT 2018*. arXiv:1804.08198.

[12] Zhu, C. et al. (2020). Language-Aware Interlingua for Multilingual Neural Machine Translation. *ACL 2020*. https://aclanthology.org/2020.acl-main.150/

[13] Hao, S. et al. (2024). Training Large Language Models to Reason in a Continuous Latent Space. *ICLR 2025*. arXiv:2412.06769.

[14] Yu, L. et al. (2023). MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers. *NeurIPS 2023*. arXiv:2305.07185.

[15] Building A Unified AI-centric Language System: Analysis, Framework and Future Work. (2025). arXiv:2502.04488.

[16] autoresearch-mlx. trevin-creator. https://github.com/trevin-creator/autoresearch-mlx
