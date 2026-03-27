# The Language We Train On: A Case for Purpose-Built Substrate Languages in Large Language Model Pre-Training

**Shaun Russell**
Independent Researcher
shaun.d.russell@gmail.com

*Preprint. Comments welcome.*

---

## Abstract

Large language models are trained almost exclusively on text produced for human readers — text shaped by millennia of communicative evolution, cultural accident, and biological constraint. We argue that this is a contingent, not necessary, choice. The substrate on which a model is trained is a design variable, and natural language is a poor default: it is morphologically irregular, syntactically ambiguous, and tokenizes with poor semantic alignment. We propose a research programme centred on three conjectures. **Conjecture 1**: a small constructed language (*conlang*) designed from the ground up around the computational properties of transformer architectures and byte-pair encoding will yield measurably better training efficiency — lower bits-per-byte at equivalent compute and parameter count — than an English-trained baseline on the same semantic content. **Conjecture 2**: models trained on such a conlang will tolerate aggressive weight quantization (ternary weights ∈ {−1, 0, +1}) better than English-trained models, because the lower information complexity of the training distribution reduces the capacity cost of discretization. **Conjecture 3**: the zeros produced by ternary quantization will cluster at the structural level — entire attention heads, not randomly distributed weights — in conlang-trained models more than in English-trained models, because grammatical regularity enables deeper head specialization; this structured sparsity enables additional post-training compression beyond what ternary quantization alone provides. We describe the design of a candidate language (Loga), ground all three conjectures in recent empirical literature, identify the gap in prior work, and propose a concrete 2×2 factorial experiment using freely available tools and consumer hardware. To our knowledge, the controlled pre-training comparison these conjectures require has not yet been performed.

**Keywords**: language model pre-training, tokenization, constructed language, weight quantization, structured sparsity, bits-per-byte, training efficiency, attention head specialization

---

## 1. The Contingency of English

Every large language model trained today inherits the accidents of English. Its spelling system conflates letters and sounds without regularity: "though," "through," "tough," and "cough" share a graphemic suffix but no phoneme-orthography correspondence. Its morphology is partially fusional and partially analytic, mixing suffixed inflection with auxiliary constructions under no systematic rule. Its syntax permits movement, topicalization, and pragmatic reordering that introduce genuine attachment ambiguity resolvable only with world knowledge. Its vocabulary is dense with near-synonyms whose distributional differences are subtle and culturally contingent.

These properties are not defects in human language — they are features that emerged over millennia to serve human communicative needs: expressiveness, implicature, social signalling, poetic compression. But they are defects for a machine learner. A transformer model expends representational capacity learning that "ran," "run," and "running" share a root; that "do not" and "don't" are semantically identical; that "the bank" requires world-knowledge disambiguation. This capacity could instead be spent on the semantic content the model is trained to learn.

The field has responded to language's irregularities with a proliferation of engineering patches: byte-pair encoding (BPE) [1] to manage the long tail of surface forms; morpheme-aware tokenizer variants to better respect linguistic unit boundaries [2, 3]; byte-level architectures to sidestep tokenization entirely [4, 5]. These are patches on a substrate that was never designed for the task. We propose a more fundamental intervention: redesign the substrate itself.

---

## 2. Evidence That Language Structure Affects Learnability

The core claim — that a more regular training language would be easier for a neural network to learn — is not merely intuitive. It has direct empirical support.

Galke, Ram, and Raviv (2024) trained recurrent networks and transformer models on artificial languages varying systematically in compositional structure [6]. Languages in which meaning was expressed through regular, transparent combinations of primitives were learned faster, generalized better to held-out combinations, and produced more consistent representations across independently trained agents. This mirrors cognitive science findings on human language acquisition: humans also learn more compositionally regular languages faster and with fewer errors. The neural network behaves, in this respect, as a rational statistical learner — and rational learners benefit from regular substrates.

This finding reframes the conlang-for-LLM proposal from speculation to an application of a demonstrated principle. The question shifts from *whether* regularity helps to *how much* it helps at the scale of modern pre-training and on a realistic corpus.

---

## 3. The Tokenization Problem Is Deeper Than It Appears

Standard practice treats tokenization as a preprocessing detail. BPE [1] is the dominant approach: a greedy algorithm that merges the most frequent byte pairs in the training corpus until a target vocabulary size is reached. BPE is corpus-efficient but linguistically naive. It splits "unfortunately" into four tokens and merges "New York" into one, based entirely on co-occurrence frequency. Morpheme boundaries are violated routinely and inconsistently.

Recent work has established that this matters, though the evidence is nuanced. MorphBPE [2] and MorphPiece [3] demonstrate that constraining BPE merges to respect morpheme boundaries is feasible and does not degrade — and in some settings improves — downstream performance. However, a large-scale evaluation across 70 languages found that morphological alignment of tokenizers does not consistently predict downstream performance [7], suggesting alignment alone is insufficient. A concurrent information-theoretic analysis argues that this is because compression ratio and linguistic granularity must be *jointly* optimized [8]: a tokenizer that aligns to morpheme boundaries without increasing the predictability of the resulting token stream gains little.

This identifies the deeper question: what properties of a language maximize the conditional predictability of its token stream without sacrificing expressive completeness? This is a design question that natural language cannot answer, because natural language was not designed. A purpose-built language can.

### 3.1 Character Set as a Design Implementation Detail

The choice of writing system for a purpose-built language is an engineering decision, though it is worth clarifying what it does and does not determine. After tokenization, a model operates on integer token IDs — the specific glyphs are invisible to the network weights. The character set matters not for its visual properties but for how it shapes the BPE merge process: character range partitioning that coincides with morpheme boundaries gives the tokenizer strong statistical signal to merge complete morphemes into single tokens, rather than cutting arbitrarily through them.

A concrete illustration: Shannon entropy per byte — $H(X) / \text{bytes}(X)$ — measures how much semantic work each byte carries. Under UTF-8 encoding:

| Alphabet | Chars | Bytes/char | Bits/byte |
|----------|-------|-----------|-----------|
| Lowercase ASCII | 26 | 1 | 4.70 |
| Full printable ASCII | 95 | 1 | **6.57** |
| Latin Extended (ü, é, …) | 256 | 2 | 4.00 |
| CJK ideographs | ~20,000 | 3 | 4.76 |

The counterintuitive result: CJK characters — despite encoding thousands of concepts visually — are *less* byte-efficient than two-character ASCII roots because the 3-byte UTF-8 encoding cost cancels the representational gain. The information-theoretic optimum within the 1-byte encoding range is the full 95-character printable ASCII set, achieving 6.57 bits per byte — a 40% improvement over lowercase-only alphabets at no encoding cost. This informs the root size in Loga: a 2-character root drawn from this alphabet encodes $95^2 = 9{,}025$ distinct concepts in 2 bytes, a 3.6× increase over phonotactically restricted systems, ensuring roots are frequent enough for reliable BPE merging.

The character partitioning is primarily a design convenience that enables the BPE tokenizer to reliably identify morpheme boundaries. The actual efficiency gains we conjecture derive from compositional regularity and conditional entropy reduction — not from the choice of glyphs. A Loga rewritten in a different character set but preserving the same morphological structure should produce similar results.

---

## 4. LLMs Develop an Internal Interlingua — and It Is Not Optimal

A striking recent finding deepens the motivation for deliberate substrate design. Schut et al. (2025) applied logit-lens probing and activation steering to multilingual models processing French, German, Dutch, and Mandarin inputs [9]. They found that models make semantic decisions in a representational space closest to English, regardless of input or output language: the model resolves meaning in an English-like latent space in intermediate layers before projecting to the target language in later ones. Activation steering is more effective when computed in the English-like space.

This is simultaneously encouraging and troubling. Encouraging, because it demonstrates that transformers spontaneously develop approximately language-independent internal representations — an emergent interlingua, consistent with earlier observations in neural machine translation [10, 11, 12]. Troubling, because that interlingua is shaped by English, the dominant training language, rather than by any principled design. If models develop an internal representational language anyway, the obvious question is: could training on a surface language specifically designed to align with the architecture's inductive biases cause the emergent interlingua to be *better* — more structured, more transferable, requiring fewer parameters to achieve equivalent compression?

Additional evidence that the reasoning substrate is a genuine design variable comes from Coconut [13], in which Meta AI replaced discrete chain-of-thought tokens with continuous latent vectors. Coconut found that language-space reasoning is suboptimal: most tokens in standard CoT chains enforce surface textual coherence rather than contributing to the inference computation. Replacing them with unconstrained latent vectors improved performance on logical reasoning tasks. The implication is that natural language imposes costs on reasoning beyond what the reasoning itself requires.

---

## 5. What Has Been Done and What Has Not

Several adjacent research directions have approached this problem without converging on the specific intervention we propose.

**Interlingua in neural machine translation.** Shared multilingual encoders develop approximately language-independent representations [10, 11, 12], enabling zero-shot translation. These representations are useful for translation quality but were not designed to improve the efficiency of the training process itself.

**Morpheme-aware tokenization.** MorphBPE [2] and MorphPiece [3] improve alignment between tokenizer outputs and morphological units, but operate within the constraint of natural language input. They cannot address the irregularities encoded in English morphology — only improve the tokenizer's response to them.

**Byte-level models.** ByT5 [4], MegaByte [14], and the Byte Latent Transformer [5] sidestep tokenization by operating on raw bytes or entropy-based patches. These approaches remove the tokenizer as a source of inefficiency but do not address the underlying irregularity of the training language. A byte-level model trained on English still must learn irregular morphological alternations and resolve syntactic ambiguity from distributional evidence.

**AI-centric language design.** A 2025 preprint [15] proposes a framework for an AI-centric language system, drawing design principles from Esperanto and Lojban and arguing that regularity, unambiguity, and conciseness would reduce computational overhead in transformer architectures. This is the closest conceptual precedent to the present work. However, it presents a theoretical framework only. No controlled experiment has been performed in which an LLM is trained from scratch on a purpose-designed constructed language and compared against an English baseline with identical architecture, parameter count, and compute budget.

**That experiment is what we propose.**

---

## 6. Loga: A Candidate LLM-Optimized Language

We describe a candidate constructed language, Loga, designed around the measurable properties identified above. The full grammar is available at https://github.com/pragmaticsc/loga.

**Character set.** Loga uses the 95 printable ASCII characters (U+0021–U+007E), all 1 byte under UTF-8. Character ranges are partitioned by semantic role: lowercase first character marks noun roots; uppercase marks verb roots; punctuation characters `!`–`/` serve as case suffixes; characters `:`–`@` serve as tense and aspect markers. A transformer can determine the syntactic function of any token from its first byte alone — no context required.

**Root structure.** All roots are exactly two characters. The noun root inventory is $26 \times 62 = 1{,}612$ possible forms; the verb root inventory is identical. A core vocabulary of approximately 300 noun roots and 200 verb roots is used, ensuring every root appears thousands of times in a Wikipedia-scale corpus — a prerequisite for reliable BPE merging.

**Morphology.** Loga is agglutinative with invariant roots. Each grammatical layer (case, tense, aspect) is expressed by a single appended character. A fully inflected word is exactly 4 bytes: $[\text{root}_1][\text{root}_2][\text{case}][\text{tense}]$. The root never changes form. English's fusional irregularities — go/went, is/was/be, good/better/best — are structurally impossible.

**Syntax.** Word order is strict SOV with no movement rules. The grammar is context-free. No topicalization, no wh-movement, no garden-path constructions.

**Compositionality.** Novel concepts are formed by compounding two roots with a joiner character `{`, preserving compositional transparency: `ku{Ma` (water-make) = irrigate; `ge{se` (fire-place) = volcano; `pa{ka` (idea-person) = philosopher. This follows Galke et al.'s finding [6] that compositional transparency supports generalization, as opposed to logographic expansion which requires each new concept to be memorized.

**Worked example.** The English sentence "The cat sat on the mat." encodes as:

| Loga | Gloss | Bytes | BPE tokens (est.) |
|------|-------|-------|-------------------|
| `ka!` | person/cat + nominative | 3 | 1 |
| `ma%` | mat + locative | 3 | 1 |
| `Si;` | sit-verb + past | 3 | 1 |
| `.` | sentence boundary | 1 | 1 |

Total: 10 bytes, ~4 tokens. English "The cat sat on the mat." is ~26 bytes and tokenises to 9–10 BPE tokens (GPT-4 tokenizer). At BPE vocab=8,192 trained on Loga text, every inflected 3–4 byte word form appears thousands of times and merges to a single token, approaching the ideal of one token per semantic+grammatical unit.

---

## 7. Two Conjectures and a Factorial Experiment

### 7.1 Conjecture 1: Training Language Affects Pre-Training Efficiency

> *A transformer model trained from scratch on a Loga-encoded corpus will achieve lower bits-per-byte (val\_bpb) than an architecturally identical model trained on the English-encoded version of the same corpus, at equivalent parameter count and compute budget.*

Bits-per-byte measures how efficiently a model compresses held-out text. Lower val\_bpb indicates more learned structure per parameter. We predict a 10–20% reduction relative to the English baseline, primarily driven by:

1. **Tokenization alignment**: BPE trained on Loga text will reliably merge 3–4 byte inflected word forms into single tokens, approaching one token per semantic+grammatical unit. English BPE produces inconsistent multi-token fragments.
2. **Conditional entropy reduction**: Loga's strict SOV syntax and agglutinative morphology reduce next-token entropy given context — the model has stronger priors about what class of token must follow.
3. **No irregular alternation**: the model need not allocate parameters to learning that go/went are the same verb, or that bank is ambiguous.

### 7.2 Conjecture 2: Structured Training Distributions Tolerate Ternary Quantization Better

> *A Loga-trained model will suffer a smaller val\_bpb penalty from ternary weight quantization (weights ∈ {−1, 0, +1}, BitNet b1.58 style [16]) than an English-trained model of identical architecture.*

Ternary weight quantization discretizes each weight to one of three values, reducing model size by approximately 10× relative to float16 while eliminating multiply-accumulate operations at inference time [16, 17]. The quantization introduces an accuracy cost proportional to the precision lost. We conjecture that this cost is smaller when the training distribution is more regular: a model encoding a lower-entropy corpus requires less numerical precision to represent the same amount of structure. Put differently, if the weights of an English-trained model are more "loaded" — carrying more bits of information per parameter — then truncating them to ternary loses more.

This conjecture has no direct prior literature. The closest evidence is indirect: Galke et al. [6] found that more compositionally regular languages produce more consistent representations across independently trained agents, which is consistent with the idea that regular training produces more "compressible" weights. The quantization-tolerance question requires a direct experiment.

### 7.3 Experimental Design

The two conjectures are tested jointly in a 2×2 factorial experiment:

|  | English corpus | Loga corpus |
|--|---|---|
| **float16 weights** | A: reference | B: tests Conjecture 1 |
| **ternary weights** | C: tests quantization alone | D: tests interaction (Conjecture 2) |

All four cells use identical model architecture (nanochat-scale, 10–50M parameters), identical compute budget (automated hyperparameter search via autoresearch-mlx [18] on Apple M4 Max), and BPE tokenizers of identical vocabulary size (8,192) trained on their respective corpora.

The training corpus for both English and Loga conditions is Simple English Wikipedia (~160MB, ~250K articles). The Loga corpus is produced by LLM-assisted translation (claude-sonnet-4-6 with the full grammar specification as system context), with back-translation validation against a sentence embedding threshold to filter low-fidelity articles. All four cells train until a matched BPE token budget (identical number of tokens seen per cell); results report both byte-normalized bpb (bits per input byte, the primary metric) and token-normalized bpb (bits per output token), allowing tokenizer efficiency to be separated from model learning efficiency.

Ternary training follows the BitNet b1.58 protocol [16]: standard linear layers are replaced with BitLinear layers (absmean quantization to {−1, 0, +1}, straight-through estimator for gradients), with floating-point activations. We use an MLX-native `BitLinear` implementation (included in the repository at `train/bitlinear.py`) rather than the official CPU `bitnet` reference library; MLX's Metal kernels enable efficient quantization-aware training on Apple Silicon without requiring CUDA. Fully ternary activations are not used; quantizing activations in LLMs causes severe accuracy loss [19] and is orthogonal to the hypotheses under test.

### 7.4 Conjecture 3: Regular Training Induces Structured Sparsity

> *In ternary-weight models, the zero weights produced by quantization will be distributed more* structurally *— clustering at the head or block level rather than uniformly across weight matrices — in Loga-trained models than in English-trained models. This structured sparsity enables post-training pruning that further reduces effective model size with smaller val\_bpb degradation than equivalent pruning of English-trained ternary models.*

The mechanism operates at the level of attention head specialization. Prior work has established that transformer attention heads divide labour, with individual heads specializing in specific syntactic functions — positional dependencies, rare tokens, syntactic agreement — and that heads which do not specialize can be pruned with minimal accuracy cost [20, 21]. Crucially, this specialization is more complete when the linguistic patterns to be learned are more regular.

In Loga, the grammar is context-free and exhaustive: case suffixes encode syntactic role exactly, word order is invariant, and morphological forms are transparent. An attention head tasked with learning subject-verb agreement, for instance, needs only to attend from each verb's tense marker to the nearest nominative case suffix — a single, consistent pattern with no exceptions. Once learned, the head's weight matrix is near-zero for all input configurations that do not match this pattern, because those configurations never appear in training.

In English, the same head would handle partially overlapping responsibilities: irregular subject-verb agreement, auxiliary constructions, long-distance dependencies, and exceptions accumulated over centuries of language change. The weights are never free to collapse toward zero because every pattern has exceptions that require residual capacity.

Under ternary quantization, near-zero weights round to exactly zero. The prediction is therefore:

1. **Higher head-level zero variance in Loga models.** Some heads become nearly fully zero (the pattern they specialized on is handled elsewhere or is redundant); others remain dense. English-trained models show more uniform zero distributions across heads.

2. **Greater pruning headroom in Loga ternary models.** Structured pruning — removing entire heads whose weight matrices are predominantly zero — degrades val\_bpb less for Loga ternary (cell D) than for English ternary (cell C) at the same pruning rate.

3. **A compression cascade.** The pipeline `conlang training → ternary QAT → structured head pruning` achieves effective model sizes smaller than any individual step alone. This partially offsets the capacity concern that ternary quantization imposes on small models.

**Measurement protocol.** Conjecture 3 is evaluated as a post-training analysis on the models from cells C and D (ternary runs), without additional training:

- Compute the fraction of zero weights per attention head for every head in every layer of both models.
- Measure the variance of this distribution across heads. Higher variance indicates more structured sparsity (Loga prediction) vs. near-uniform distribution (English prediction).
- Apply magnitude-based head pruning at thresholds 10%, 20%, 30%, 40% of heads removed. Record val\_bpb after each pruning step for both models.
- Compare the pruning degradation curves: a shallower curve for cell D than cell C supports Conjecture 3.

**Learning curve measurement.** All four cells log val\_bpb at regular training step intervals, not only at convergence. Plotting val\_bpb against training steps (compute) directly tests sample efficiency: if the Loga cells (B and D) reach a given val\_bpb threshold in fewer steps than their English counterparts (A and C), this supports the interpretation that a more regular training substrate reduces the data required to learn equivalent structure — the core prediction underlying all three conjectures.

---

## 8. Implications and Limitations

If Conjecture 1 is supported, it implies that the choice of training language is an unexploited lever on model efficiency. Pre-training corpora could be designed or curated for tokenization alignment, not just scale and quality. If Conjecture 2 is supported, it implies a design principle for quantized models: train on simpler, more regular data to make the quantization penalty cheaper. If Conjecture 3 is supported, the implications compound: the three techniques — conlang training, ternary quantization, and structured pruning — form a compression cascade in which each step makes the next cheaper, potentially enabling capable models at parameter counts previously considered inadequate. All three results would motivate further work at larger scales and with richer evaluation.

**Limitations.** The primary limitation is translation fidelity. The Loga training corpus is an LLM-generated translation; systematic translation errors would mean the model learns noise rather than structure. Back-translation validation at 1% of articles provides a partial check but cannot eliminate this risk. A secondary limitation is corpus byte parity: the Loga corpus will differ in byte count from English for the same semantic content. Controlling for this requires training to equivalent token budgets rather than byte counts. Both limitations are methodological challenges with straightforward mitigations, not objections in principle.

A third limitation is sequence length. Loga's explicit morphology — encoding "went" as root + tense suffix rather than a single token — may expand token sequences for equivalent semantic content. Longer sequences raise attention cost (O(n²)) and, within autoresearch-mlx's fixed 5-minute budget, reduce the number of optimizer steps per run. Two factors mitigate this. First, the primary metric val_bpb normalises by bytes of *input*, not number of tokens, so a longer sequence that compresses better still registers a lower val_bpb; the metric directly captures whether the added length is earning its keep. Second, BPE trained on Loga text will merge frequent 4-byte inflected word forms into single tokens, recovering much of the sequence length that morphological explicitness adds. The experiment will report tokens-per-article alongside val_bpb to make the sequence length effect directly observable.

A fourth limitation is confounded design. Loga bundles three innovations: (a) compositional morphology, (b) reduced lexical ambiguity and polysemy, and (c) BPE-aligned character partitioning. A difference in val_bpb between cells A and B will not isolate which factor drives it. A natural extension for future work is a "Normalised English" control condition — English text with irregular verbs lemmatised, contractions expanded, and synonym variation reduced — which would share property (b) with Loga without (a) or (c). Including such a condition in a follow-on experiment would disentangle the contribution of morphological structure from the simpler effect of surface regularisation.

A further consideration is scale. BitNet b1.58's strongest results are at ≥1B parameters; the gap between ternary and float16 narrows as scale increases [16]. At 10–50M parameters, Conjecture 2's interaction effect may be difficult to observe against the baseline quantization penalty. We consider this an acceptable limitation for a first experiment: the within-ternary comparison (cell D versus cell C) is well-defined regardless of whether either ternary model matches its float16 counterpart in absolute terms.

Conjecture 3 faces an additional limitation: head-level sparsity structure may be an artefact of model size or architecture rather than training language. Controlling for this requires comparing zero-fraction variance across heads at matched parameter counts, which the 2×2 design provides. A further confound is that structured sparsity could emerge from ternary quantization alone, independent of the training language; cell C (English ternary) provides the baseline against which cell D (Loga ternary) must be compared, not the float16 cells.

---

## 9. Conclusion

The language a model is trained on is not a fixed background condition. It is an engineering choice, subject to optimization like any other. Natural language was not designed for machine learning; it evolved for human communication over millennia of biological and social constraint. The tokenization workarounds, architectural extensions, and post-hoc alignment procedures the field has developed are evidence of a mismatch, not a resolution of it.

A converging body of evidence — from compositional learning theory [6], tokenization information theory [7, 8], multilingual representation probing [9], continuous reasoning research [13], and attention head specialization studies [20, 21] — now provides principled grounds for all three conjectures. What remains is the experiment. The required tools — autoregressive pre-training at modest scale, automated hyperparameter search, BPE tokenizer training, LLM-assisted corpus translation — are all freely available, and the compute required is accessible on consumer hardware.

The three conjectures form a logical chain. If the training language is a design variable (Conjecture 1), then simpler languages may lower the information-theoretic floor on what a model needs to represent. If the representational burden is lower (Conjecture 2), less numerical precision is needed to represent it, making quantization cheaper. If the representational structure is more organized (Conjecture 3), the sparsity that quantization induces will be more structured, making further compression cheaper still. Each step in this chain is individually testable and individually useful; the chain as a whole, if supported, describes a new approach to efficient language model training from first principles.

We offer this paper as a statement of the conjectures and a map of the territory. We expect to be wrong about some things. We invite the community to tell us which ones, and to run the experiment if we do not. This paper does not report experimental results; it states the conjectures precisely, grounds them in existing empirical literature, and describes the experiment required to test them. If Conjecture 1 holds, the more consequential follow-on question is whether reduced training data requirements extend to logical reasoning specifically — whether a Loga pre-trained model, fine-tuned on a translated reasoning benchmark, reaches higher accuracy than an identically-sized English-trained counterpart with equivalent fine-tuning data. That question is left for subsequent work.

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

[16] Ma, S., Wang, H. et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764.

[17] Ma, S. et al. (2025). BitNet b1.58 2B4T Technical Report. arXiv:2504.12285.

[18] autoresearch-mlx. trevin-creator. https://github.com/trevin-creator/autoresearch-mlx

[19] Deng, L. et al. (2018). GXNOR-Net: Training Deep Neural Networks with Ternary Weights and Activations without Full-Precision Memory. *Neural Networks* 108, 400–412. arXiv:1705.09283.

[20] Voita, E. et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. *ACL 2019*. https://aclanthology.org/P19-1580/

[21] Michel, P., Levy, O. & Neubig, G. (2019). Are Sixteen Heads Really Better than One? *NeurIPS 2019*. arXiv:1905.10650.
