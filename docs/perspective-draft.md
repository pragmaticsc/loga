# The Language We Train On: A Case for Purpose-Built Substrate Languages in Large Language Model Pre-Training

**A Perspective**

---

## Abstract

Large language models are trained almost exclusively on text produced for human readers — text shaped by millennia of communicative evolution, cultural accident, and biological constraint. We argue that this is a contingent, not necessary, choice. The substrate on which a model is trained is a design variable, and natural language is a poor default: it is irregular, ambiguous, morphologically unpredictable, and tokenizes inefficiently. We propose a research programme in which a small constructed language, designed from the ground up around the computational properties of transformer-based models and byte-pair encoding, is used as the pre-training corpus for a controlled comparison against an English-trained baseline. Drawing on emerging evidence from compositional learning theory, tokenization information theory, and multilingual representation research, we argue that such a language could yield measurably better training efficiency — more learned structure per parameter, at equivalent compute — and that the controlled experiment required to test this has not yet been performed.

---

## 1. The Contingency of English

Every large language model trained today inherits the accidents of English. Its spelling system conflates letters and sounds without regularity ("though," "through," "tough," "cough" share no phoneme-orthography mapping yet share a suffix). Its morphology is partially fusional and partially analytic, meaning inflection is expressed through a mixture of suffixed forms and auxiliary words with no systematic rule. Its syntax permits movement, topicalization, and pragmatic reordering that creates genuine ambiguity resolvable only with world knowledge. Its vocabulary contains thousands of near-synonyms whose distributional differences are subtle and culturally loaded.

These properties are not bugs in human language; they are features that emerged to serve human communicative needs — expressiveness, implicature, social signalling, poetic compression. But they are bugs for a machine learner. A transformer model expends significant representational capacity learning that "ran," "run," and "running" share a root, that "do not" and "don't" are identical in meaning, that "the bank" requires disambiguation based on sentence context. This capacity could instead be spent learning facts about the world.

This observation is not new. The field has responded with a proliferation of tokenization workarounds: byte-pair encoding (BPE) to manage the long tail of surface forms, morpheme-aware variants to better respect linguistic units [1, 2], byte-level models to sidestep tokenization entirely [3, 4]. But all of these are patches on a substrate that was never designed for the task. We propose a more fundamental intervention: redesign the substrate itself.

---

## 2. Evidence That Language Structure Affects Learnability

The claim that a more regular language would be easier for a neural network to learn is not merely intuitive — it now has direct empirical support.

Galke, Ram, and Raviv (2024) trained recurrent neural networks and transformer models on artificial languages varying systematically in their degree of compositional structure [5]. Languages in which meaning was expressed through regular, transparent combinations of primitives were learned faster, generalized better to held-out combinations, and produced more consistent representations across independently trained agents. This result mirrors findings from the cognitive science of language learning: humans also acquire more compositionally regular languages faster and with fewer errors. The neural network, in this respect, behaves like a rational statistical learner — and a rational learner benefits from a regular substrate.

This finding reframes the conlang-for-LLM proposal. It is not speculative; it is an application of a demonstrated principle to the specific setting of large-scale pre-training. The question is not *whether* regularity helps, but *how much* it helps at the scale and architecture of modern language models, and on a realistic pre-training corpus.

---

## 3. The Tokenization Problem Is Deeper Than It Appears

Current practice treats tokenization as a preprocessing detail — an engineering necessity to manage vocabulary size. Byte-pair encoding [6] is the dominant approach: a greedy algorithm that merges the most frequent byte pairs in the training corpus iteratively until a vocabulary of fixed size is reached. BPE is fast and produces reasonable results on English, but it has no linguistic awareness. It will split "unfortunately" into four tokens and merge "New York" into one, based purely on corpus frequency. Morpheme boundaries are violated routinely.

Recent work has shown that this matters, though the evidence is nuanced. MorphBPE [1] and MorphPiece [2] both demonstrate that constraining BPE merges to respect morpheme boundaries is feasible and does not degrade — and in some settings improves — downstream task performance. A large-scale evaluation across 70 languages found, surprisingly, that morphological alignment of tokenizers did not consistently predict downstream performance [7], suggesting that alignment alone is insufficient. But a concurrent information-theoretic analysis argues that this is because compression ratio and linguistic granularity must be *jointly* optimized [8] — a tokenizer that merely aligns to morpheme boundaries without increasing predictability of the token stream gains little.

This points toward the deeper issue: what makes a tokenization scheme good for a language model is not morphological awareness per se, but *predictability*. A token stream in which each token's identity is highly constrained by its context requires fewer bits to encode — and a model that achieves low bits-per-byte has learned more structure per parameter. The right design question is therefore: what properties of a language maximize the predictability of its token stream without sacrificing expressive completeness? This is precisely the design question a purpose-built language can answer, because unlike natural language, every aspect of the design is under the engineer's control.

---

## 4. LLMs Already Develop an Internal Interlingua — and It Is Not Optimal

A striking recent finding deepens the case for deliberate substrate design. Schut et al. (2025) used logit-lens probing and activation steering across multiple multilingual language models processing inputs in French, German, Dutch, and Mandarin [9]. They found that models make semantic decisions in a representation space whose structure is closest to English, regardless of input or output language: the model appears to first resolve the meaning of an utterance in an English-like latent space, then translate to the target language in later layers.

This finding is simultaneously reassuring and troubling. It is reassuring because it shows that transformers are capable of developing an internal representational language that is at least partially language-independent — an emergent interlingua. It is troubling because that interlingua appears to be shaped by English, the dominant language in most pre-training corpora, rather than by any principled design. Google's multilingual NMT work showed that shared representations develop spontaneously when models are trained on multiple translation pairs, enabling zero-shot translation between language pairs never seen in training [10] — an implicit interlingua. But no one designed these representations to be maximally efficient for the model's internal computation.

If models develop an emergent internal language anyway, a natural question follows: could training on a surface language specifically designed to align with the inductive biases of the architecture — regular morphology, predictable syntax, BPE-aligned token boundaries — cause that emergent interlingua to be *better*? A model trained on a regular, unambiguous language might develop internal representations that are more structured, more transferable, and require fewer parameters to achieve equivalent compression. This is the central conjecture of the research programme we propose.

Further support comes from Meta AI's Coconut work (2024), which replaced discrete chain-of-thought tokens with continuous latent vectors fed back as input [11]. Coconut found that language space is not optimal for reasoning: most tokens in standard chain-of-thought sequences enforce surface textual coherence rather than contributing to the reasoning computation. Replacing language tokens with unconstrained latent vectors improved performance on logical reasoning tasks requiring significant planning. This result implies that the discrete language in which a model reasons is a genuine design variable, not an architectural constant — and that natural language may be a suboptimal choice.

---

## 5. What Has Been Done and What Has Not

Several adjacent research directions have approached this problem without arriving at the specific intervention we propose.

Interlingua research in neural machine translation has shown that shared multilingual encoders develop approximately language-independent representations [10, 12, 13]. These representations are useful for zero-shot translation but were not designed to improve the efficiency of the training process itself — they are a means to an end (translation quality), not an end (pre-training efficiency).

Morpheme-aware tokenization work [1, 2] improves the alignment between tokenizer outputs and linguistic units, but operates within the constraint of natural language input. It cannot address the irregularities baked into English morphology itself — only improve the tokenizer's handling of them.

Byte-level models (ByT5 [3], MegaByte [14], the Byte Latent Transformer [4]) sidestep tokenization entirely by operating on raw bytes or dynamically computed patches. These models remove the tokenizer as a source of inefficiency but do not address the deeper irregularity of the language itself. A byte-level model trained on English still must learn that the same morpheme can be spelled multiple ways and that syntactic structure is recoverable only from distributional patterns.

A 2025 preprint (arXiv:2502.04488) provides the closest conceptual precedent [15]: it proposes an AI-centric language system drawing on principles from Esperanto and Lojban, arguing that regularity, unambiguity, and conciseness would reduce computational overhead in transformer architectures. But this paper presents a framework, not an experiment. No controlled comparison has been performed in which an LLM is trained from scratch on a purpose-designed constructed language and compared against an English-trained baseline with identical architecture, parameter count, and compute budget.

**That experiment is what we propose.**

---

## 6. The Proposed Experiment and Its Design Principles

The experimental design is a controlled comparison: two models, identical in architecture and parameter count, trained on the same information (Simple English Wikipedia) encoded in two different surface languages — English and a purpose-built constructed language we call Loga. Bits-per-byte on a held-out validation set is the primary metric, following the evaluation methodology of autoresearch-style automated hyperparameter search [cf. 16]. Both models use custom BPE tokenizers trained on their respective corpora at identical vocabulary sizes.

Loga is designed around the following properties, each motivated by the research reviewed above:

**BPE-aligned morphology.** All roots are exactly four characters in a consonant-vowel-consonant-vowel (CVCV) pattern, with a minimal phoneme inventory of ten consonants and five vowels. High-frequency roots will reliably merge into single BPE tokens, creating a near-logographic token vocabulary for core concepts while preserving compositional structure through systematic suffixation. This addresses the morpheme boundary misalignment documented in [1, 7].

**Agglutinative morphology with invariant roots.** Each grammatical role — case, tense, aspect — is expressed by a fixed suffix appended to an invariant root. The root never changes form. This eliminates the fusional irregularity of English ("go"/"went", "is"/"was"/"be") that forces a model to learn irregular alternations rather than semantic relationships.

**Strict SOV word order with no movement rules.** Syntax is context-free. No topicalization, no wh-movement, no garden-path constructions. This eliminates the syntactic ambiguity that consumes attention capacity in English-trained models.

**A small closed root vocabulary (~300 roots) with productive compounding.** Limiting the core vocabulary ensures that all roots appear frequently enough in the training corpus for their BPE tokens to be well-trained. Galke et al.'s finding [5] that compositionality supports generalization motivates productive compounding over logographic expansion: new concepts are expressed as combinations of known roots, allowing the model to infer meaning compositionally rather than memorizing arbitrary symbol-meaning pairs.

---

## 7. Implications and Limitations

If the hypothesis is supported — if a Loga-trained model achieves lower bits-per-byte than the English baseline at equivalent compute — the implications extend beyond the immediate experiment. It would suggest that the choice of training language is a lever on model efficiency that the field has not yet exploited. It would provide an empirical basis for designing pre-training corpora, intermediate reasoning languages, and potentially the token vocabularies of future architectures.

The primary limitation is translation quality. The training corpus for the Loga model is a machine translation of English Wikipedia produced by an existing LLM. If this translation introduces noise — concepts rendered inaccurately, sentence structure distorted — the model learns a noisy representation of the world rather than the language structure we intend. Back-translation validation using semantic similarity metrics provides a partial check on this, but cannot eliminate the risk. A second limitation is corpus size parity: both models should train on equivalent *semantic* content, but the Loga corpus will differ in byte count from the English corpus. Controlling for this requires careful normalization.

These are methodological challenges, not objections in principle. They are the normal problems of a first controlled experiment, and their solutions are straightforward.

---

## 8. Conclusion

The language a model is trained on is not a fixed background condition. It is an engineering choice, and like all engineering choices, it can be made well or poorly relative to the task. Natural language was not designed for machine learning; it evolved for human communication. The tokenization hacks, architectural extensions, and post-hoc alignment procedures the field has developed to manage language's irregularities are evidence of a mismatch, not a solution to it.

A small but converging body of research now provides principled grounds for the conjecture that a more regular, compositionally transparent training language would produce more efficient learners. What remains is the experiment. The tools exist — autoregressive training at modest scale, automated hyperparameter search, BPE tokenizer training, machine translation for corpus generation — and the compute required is accessible on modern consumer hardware. The question of whether the language we train on matters is both answerable and, we argue, worth answering.

---

## References

[1] MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies. arXiv:2502.00894 (2025).

[2] Jabbar, H. et al. MorphPiece: A Linguistic Tokenizer for Large Language Models. arXiv:2307.07262 (2023).

[3] Xue, L. et al. ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models. *Transactions of the Association for Computational Linguistics* 10 (2022). arXiv:2105.13626.

[4] Meta AI Research. Byte Latent Transformer: Patches Scale Better Than Tokens. arXiv:2412.09871 (2024).

[5] Galke, L., Ram, Y. & Raviv, L. What Makes a Language Easy to Deep-Learn? Deep Neural Networks and Humans Similarly Benefit from Compositional Structure. *Nature Communications* (2024). arXiv:2302.12239.

[6] Sennrich, R., Haddow, B. & Birch, A. Neural Machine Translation of Rare Words with Subword Units. *ACL* (2016).

[7] Evaluating Morphological Alignment of Tokenizers in 70 Languages. arXiv:2507.06378 (2025).

[8] Erdogan, M. et al. An Information-Theoretic Perspective on LLM Tokenizers. arXiv:2601.09039 (2026).

[9] Schut, L., Gal, Y. et al. Do Multilingual LLMs Think In English? *ICLR* (2025). arXiv:2502.15603.

[10] Johnson, M. et al. Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation. *Transactions of the Association for Computational Linguistics* 5 (2017). arXiv:1611.04558.

[11] Meta AI Research. Training Large Language Models to Reason in a Continuous Latent Space (Coconut). *ICLR* (2025). arXiv:2412.06769.

[12] A Neural Interlingua for Multilingual Machine Translation. arXiv:1804.08198 (2018).

[13] Language-Aware Interlingua for Multilingual Neural Machine Translation. *ACL* (2020).

[14] Yu, L. et al. (Meta AI). MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers. *NeurIPS* (2023). arXiv:2305.07185.

[15] Building A Unified AI-centric Language System: Analysis, Framework and Future Work. arXiv:2502.04488 (2025).

[16] Rethinking Tokenization for Rich Morphology: The Dominance of Unigram over BPE and Morphological Alignment. arXiv:2508.08424 (2025).
