# Loga: LLM-Optimized Constructed Language — Formal Specification

**Version**: 0.2
**Design goal**: A constructed language whose surface form maximally aligns with Byte-Pair Encoding (BPE) tokenization, eliminates syntactic ambiguity, and expresses meaning compositionally so that small language models can achieve better bits-per-byte than equivalent English-trained models.

**v0.2 changes**: Replaced the 20-character CVCV alphabet with a 95-character full printable ASCII alphabet. Information-theoretic analysis shows this is the density optimum within the 1-byte-per-character constraint imposed by UTF-8 encoding.

---

## 1. Design Principles

| Principle | Rationale |
|---|---|
| Full printable ASCII alphabet (95 chars) | Maximum information density per byte within the 1-byte UTF-8 range: 6.57 bits/byte vs. 4.70 for lowercase-only |
| Semantic role encoded in character class | First byte of every token carries its syntactic function; transformers can determine role with zero context |
| 2-character roots | 26×62 = 1,612 noun roots + 1,612 verb roots (3,224 usable) in 2 bytes; >1.3× more expressive than prior CVCV design at half the byte cost |
| 1-character suffixes | Case (nouns) or tense/aspect (verbs) each cost exactly 1 byte; full inflected word = 3 bytes total |
| Strict SOV word order | Eliminates attachment ambiguity; parser never needs lookahead |
| Agglutinative morphology | Root + case (nouns) or Root + tense (verbs); each suffix is compositionally appended, never fused |
| No homographs | Every assigned root maps to exactly one meaning; proper nouns share the uppercase namespace with verbs but are unambiguously distinguished by their suffix type |
| Context-free parseable | Sentence structure derivable from a finite EBNF grammar with no heuristics |

---

## 2. Why 95 Printable ASCII Characters

### 2.1 Information Density Analysis

UTF-8 encodes codepoints above U+007F as 2 or more bytes, regardless of how visually compact the character appears. This creates a hard efficiency ceiling at the ASCII boundary:

```
Alphabet               Chars   Bytes/char   Bits/byte   2-char roots
---------------------  ------  -----------  ----------  ------------
Lowercase ASCII (v0.1)     20          1.0        4.32           400
Lowercase + upper ASCII    52          1.0        5.70         2,704
Full printable ASCII       95          1.0        6.57         9,025  ← optimum
Latin Extended (ü, é…)    256          2.0        4.00            —
2-byte Unicode block     1920          2.0        5.45            —
CJK ideographs          20000          3.0        4.76            —
```

CJK characters — despite encoding thousands of concepts visually — are *less* byte-efficient than 2-character ASCII roots because the 3-byte UTF-8 encoding cost cancels the representational gain. The 95-character printable ASCII set is the information-theoretic optimum given the UTF-8 encoding layer.

### 2.2 Practical Gains Over v0.1

| Metric | v0.1 (CVCV) | v0.2 (95-char ASCII) | Improvement |
|--------|-------------|----------------------|-------------|
| Root vocabulary | 2,500 | 3,224 (1,612 noun + 1,612 verb) | >1.3× |
| Bytes per root | 4 | 2 | 50% fewer |
| Bytes per inflected word | 4–6 | 3 | consistent |
| Bits/byte in root position | 4.32 | 6.57 | +52% |

---

## 3. Character Set and Semantic Partitioning

All 95 printable ASCII characters (U+0020 through U+007E) are used. The space character (U+0020) serves as the word boundary; the remaining 94 visible characters (`!` through `~`, U+0021–U+007E) serve as root characters and grammatical suffixes.

**Critical design feature**: character class encodes grammatical role. A transformer's attention head can determine the syntactic function of any character from its byte value alone — no context required.

### 3.1 ASCII Range Assignments

```
Range         Chars     Decimal    Count   Role
------------- --------- ---------  ------  --------------------------------
a–z           a…z       97–122        26   Noun root (first char, lowercase)
A–Z           A…Z       65–90         26   Verb root (first char, uppercase)
0–9           0…9       48–57         10   Numeric literals
Second chars: any of a–z, A–Z, 0–9       62   Root second character (52 combos used)
!–/           !"/…/    33–47         15   Case suffixes  (see §4.1)
:–@           :;<=>?@  58–64          7   Tense / aspect markers (see §4.2)
[–`           [\]^_`   91–96          6   Quantifiers and particles
{–~           {|}~     123–126         4   Compound / derivation markers
```

**Word-level structure** (always 3 bytes, space-delimited):

```
[C₁ C₂][suffix]
 ──────  ──────
 root    case (nouns) OR tense (verbs)

C₁: first root char  → encodes noun vs. verb class (lowercase = noun, uppercase = verb)
C₂: second root char → selects specific root within class
suffix: one character — case marker for nouns (§4.1), tense marker for verbs (§4.2)
```

No word carries both a case suffix and a tense marker simultaneously.
Examples: `ka!` = noun-root `ka`, nominative case `!`; `Se:` = verb-root `Se`, present tense `:`

**Disambiguation: `.` as adverbial suffix vs. sentence terminator.** The adverbial case suffix `.` (§4.1) always appears as the third byte of a space-delimited 3-byte word. The sentence-final `.` always follows a complete verb word whose suffix is a tense marker from the `:`–`@` range. These are unambiguous by local structure: verbs cannot take case suffixes, so a `.` after a tense-marked verb is always sentence-final.

---

## 4. Morphology

### 4.1 Case Suffixes (1 character, from `!`–`/` range)

| Char | Decimal | Case | Semantic role |
|------|---------|------|---------------|
| `!` | 33 | Nominative | Subject of verb |
| `"` | 34 | Accusative | Direct object |
| `#` | 35 | Genitive | Possessor / of-relation |
| `$` | 36 | Dative | Indirect object / recipient |
| `%` | 37 | Locative | Location (at / on / in) |
| `&` | 38 | Lative | Direction (toward / to) |
| `'` | 39 | Ablative | Source (from) |
| `(` | 40 | Instrumental | Means / tool |
| `)` | 41 | Comitative | Accompaniment (with) |
| `*` | 42 | Causative | Cause / because of |
| `+` | 43 | Benefactive | For the benefit of |
| `,` | 44 | Comparative | More than |
| `-` | 45 | Adjectival | Modifies preceding noun |
| `.` | 46 | Adverbial | Modifies verb |
| `/` | 47 | Vocative | Direct address |

Nouns always carry exactly one case suffix character. No case stacking.

### 4.2 Tense and Aspect Markers (1 character, from `:`–`@` range)

| Char | Role |
|------|------|
| `:` | Present |
| `;` | Past |
| `<` | Future |
| `=` | Perfective (completed) |
| `>` | Imperfective (ongoing / habitual) |
| `?` | Interrogative (marks question verb) |
| `@` | Conditional |

Verbs always carry exactly one tense/aspect character. Combined aspect+tense is expressed lexically via an auxiliary verb, not additional suffixation.

### 4.3 Quantifiers and Particles (from `[`–`` ` `` range)

| Char | Meaning |
|------|---------|
| `[` | All, every |
| `\` | Some, a few |
| `]` | One (indefinite) |
| `^` | None, zero |
| `_` | Negation particle (precedes word) |
| `` ` `` | Subordinate clause introducer ("that") |

### 4.4 Derivation Markers (from `{`–`~` range)

| Char | Meaning |
|------|---------|
| `{` | Compound joiner (joins two roots into one concept) |
| `\|` | Agent nominalizer (root → "one who does") |
| `}` | Instrument nominalizer (root → "thing used to do") |
| `~` | Abstract nominalizer (root → "the act / quality of") |

---

## 5. Vocabulary System

### 5.1 Root Structure

All roots are exactly **2 ASCII characters**. The first character determines grammatical class:
- Lowercase first character → **noun root**
- Uppercase first character → **verb root**
- Digit first character → **number literal**

The second character can be any of a–z, A–Z, 0–9 (62 choices), giving:
- Noun roots: 26 × 62 = **1,612 possible**
- Verb roots: 26 × 62 = **1,612 possible**

We use a constrained subset of ~300 noun roots and ~200 verb roots for core vocabulary, ensuring every root appears many thousands of times in the training corpus (critical for BPE token stability).

### 5.2 Core Vocabulary (representative sample)

**Pronouns** (noun roots):

| Root | English |
|------|---------|
| `mi` | I / me |
| `tu` | you (singular) |
| `si` | he / she / it |
| `ma` | we |
| `na` | you (plural) |
| `sa` | they |

**Core nouns**:

| Root | English |
|------|---------|
| `ka` | person, human |
| `ku` | water |
| `ge` | fire |
| `to` | time |
| `se` | place, world |
| `li` | life |
| `bo` | city |
| `pa` | idea, concept |
| `da` | thing, object |
| `ne` | name, word |
| `wi` | what, who, which (interrogative pronoun) |

**Core verbs** (uppercase first char):

| Root | English |
|------|---------|
| `Be` | be, equal (copula) |
| `Go` | go, move |
| `Se` | see, observe |
| `Sa` | say, speak |
| `Ma` | make, create |
| `Gi` | give |
| `Ta` | take |
| `Ea` | eat |
| `Sl` | sleep |
| `Th` | think |
| `Kn` | know |
| `Wa` | walk |
| `Si` | sit |

**Numbers**: digits 0–9 as literal characters; compound numbers concatenated:
- `42` = forty-two, `100` = one hundred

### 5.3 Compounds

Two roots joined with `{`. The grammatical class of the compound is determined by its **first character**: lowercase first char = noun compound; uppercase first char = verb compound. The compound takes a case suffix (noun) or tense marker (verb) accordingly:
- `ku{Ma!` = water-make, nominative (irrigator) — noun compound, `ku` has lowercase first char
- `ge{se%` = fire-place, locative (at the volcano) — noun compound
- `pa{ka!` = idea-person, nominative (philosopher) — noun compound
- `Ku{ma:` = Water-move, present (irrigate) — verb compound using uppercase-first verb root

### 5.4 Proper Nouns

Transliterated into Loga phonotactics using available root characters, capitalized on first character. Proper nouns must always be fully inflected with a case suffix — the bare root is not a valid word form:
- "England" → `En!` (nominative)
- "Paris" → `Pa!` (nominative) — `Pa` shares the uppercase namespace with potential verb roots, but the case suffix `!` (from the `!`–`/` range) unambiguously marks it as a proper noun, not a verb
- "Albert Einstein" → `Ab! Es!` (two proper noun tokens)

---

## 6. Syntax

### 6.1 Basic Word Order: SOV

```
SUBJ! OBJ" VERB:
```

Every declarative sentence: subject (nominative `!`) → object (accusative `"`) → verb (tense marker). No exceptions, no movement.

### 6.2 Intransitive Sentences

```
SUBJ! VERB:
```

### 6.3 Copular Sentences

The copula verb root is `Be`. Identity, classification, property ascription:

```
mi! da" Be:     = I am a thing.
ka! bo% Be;     = The person was in the city.
```

### 6.4 Adjectives

Adjective roots take the adjectival case suffix `-` and immediately follow the noun they modify:

```
ka! bi- da" Se:
= The big person sees the thing.
  (person-SUBJ big-ADJ thing-OBJ see-PRES)
```

### 6.5 Embedded Clauses

Subordinate clauses introduced by `` ` `` (backtick), terminated by space before the main verb:

```
mi! ` ka! da" Se; " Kn:
= I know that the person saw the thing.
  I-SUBJ COMP person-SUBJ thing-OBJ see-PAST OBJ know-PRES
```

### 6.6 Questions

**Yes/no questions**: replace the tense marker with `?` on the main verb. Note that explicit tense is lost and must be inferred from context — this is a deliberate design tradeoff favouring simplicity.

```
mi! da" Se?     = Do I see / Did I see / Will I see the thing?
```

**Wh-questions**: use the interrogative pronoun `wi` (what/who/which) with the appropriate case suffix. The verb retains its normal tense marker — tense is not lost.

```
wi! da" Se:     = Who sees the thing?       (wi = nominative, verb present)
mi! wi% Go:     = Where am I going?         (wi = locative, verb present)
```

### 6.7 Negation

Precede the target word with the negation particle `_`:

```
mi! da" _Se:    = I do not see the thing.
_mi! da" Se:    = Not I (but someone else) sees the thing.
```

---

## 7. EBNF Grammar

```ebnf
sentence         ::= declarative | question
declarative      ::= noun_arg+ decl_verb_phrase SPACE "."
question         ::= noun_arg+ quest_verb_phrase

noun_arg         ::= (quantifier SPACE)? (noun | proper_noun | clause_arg)
                     (SPACE adjective)* (SPACE SPACE)?
noun             ::= noun_root case_suffix
noun_root        ::= [a-z] [a-zA-Z0-9]
proper_noun      ::= proper_noun_root case_suffix
proper_noun_root ::= [A-Z] [a-zA-Z0-9]
case_suffix      ::= "!" | '"' | "#" | "$" | "%" | "&" | "'" | "(" | ")" | "*" | "+" | "," | "-" | "." | "/"

clause_arg       ::= "`" noun_arg+ decl_verb_phrase SPACE case_suffix
                   (* subordinate clause used as a syntactic argument;
                      case_suffix is standalone, marking the clause's role *)

adjective        ::= (noun_root | verb_root) "-"
adverb           ::= (noun_root | verb_root) "."

decl_verb_phrase  ::= ("_" SPACE)? verb decl_tense
quest_verb_phrase ::= ("_" SPACE)? verb "?"
verb              ::= verb_root
verb_root         ::= [A-Z] [a-zA-Z0-9]
decl_tense        ::= ":" | ";" | "<" | "=" | ">" | "@"
tense_marker      ::= decl_tense | "?"

quantifier       ::= "[" | "\" | "]" | "^"

compound         ::= root "{" root
root             ::= noun_root | verb_root

SPACE            ::= " "
```

Notes:
- `proper_noun_root` and `verb_root` share the pattern `[A-Z][a-zA-Z0-9]`; suffix type distinguishes them — case suffixes (`!`–`/`) mark proper nouns, tense markers (`:`–`@`) mark verbs.
- `clause_arg` is the one structural exception to the 3-byte word rule: the case suffix following a subordinate clause is a standalone token marking the clause's syntactic role.
- Yes/no questions use `?` as the verb's tense marker (losing explicit tense, which must be inferred from context); wh-questions use the interrogative pronoun `wi` with a normal tense marker.
- Negation particle `_` is space-delimited before its target word.

This grammar is **context-free**. Every syntactic role is recoverable from local character-class information without lookahead or world knowledge.

---

## 8. Sample Sentences (v0.2)

### 8.1 "The person sat in the city."

```
ka!  bo%  Si; .
person-SUBJ  city-LOC  sit-PAST  .
```

### 8.2 "Water is life."

```
ku!  li"  Be: .
water-SUBJ  life-OBJ  be-PRES  .
```

### 8.3 "I know that the person sat in the city."

```
mi!  ` ka!  bo%  Si; "  Kn: .
I-SUBJ  COMP  person-SUBJ  city-LOC  sit-PAST  OBJ  know-PRES  .
```

### 8.4 "The philosopher will not see the beautiful world."

```
pa{ka!  se"  bi-  _Se< .
philosopher-SUBJ  world-OBJ  beautiful-ADJ  NEG-see-FUT  .
```

### 8.5 "Some people will go toward the city."

```
\ ka!  bo&  Go< .
some  person-SUBJ  city-LAT  go-FUT  .
```

---

## 9. BPE Tokenization Analysis (v0.2)

### 9.1 Expected Token Structure

With BPE vocab=8192 trained on Loga text:

| Form | Example | Bytes | Expected BPE tokens |
|------|---------|-------|---------------------|
| Inflected noun | `ka!` | 3 | 1 (very frequent) |
| Inflected verb | `Si;` | 3 | 1 (very frequent) |
| Compound | `pa{ka!` | 6 | 2 (`pa{ka` + `!`) |
| Sentence boundary | `. ` | 2 | 1 |

### 9.2 Predicted Efficiency

With ~300 noun roots × 15 cases = 4,500 inflected noun forms, each appearing thousands of times in the training corpus, BPE at vocab=8192 will learn nearly all of them as single tokens. The token stream then approaches the ideal of **one token per semantic+grammatical unit**.

A Loga inflected word is 3 bytes/token by construction.

Predicted efficiency gain: a measurable reduction in tokens for equivalent semantic content, compared to English BPE at the same vocabulary size. The specific magnitude is an empirical question.

### 9.3 Theoretical Ceiling

The information-theoretic floor for a corpus with this structure: each word is 3 bytes (24 bits). Since nouns take case suffixes and verbs take tense markers (never both simultaneously), the information per word is:

- **Nouns**: log₂(1,612 roots × 15 cases) ≈ log₂(24,180) ≈ 14.6 bits in 24 bits → **~61% utilization**
- **Verbs**: log₂(1,612 roots × 7 tenses) ≈ log₂(11,284) ≈ 13.5 bits in 24 bits → **~56% utilization**

English achieves roughly 40–55% (Shannon, 1951; estimated from natural language BPB measurements).

---

## 10. Design Tradeoffs and Risks

| Risk | Mitigation |
|------|------------|
| ASCII punctuation as grammar is visually confusing | Human readability is not the goal; BPE sees only bytes |
| 95-char alphabet may not all be preserved in HTML/JSON | Corpus is stored as raw UTF-8; problematic chars (`"`, `\`, `<`) escaped in JSON output layer only |
| Root collision (e.g., `Pa` for proper noun vs. `pa` for "idea") | Case-sensitivity (uppercase first char = verb, lowercase = noun) eliminates collision; proper nouns use uppercase |
| Back-translation validation harder to evaluate for this system | Use embedding-based semantic similarity (sentence-transformers), not surface string matching |

---

## 11. File Conventions

- **Encoding**: UTF-8 (all characters are single-byte ASCII)
- **Word delimiter**: space (U+0020)
- **Sentence delimiter**: standalone `.` token after the verb (declarative sentences only). Interrogative sentences have no sentence-final delimiter — they are identified by the `?` tense marker on the final verb.
- **Paragraph delimiter**: `\n\n` (blank line)
- **File extension**: `.loga`
- **Tokenizer vocab size**: 8,192 (matches autoresearch-mlx default)
