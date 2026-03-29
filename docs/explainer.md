# What If AI Had Its Own Language? — A Plain-English Guide to the Loga Project

*This is an informal explainer for the paper "The Language We Train On." No background in AI or linguistics required.*

---

## The Big Idea in One Sentence

AI systems are trained on human language — but human language was designed for humans, not computers. What if we built a tiny, mathematically clean language from scratch specifically for AI, and trained on that instead?

---

## Why Does It Matter What Language an AI Learns From?

When a company trains a large language model — the kind of AI behind tools like ChatGPT — they feed it billions of pages of text. Almost all of that text is in English (or other human languages). The AI learns by trying to predict the next word in every sentence, over and over, billions of times.

Here's the thing: **English is kind of a mess.**

- "Run," "ran," and "running" all mean the same basic action, but look completely different.
- "Bank" can mean a riverbank or a financial institution — the AI has to figure out which from context.
- "Do not" and "don't" are identical in meaning but look different on the page.
- Spelling is all over the place: "though," "through," "tough," and "cough" all end in "-ough" but rhyme with four completely different sounds.

The AI spends enormous effort learning all these irregularities — the exceptions, the weird spellings, the words that mean different things in different contexts. That effort uses up memory and computing power that could instead go toward actually understanding ideas.

This is like hiring a new employee and having them spend their first year memorising all the typos in the company's filing system before they can actually do their job.

---

## So What's the Solution?

The paper proposes: **design a new language specifically for AI training.**

Not a language for humans to speak — nobody has to learn it. It's more like a computer file format for ideas. Think of it like how music can be stored as an MP3 (designed for human ears) or as sheet music notation (designed for musicians to read). This project proposes something like sheet music for AI — a clean, compressed format for knowledge.

The constructed language in this project is called **Loga**.

---

## What Makes Loga Different From English?

Loga is designed around a few simple rules that English violates constantly:

**1. Every word is exactly 3 characters.**
- `ka!` = "the person" (subject of a sentence)
- `Se:` = "sees" (present tense)
- `da"` = "the thing" (object of a sentence)
- So "The person sees the thing." = `ka! da" Se: .`

Compare to English: T-h-e-space-p-e-r-s-o-n-space-s-e-e-s-space-t-h-e-space-t-h-i-n-g-period = 26 characters. Loga uses 13 characters for the same sentence (about half as many).

**2. Roots never change form.**
In English, "go" becomes "went" in the past tense — completely different word. In Loga, you just change the third character: `Go:` (going now) → `Go;` (went) → `Go<` (will go). The root `Go` stays the same. The AI doesn't need to memorise thousands of irregular verb forms.

**3. Word order never changes.**
English lets you say "The cat saw the dog" and "The dog was seen by the cat" — same meaning, different word order. Loga always puts things in the same order: Subject → Object → Verb, no exceptions. The AI always knows where to look for what.

**4. Every character tells you something.**
- First character lowercase? It's a noun (a person, place, or thing).
- First character uppercase? It's a verb (an action).
- Third character from `!` to `/`? Tells you the grammatical role (subject, object, etc.).
- Third character from `:` to `@`? Tells you the tense (past, present, future, etc.).

The AI can figure out whether a word is a noun or a verb from the *first letter alone*, without reading the rest of the sentence. In English, "run," "swim," and "laugh" can all be nouns or verbs depending on context.

---

## What Are the Three Predictions (Conjectures)?

The paper makes three specific predictions about what would happen if you trained an AI on Loga instead of English:

**Conjecture 1: The AI learns faster and more efficiently.**

Think of it this way: if you're teaching someone to solve maths problems, it's easier if the notation is clean and consistent. You'd make faster progress with standard algebra notation than if every textbook used different symbols. The prediction is that the AI will get smarter (in a measurable way) per hour of training when it learns from Loga than when it learns from English.

**Conjecture 2: The AI can be shrunk down more aggressively without losing ability.**

There's a technique called "ternary quantization" that compresses an AI model by replacing every number stored in the model with just one of three values: -1, 0, or +1 (instead of a full floating-point number like 0.7341...). This makes the model much smaller — like compressing a photo from RAW format to a small JPEG. The prediction is that an AI trained on Loga handles this compression better — loses less quality — because a cleaner training language means the model doesn't need as much numerical precision to store what it learned.

**Conjecture 3: The AI develops cleaner internal structure.**

Inside an AI model, there are thousands of units called "attention heads" — you can think of each one as a specialist that looks for a specific pattern. Some heads notice subject-verb relationships; others notice when sentences are questions; others track long-range references. The prediction is that in a Loga-trained AI, these specialists become *really* specialised — each one does exactly one thing — while in an English-trained AI, they have to hedge because English has so many exceptions. And if the specialists are clean, you can prune out the ones doing almost nothing, making the model even smaller.

---

## Why Would This Be Useful?

Right now, running a powerful AI requires expensive servers in data centres. The AI models are huge — billions of numbers that need to be stored and processed.

If all three predictions hold, you could build an AI that's far more capable than its size would suggest, because:
- It learned from a cleaner data source (Conjecture 1)
- You compressed it more aggressively (Conjecture 2)
- Then you pruned out the parts doing nothing (Conjecture 3)

The paper imagines a future system where:

1. A regular large AI (like the ones that exist today) translates your question into Loga.
2. A much smaller, cheaper specialist AI thinks through the problem entirely in Loga.
3. The large AI translates the answer back to English.

The expensive part (understanding messy human language) only happens once. The cheap part (actually reasoning about the problem) is done by a tiny model that could run on your phone, because it only needs to work in clean, simple Loga.

---

## Has Anyone Done This Before?

Not exactly. There have been related efforts:

- People have tried to make better tokenizers for English (the system that chops text into chunks before feeding it to the AI). These help, but they're still working around English's messiness rather than fixing the root cause.
- Researchers have tried training AIs on made-up toy languages to see what they can learn. These experiments are on tiny scales with simple tasks.
- One 2025 paper proposed the idea of an AI-designed language theoretically, but didn't actually run the experiment.

This project is the first to propose actually running the full experiment: take the same knowledge base (Simple English Wikipedia), translate it into a purpose-designed language, train two identical AI models (one on English, one on the translated version), and measure the difference.

---

## Is This Definitely Going to Work?

Honestly — we don't know yet. The paper is upfront about this.

There's a real possibility it doesn't work, for reasons the paper acknowledges:

- **Translation quality**: The Loga corpus is created by an existing AI translating Wikipedia into Loga. If the translations are inaccurate, the model learns noise rather than structure.
- **Vocabulary size**: Loga has a much smaller vocabulary than English. Some of the predicted efficiency gains might just be "smaller vocabulary = easier prediction task," not anything deep about grammatical regularity. That would be a real but less interesting result.
- **Scale**: The experiment uses models with 10–50 million parameters — relatively small. Some techniques only show their advantages at much larger scales.

The paper is careful to describe these limitations and suggests follow-up experiments to disentangle the effects. **A null result — where the AI trained on Loga doesn't do better — would itself be important**, because it would tell us that grammatical regularity alone isn't the bottleneck in AI training.

---

## The Short Version

| Question | Answer |
|----------|--------|
| Why English? | Historical accident — there was lots of it available. |
| What's wrong with English for AI? | Irregular, ambiguous, and wastes the model's capacity on exceptions. |
| What's Loga? | A purpose-built 3-character-per-word language designed to be maximally clean for AI learning. |
| What's being tested? | Whether an AI trained on Loga is more efficient, handles compression better, and develops cleaner internal structure than one trained on English. |
| Has it been done? | No. This paper proposes the experiment and explains why it's worth running. |
| Will it work? | Unknown — and that's why the experiment matters. |
