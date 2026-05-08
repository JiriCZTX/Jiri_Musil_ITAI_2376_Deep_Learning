# A04 — Analyzing *Arrival* Through the Lens of NLP

**Course:** ITAI 2376 — Deep Learning Artificial Intelligence
**Author:** Jiri Musil

## Problem Statement

The 2016 film *Arrival* is, on its surface, a science fiction movie about an alien encounter. Strip away the spaceships and what remains is a meticulous depiction of a low-resource translation problem under high political stakes. This assignment uses the film as a case study to map cinematic communication breakdowns onto the real challenges that NLP systems face: tokenization without word boundaries, word-sense ambiguity in high-consequence settings, pragmatics, idioms, sarcasm, cultural variation, and domain shift.

## Approach

- Identify the linguistic and computational problems Dr. Banks (the protagonist linguist) confronts and trace each one to its NLP analog.
- Cover four core NLP challenges illustrated by the film:
  1. **Low-resource learning and grounding** — Banks has no dictionary or grammar. She starts from concrete labels and observable context, mirroring the principle that meaning needs grounding, not just symbols.
  2. **Segmentation, representation, and structure** — heptapod logograms encode a whole thought in one circular mark, raising the same "what are the basic units?" question that drives tokenization research.
  3. **Ambiguity and word sense under pressure** — the "offer weapon" mistranslation crisis maps directly to word-sense ambiguity in machine translation, with the added human layer of fear and prior bias amplifying the wrong reading.
  4. **Pragmatics, idioms, sarcasm, and cultural variation** — the heptapod nicknames "Abbott and Costello," and the way different countries interpret the same translations, illustrate the cost of assuming meaning is universal when background assumptions are not shared.
- Map the team's workflow onto three eras of NLP methodology: rule-based (controlled protocols), statistical (pattern discovery from a growing corpus), and neural / hybrid toolkits.
- Propose an "innovative perspective": a practical NLP system should treat translation as an interactive, risk-aware protocol — return the top-3 interpretations with confidence scores, refuse to finalize when stakes are high and confidence is low, and include a cross-cultural evaluation loop. This mirrors Banks's strategy and is also how I would design a high-stakes language pipeline at work.

## Files

- `A04_Arrival_Through_NLP_Lens.pdf` — full written analysis (8 pages, 8 references)

## Learning Outcomes

- Reading *Arrival* as an NLP case study made the standard course topics — tokenization, ambiguity, pragmatics, domain adaptation — concrete in a way that a textbook chapter could not.
- The most useful takeaway was the deployment lesson: translation outputs do not stay on a whiteboard. They drive decisions, emotions, and politics. NLP evaluation usually optimizes for average accuracy, but in high-stakes settings the worst-case behavior is what matters.
- This perspective directly shaped my final project — the multi-agent workforce intelligence system has explicit uncertainty handling, refusal behavior on low-confidence high-stakes verdicts, and human-in-the-loop escalation, all of which trace back to ideas I worked through in this assignment.

## References

The PDF cites Brower (LARB), Coon's manuscript on the linguistics of *Arrival*, Jurafsky and Martin's *Speech and Language Processing* (3rd ed., 2026), WIRED articles, and academic sources on idioms, neural NLP, and unsupervised domain adaptation. Full bibliography in the PDF.
