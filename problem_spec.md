## Preface

The primary challenge in creating an effective AI language tutor is managing the learning data: **what vocabulary and grammatical structures should be tracked, and how?** This will differ by language; for instance, Spanish (our initial focus) requires tracking words and conjugations, while agglutinative languages might focus on morphemes. The system must adapt to each language's unique structure.

## The Core Problem: Personalized Language Learning

Our goal is to build a language learning system where a user interacts with an LLM agent. This system should emulate a skilled teacher—adaptive, aware of learner progress, and capable of guiding conversations toward specific learning objectives.

-   **LLM Agent Role**: Handles natural conversation and answers questions.
-   **Limitation**: LLMs alone cannot define learning objectives or progression. They require structured backend logic and a "knowledge bank" of learner data (mastered vocabulary, grammar) to steer interactions effectively.

The system relies on **interactive, conversation-driven learning**. Progress can be tracked implicitly (through usage) or explicitly (e.g., quizzes). A key requirement is that the LLM’s output must align with the learner’s current proficiency level.

## Challenge: Aligning LLM Output with Learner Level

A significant hurdle is ensuring the LLM uses vocabulary appropriate for the learner. If the LLM introduces too many unknown words or complex grammar, it can overwhelm the user. Hardcoding "allowed" words is impractical.

The desired solution involves dynamically adjusting the LLM's output to:
1.  **Prioritize** words the user knows, weighted by a confidence score reflecting their mastery.
2.  **Strictly avoid** entirely unfamiliar words.

This control should be probabilistic for preference (favoring known words) but deterministic for exclusion (blocking unknown words). The technical approach for achieving this, centered around constrained decoding using a Trie, is detailed in the [`Approach.md`](llm_control_experiments/TrieLLM/Approach.md) document and has an initial implementation in the accompanying Python code.

## Key Considerations for Vocabulary Management

-   **Representing Learner Vocabulary**: We need an efficient way to store the learner's known vocabulary and associated confidence scores. This dataset could become large (e.g., 20,000 words for advanced learners).
-   **Scalability**: Strategies to manage large vocabularies might include:
    -   Focusing on a subset of vocabulary (e.g., CEFR A1/A2 levels, recently learned words, target words for the current lesson).
    -   Potentially categorizing words (nouns, verbs, etc.) if it offers processing efficiencies, though this adds complexity.

The current implementation provides a foundation for Trie-based vocabulary management, which we will build upon.
