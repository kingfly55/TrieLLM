## Guiding LLM Generation with Trie-Based Vocabulary Constraints

**Document Version:** 1.1
**Date:** October 26, 2023 (Updated May 8, 2025)

### 1. Introduction & Goal

This document details the technical approach for guiding Large Language Model (LLM) text generation. The primary goal, as outlined in [`problem_spec.md`](llm_control_experiments/TrieLLM/problem_spec.md:1), is to ensure the LLM's output aligns with a learner's current vocabulary, promoting known words and strictly avoiding unknown ones.

We will build upon the existing Python implementation found in [`generate.py`](llm_control_experiments/TrieLLM/generate.py:1) and [`trieLogists.py`](llm_control_experiments/TrieLLM/trieLogists.py:1), which already provides a foundation for Trie-based constrained decoding using a custom `LogitsProcessor`.

Our objective is to refine and extend this system to:
1.  **Strictly adhere** to a dynamic list of "known" vocabulary items.
2.  **Probabilistically promote or avoid** words within this known vocabulary based on learner confidence scores (0-1).

### 2. System Overview & Existing Implementation

The current system ([`generate.py`](llm_control_experiments/TrieLLM/generate.py:1), [`trieLogists.py`](llm_control_experiments/TrieLLM/trieLogists.py:1)) integrates a vocabulary store (Trie) with the LLM's generation process via a custom `LogitsProcessor`.

*   **Learner Profile (To Be Enhanced):** Currently, vocabulary is loaded from [`allowed_sequences.json`](llm_control_experiments/TrieLLM/allowed_sequences.json). This will be expanded to include confidence scores for each word/phrase.
*   **Vocabulary Manager (Trie):**
    *   The [`TrieMachine`](llm_control_experiments/TrieLLM/trieLogists.py:11) class in [`trieLogists.py`](llm_control_experiments/TrieLLM/trieLogists.py:1) constructs a Trie from tokenized sequences.
    *   Terminal nodes implicitly mark word ends. This will be enhanced to store confidence scores.
*   **Constrained Generation Engine:**
    *   Utilizes a Hugging Face Transformer model ([`generate.py`](llm_control_experiments/TrieLLM/generate.py:31)).
    *   The [`TrieLogitsProcessor`](llm_control_experiments/TrieLLM/trieLogists.py:30) modifies logits at each step.
    *   Supports beam search ([`generate.py`](llm_control_experiments/TrieLLM/generate.py:51)).

**Existing Flow ([`generate.py`](llm_control_experiments/TrieLLM/generate.py:24)):**
1.  Loads allowed sequences ([`load_allowed_sequences`](llm_control_experiments/TrieLLM/generate.py:10)).
2.  Encodes sequences into token IDs ([`encode_sequences`](llm_control_experiments/TrieLLM/generate.py:16)).
3.  Initializes `TrieMachine` and `TrieLogitsProcessor`.
4.  Generates text using `model.generate()` with the custom processor.

### 3. Core Components & Planned Enhancements

#### 3.1. Vocabulary Representation: Trie with Confidence Scores

*   **Current Structure:** The [`Trie`](llm_control_experiments/TrieLLM/trieLogists.py:6) class uses a dictionary for children, representing token IDs.
*   **Enhancement:**
    *   Modify the Trie nodes or a parallel structure to store a confidence score (float 0.0-1.0) at terminal nodes (i.e., end of a known word).
    *   The input `allowed_sequences.json` (or a similar data source) will need to be updated to include these scores.

#### 3.2. Constrained Generation Mechanism (`TrieLogitsProcessor`)

*   **Current Logic ([`TrieLogitsProcessor.__call__`](llm_control_experiments/TrieLLM/trieLogists.py:37)):**
    *   Identifies the current state in the Trie based on the generated `input_ids` after a `last_token` (e.g., ':').
    *   Masks logits, allowing only valid next tokens from the Trie.
    *   Sets scores for invalid paths to a very low value.
    *   Includes logic to handle cases where `len(next_states) < num_beams`.
*   **Enhancements for Soft Constraints (Confidence-Based Biasing):**
    1.  **Retrieve Confidence:** When a token could complete a word in the Trie, fetch its confidence score.
    2.  **Calculate Bias:** Implement a `bias_function(confidence_score, target_confidence_range, strength_factor)`.
        *   Example: `bias = strength * (confidence_score - 0.5) * 2` (maps 0-1 confidence to `[-strength, +strength]` bias).
    3.  **Apply Bias:** Add this calculated bias to the logits of corresponding valid tokens.
        `scores[i, token_id] += bias`.
    4.  The `last_token` dependency might need to be made more flexible or configurable for general conversational use.

### 4. Experimentation & Development Path (Building on Existing Code)

1. **Benchmark performance:**
    * We will need a way to load n english words into allowed_sequences. We should just grab the n most common english words.
    * Then, we will need a way to benchmark the performance.

2.  **Refine Vocabulary Input:**
    *   Modify [`allowed_sequences.json`](llm_control_experiments/TrieLLM/allowed_sequences.json) (or create a new format) to include words and their confidence scores.
    *   Update [`load_allowed_sequences`](llm_control_experiments/TrieLLM/generate.py:10) and Trie construction in [`TrieMachine`](llm_control_experiments/TrieLLM/trieLogists.py:11) to handle these scores.

3.  **Implement Confidence Score Storage in Trie:**
    *   Adjust the [`Trie`](llm_control_experiments/TrieLLM/trieLogists.py:6) node structure or [`TrieMachine`](llm_control_experiments/TrieLLM/trieLogists.py:11) to associate confidence scores with terminal token sequences.

4.  **Enhance `TrieLogitsProcessor` for Soft Constraints:**
    *   Integrate the confidence score retrieval and bias calculation logic as described in section 3.2.
    *   Make the `bias_function` and `strength_factor` configurable.

5.  **Testing & Evaluation:**
    *   **Baseline:** Use existing [`generate.py`](llm_control_experiments/TrieLLM/generate.py:1) to test hard constraints (vocabulary filtering).
    *   **Soft Constraint Tests:**
        *   Create test vocabularies with varied confidence scores.
        *   Design prompts to observe if the model preferentially uses words with targeted confidence levels (e.g., promote high confidence, avoid low confidence).
        *   Tune biasing parameters.
    *   **Evaluate Fluency and Latency:** Measure impact on output quality and generation speed.

6.  **Iterate on `last_token` / Context Handling:**
    *   The current reliance on a specific `last_token` (e.g., ':') to activate Trie-based filtering ([`TrieLogitsProcessor`](llm_control_experiments/TrieLLM/trieLogists.py:42)) is suitable for specific generation tasks but may need generalization for broader conversational contexts. Explore ways to apply constraints more dynamically based on the ongoing dialogue or learning objectives.

This iterative approach, leveraging the existing codebase, will allow us to progressively build towards a robust system for confidence-aware, vocabulary-constrained LLM generation.