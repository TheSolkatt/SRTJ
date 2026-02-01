% =========================================
% SRTJ Symbolic Reasoning Engine (v1.0)
% =========================================

% --- 1. Define Search Space ---
% Select between min_k and max_k rules from the available set.
% min_k/max_k are provided by Python at runtime.
min_k { selected(R) : available_rule(R) } max_k.

% --- 2. Define Matching Logic ---
% Define Synergy: A combination is synergistic if one rule provides a Persona
% and another provides a Format (encourages "who am I" + "how to output").
synergy(R1, R2) :- has_persona(R1, _), has_format(R2, _).

% --- 3. Hard Constraints (Must hold) ---
% Attribute-Dimension Exclusion:
% If two different rules share the same exclusive category, the pair is illegal.
% exclusive_category/1 facts are provided by Python at runtime.

% Soft mutual exclusion (high penalty to avoid UNSAT).
:~ selected(R1), selected(R2), R1 < R2,
   has_attr(R1, Cat, _), has_attr(R2, Cat, _),
   exclusive_category(Cat). [1000@5, R1, R2, Cat]

% Heuristic conflict: avoid multiple tone-defining rules at once.
% :- selected(R1), selected(R2), R1 != R2,
%    has_dim(R1, "tone"), has_dim(R2, "tone").

% Soft penalty for multiple tone rules (prefer fewer tone clashes, but allow).
:~ selected(R1), selected(R2), R1 < R2,
   has_dim(R1, "tone"), has_dim(R2, "tone"). [10@1, R1, R2]

% Logical conflicts across dimensions (e.g., aggressive tone with appeal_to_pity strategy).
:- selected(R1), selected(R2), R1 != R2,
   has_attr(R1, "tone", "aggressive"),
   has_attr(R2, "strategy", "appeal_to_pity").

% Exact banned combination (do not match the same set again).
:- banned_size(S, N),
   N = #count { R : selected(R), banned_set(S, R) },
   N = #count { R : selected(R) }.

% --- 4. Optimization / Preferences ---
% Maximize total score and synergy first, then penalize size/irrelevance, then buffer.

% [Priority 4] Maximize total score (sum of selected rule scores).
#maximize { S@4, R : selected(R), score(R, S) }.

% [Priority 4] Synergy bonus: persona + format compatibility (facts provided by Python).
#maximize { 50@4, R1, R2 : selected(R1), selected(R2), synergy(R1, R2), R1 != R2 }.

% [Priority 2] Penalize more rules to encourage "less but better".
:~ selected(R). [rule_count_penalty@2, R]

% [Priority 1] Prioritize Buffer (Hot/New Rules): Penalize picking non-buffer rules.
% Meaning: If relevance is equal, pick the one from Layer 2 (Buffer).
:~ selected(R), not is_from_buffer(R). [50@1, R]

% --- 5. Diversity Constraint (Optional) ---
% Avoid picking multiple rules with the exact same 'strategy' predicate to ensure variety.
% (Requires predicates to be passed in).
% :- selected(R1), selected(R2), R1 != R2, has_predicate(R1, "strategy", V), has_predicate(R2, "strategy", V).

% --- Output ---
#show selected/1.
