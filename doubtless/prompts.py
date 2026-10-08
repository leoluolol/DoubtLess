"""Instructions appended to the small model's two copies of the user prompt."""

BRIEF = (" Be concise: go straight to the answer with the fewest steps. Once you have a"
         " result, do NOT re-derive, re-verify, or repeat any calculation, and do not"
         " second-guess it.")

VERBOSE = (" Be extremely thorough: double-check every step, re-verify each calculation in"
           " more than one way, and exhaustively consider alternative cases before you"
           " commit to a final answer.")
