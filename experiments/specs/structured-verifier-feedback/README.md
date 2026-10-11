# Structured verifier feedback

Status: draft. This is a concrete experimental contract awaiting a chosen model,
prompt/tool-schema identities, and frozen repair corpus.

Compare raw verifier output with an information-equivalent typed adapter carrying
failure location, observed state, required invariant, admissible alternatives,
and an evidence handle. Keep task order, tool powers, maximum attempts, and
token budget fixed. The primary measure is completion within three attempts;
secondary measures are invalid edits, verifier calls, tokens, elapsed time, and
wrong-location repairs. Review every discordant pair manually.

Do not preregister until the selected provider/model, parameters, fixtures, and
adapter output digest are frozen.

Research packet: [`model-dependent-harness`](../../../docs/research/papers/model-dependent-harness/).
