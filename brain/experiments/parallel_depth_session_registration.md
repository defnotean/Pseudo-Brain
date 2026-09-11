# Token session for the parallel-depth POMDP core

Prepare an opt-in session adapter distinct from the legacy role-addressed agent.
Persist exactly the ordinary4096-byte state, model reference and immutable
initial task pointer prompt. No cached logits, prior pointer alignments,
generated-token history or observations may influence later model calls.
Eight logical layers packed as high/low components are not sixteen semantic
thought slots. Return generated tokens to the caller as output only.

Start each independent task with fresh parallel prefill. Consume new observation
chunks and explicit response prefixes through parallel carried-state ingestion.
Use greedy autoregressive generation without output repair or repetition rules.
Every selected token, including EOS, must enter state exactly once. At the token
limit, consume the final returned token without fabricating EOS; report that
output is truncated. The caller must not equate truncation with a valid action.

First validate on Colab CPU with CUDA hidden and one thread: multiple alternating
observations/actions against serial execution, pointer on/off, lengths31/257,
EOS state transition followed by observation, fresh-task isolation, prompt and
state export immutability, and invalid input rejection before state changes.
Require full-vocabulary logit parity1e-6, decoded-state parity1e-12 and4096 bytes.
Use seed1907 and reduced width32/vocab128 with the actual eight-layer model.

The adapter performs no environment actions, inserts no heuristic repair phase,
and makes no task-success judgments. Native trained multi-turn validation and
an isolated tool environment with external task validation remain required
before autonomous software benchmark claims. Existing agents, frozen decoders
and V5 training are unchanged.
