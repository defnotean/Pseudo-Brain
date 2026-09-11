# Target-line ordering probe v1

The completed real-prefix collection produced0/32 expected target files for each
actor. The procedural policy training prompt ended with Target, while MBPP adds
public interface declarations afterward. Test only this ordering difference.

Use the first8 cases of the already frozen32-case training selection from the
completed behavior collection. Keep original responses as the baseline, bound to
pair report0140ee38b7665ef8a0f9316b3f3f430a8ab44688e04a59775c1b5ce5ae328171 and
every raw-attempt hash. Both unchanged completed policy-r2 checkpoints must match
their original hashes. No new development or benchmark cases are used.

Move the one exact existing `Target: <filename>\n` line to the end of the initial
prompt. Preserve every other byte and verify the line multiset is unchanged.
Do not change public files, interface declarations, task text, model code,
decoder, action parser or output tokens. Do not modify the original bank.

Generate one new two-action prefix per case per actor,16 total, with the original
512 action-token,4096 prompt/observation and32768 transformer-prefix limits.
Fresh state, deterministic seed198, <36M parameters, recurrent4096bytes/no replay.
One actor on Colab A100 at a time;600-second overall bound. No optimizer updates,
teacher continuations, retries, substitutions or adaptive settings.

Save every new raw trace and workspace, including caps. Compare exact target
WRITE headers, successful target writes and expected target-file presence per
eight planned cases, plus action/stop distributions. A written file need not
contain correct code; this is not a full-episode capability or speed comparison.
Archive source, changed prompts, baseline hashes, all raw results and receipts.
A positive result diagnoses conditioning sensitivity, not frontier competence.
