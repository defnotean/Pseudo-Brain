# Code signature conditioning preparation

The static v4 inventory found6870 of8104 parseable training functions had no
target function name in their question, and24 normalized questions mapped to
multiple target names. This under-specifies exact-signature instruction tasks.

The opt-in `signature_conditioned_code` formatter supplies the existing function
declaration alongside its existing documentation. Preserve declaration spelling,
decorators, async, multiline/default/positional-only arguments and annotations;
never copy the function body into the question. The answer remains byte-for-byte
the complete_exchange answer. Retain all previous isolation keys and add new
question/text keys so formatting cannot bypass related-record quarantine.

This is preparation for a later registered data experiment. Do not mutate the
frozen v4 bank, live run, model or evaluator. Recompute token lengths and all
selection/isolation metadata before using any transformed record for learning.
An explicit declaration fixes missing interface information; it does not prove
the documentation fully specifies behavior or provides every external dependency.
Reject invalid/ambiguous source and record failures rather than inventing a name.

Run functional tests only on Colab CPU while the GPU is occupied. Test exact
interface preservation, unchanged answers, retained quarantine keys and rejection
of invalid/multiple definitions. No competence claim follows from these tests.
