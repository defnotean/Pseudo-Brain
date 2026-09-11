# MBPP training-bank v1: underscore API import correction

Inspection of r0's sole canonical-control failure, task 798, showed a valid
reference `def _sum(...)` omitted by the adapter's wildcard import. This was an
adapter error, not a bad reference program. The diagnosis preceded eligibility
of any downstream training corpus. Preserve r0 and r1 source/controls; r1 may be
stopped with a separate explicit supersession receipt rather than finishing an
audit with a known binding defect. Never treat partial r1 output as complete.

Retain wildcard imports for original exported helpers/constants, and explicitly
import every top-level function/class declaration in both public and private
test wrappers. This preserves required underscore-prefixed APIs without changing
reference programs or assertions. Keep the module-docstring fix from amendment1.
Add an underscore-API regression and require actual public/private positive and
negative sandbox controls on original task798 before launching the full r2 audit.

Freeze r2 source and rerun the same374-candidate procedure from scratch with the
original900-second bound, no lowered quotas, no replacement tasks, and no model
updates. Archive the failed/partial variants separately and report the corrected
eligibility and all exclusions only after r2 is complete.
