# Behavior codec v2: explicit legacy codec dependency

The r1 evaluation snapshot supplied the correct episode schema but omitted the
training-only executed_trajectory module. Import collection failed before tests.
For r2, copy that one unchanged module from the frozen procedural v1 source into
the new snapshot, alongside the frozen evaluation episode dependencies. Record
all hashes in the manifest and retain both failed packaging attempts. No codec
requirements, tests, data, or180-second bound change.
