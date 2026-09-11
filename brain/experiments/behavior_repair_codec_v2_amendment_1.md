# Behavior codec v2: frozen episode dependency correction

The first CPU bundle inherited the older procedural-bank episode schema through
the MBPP case preparation snapshot. It lacked the transformer prefix_tokens
field; all22 tests stopped while constructing that schema. No model was generated
or trained, and compatibility was not run. Preserve the failed r0 bundle/logs.

Use the frozen source from completed post-training tool evaluation r2, manifest
`98b85f02412d4b8bf02f8cea4b50322fe4815087b8eed8ec17d4f467653d9cee`,
which is the intended actual collection episode/session dependency. Add the new
codec separately; do not patch completed model-evaluation sources. Rerun the
same tests and180-second preflight bound in a new r1 snapshot.

The v1 comparison additionally loads its observation_frame from the original
frozen procedural source, rather than allowing both encoders to resolve a shared
current import. This makes the reference framing independent. Requirements,
record data and tolerances remain unchanged.
