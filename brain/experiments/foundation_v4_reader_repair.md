# V4 reader repair before training

The first v4 launch, PID29039, failed before model initialization/training at
`training.load_partition`: Python `str.splitlines()` split a Unicode line
separator inside a valid JSON string. No weights or optimizer updates occurred.
Preserve its `run/status.json` and `training.log`; do not overwrite the failure.

The frozen v4 bank and its hashes remain unchanged. The v4 wrapper now reads
JSONL records using literal byte LF delimiters, preserving embedded Unicode
separators. It retains raw/gzip/row-count checks and rejects malformed/blank
records. The v3 helper file and all training/model/optimizer functions remain
unchanged and hash-pinned. This is a reader bug repair, not a new training budget
or relaxed scientific gate.

Before retrying into a distinct output directory, run the remote regression tests
for U+0085/U+2028/U+2029, escaped newlines, CRLF, malformed and blank records.
Load both complete frozen partitions with the repaired reader, check exact row
and token totals, and preserve all source/test/failure receipts with new hashes.
The retry uses the same corpus, initialization, ordering, schedule and limits.
The builder also uses the shared reader and asserts a serialization round trip
for future preparations. The already frozen bank was produced by the archived
original builder; it is not rebuilt or relabeled with the newer source hash.
