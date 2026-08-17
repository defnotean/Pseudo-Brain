# RCQ-v2 TEST range retirement and replacement

Status: permanent chronology record. This is not an RCQ result or a claim of
model competency.

During evaluator development on 2026-08-16, the proposed RCQ-v2 recipient
range `[1048576, 1049088)` was locally materialized before a durable once-only
claim. Label-derived aggregate counts were inspected. That invalidates the
range as untouched evidence. Those counts must not appear in production
criteria, registration, tests, or scientific claims.

The following contiguous RCQ development ranges are permanently retired:

- recipient `[1048576, 1049088)` — opened; never valid for a claim;
- adjacent donor `[1049088, 1049600)` — retired with the contaminated design;
- adjacent guard `[1049600, 1050112)` — retired to keep the boundary explicit.

The replacement RCQ-v2 final-evaluation namespace is:

- recipient `[3145728, 3146240)`;
- donor `[3146240, 3146752)`;
- guard `[3146752, 3147264)`.

During guard testing, a denied public-constructor invocation involving a fresh
range may have occurred. The available execution evidence does not establish
which exact fresh role or how many denied calls occurred, so this chronology
records the conservative possibility rather than asserting a more precise
history. The guard rejects before dataset object/state or manifest creation,
sequence materialization, indexing, iteration, content access, or label access.
No replacement-range dataset was successfully constructed and no
replacement-range example or label was opened. The guard is now tested through
a target-blind range-overlap predicate; the literal protocol forbids any
further constructor attempt before claim.

Before final evaluation, registration may bind only target-blind geometry and
metadata-derived manifests. The evaluator must atomically claim both recipient
and donor ranges before constructing either dataset. All target,
changed-action, previous-policy, lookup, and value counts are post-claim facts
and belong only in create-once ledgers and the final receipt.

The already reserved matched-campaign bands remain separate and unchanged:

- `[2097152, 2097408)`;
- `[2097408, 2097664)`;
- `[2097664, 2097920)`.

No future run may silently reuse, relabel, or unretire any range listed here.
