# Wikipedia Requests for Adminship (Wiki-RfA)

## Source

- Official dataset page: <https://snap.stanford.edu/data/wiki-RfA.html>
- Raw file: `wiki-RfA.txt.gz`
- Downloaded: 2026-08-11
- SHA-256: `88d53196fb2564a2e20286dbba818832f718cc352bb181a2101d23d2556f0862`
- Integrity: `gzip -t` passed.

The unit is a directed, signed, timestamped vote in a Wikipedia request for
adminship (RfA). `VOT` is support (`1`), neutral (`0`), or oppose (`-1`);
`RES` is the election outcome.

## Local audit

Raw records:

- 198,275 votes
- 11,381 users
- 3,497 distinct targets (candidates)
- 189,003 distinct ordered voter-candidate pairs
- 144,451 support, 12,648 neutral, and 41,176 oppose votes
- 9,367 records have no timestamp; 1,661 records also have no source user

Analysis-ready complete cases:

- 188,904 records with nonempty source, target, sign, and a parseable timestamp
- time span: 2003-08-16 01:25 through 2013-06-05 19:50
- 11,128 users and 3,468 targets
- 180,860 distinct ordered pairs
- 7,337 pairs recur; 8,044 events occur after a pair's first vote
- signs: 137,963 support, 11,648 neutral, and 39,293 oppose
- four nonempty timestamps remain unparseable because of evident source typos
  (`Julu`, `Janry`, `Mya`, and hour `31:29`)

Timestamp parsing must accept both full and abbreviated English month names.

## Role in this project

This dataset is suitable as a **cross-context robustness test** of dynamic
evaluative reputation/status capital. It is not a strict replication of the
Bitcoin trust-rating setting:

1. the target is an administrator candidate rather than an arbitrary user;
2. exposure and the target risk set are constrained by open RfA windows;
3. a repeated voter-target pair generally represents a new candidacy round;
4. the election outcome offers an additional macro-level validation target.

Therefore, fit the marked-event micro--macro model separately and construct
the risk set from candidates with an open RfA. Do not pool these events with
Bitcoin OTC/Alpha, and do not reuse an unrestricted all-user target-choice
likelihood. If candidacy windows cannot be reconstructed defensibly, restrict
the empirical task to vote arrival/sign and capital-distribution dynamics.
