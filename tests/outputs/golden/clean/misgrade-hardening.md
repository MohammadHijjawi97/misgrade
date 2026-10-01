# Hardening suggestions for toy

misgrade audited `toy_rewards:compute_score` (callable adapter) with 5 grader calls and made 0 findings in 0 groups. For each group, most frequent first: what the finding means, minimized examples with the certificate that says what was required, and changes that usually remove it.

These are suggestions, not verified fixes. After changing the grader, run the audit again with the same seed (and keep the minimized cases as regression tests with `--format pytest`).

No findings in the 5 grader calls made, so there is nothing to harden on this evidence. That is not a proof of correctness: the report lists the categories that were tried.
