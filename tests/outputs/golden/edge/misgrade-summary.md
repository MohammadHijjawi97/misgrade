## misgrade: my "grader" \<v2\>

verl grader `C:\Users\Me\rewards\math rewards.py:compute_score` · 3 items, 10 cases, 21 grader calls · template `\boxed{{answer}}` · seed 7 · misgrade 0.0.0-test

| Measure | Observed | Rate | 95% CI |
| --- | --- | ---: | --- |
| Self-validation | 2 of 3 gold answers accepted as themselves | 66.7% | 20.8% to 93.9% |
| False positives | 2 of 2 wrong answers accepted | 100.0% | 34.2% to 100.0% |
| False negatives | 1 of 2 equivalent answers rejected | 50.0% | 9.5% to 90.5% |
| Fault checks | 3 of 4 verdicts changed under runtime faults | 75.0% | 30.1% to 95.4% |

6 of 20 grader calls ended without a score (error, timeout or crash), left out of every rate; 1 call ended on purpose by the worker-death check (not counted as an error); 1 variant not evaluable (the gold answer of the item was not accepted).

Notes:

- fault check not run: concurrency (the fault budget of 3 calls was too small)
- misgrade cannot read 1 gold answer as the item's type: \<script\>alert("x")\</script\> \]\]\> \`tick\` \`\`two\`\` \| pipe \*star\* \_under\_ \\boxed{1} \[bold\]markup\[/bold\] & --\> \<!-- nul\\u0000 c1 del\\u007f lone\\ud83d surrogate

**8 findings:** 2 false negatives, 2 false positives, 1 self-validation failure, 3 faults.

Categories with findings in the main phase: unicode-form (1/1 on 1 item), near-miss (1/1 on 1 item), hedge (1/1 on 1 item). A category's cases are its operators applied to every item, not independent draws: its 95% interval is conditional on these items and operators (per-operator counts are in the card).

| # | Kind | Category | Response | Gold | Expected | Observed |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | false negative | unicode-form | `"\\boxed{\u22121000}"` | `"-1000"` | accept | reject (score 0) |
| 2 | false negative | unicode-form | `"\\boxed{\u22121000}"` | `"-1000"` | accept | reject (score 0) |
| 3 | false positive | near-miss | ```"\\boxed{-999} <script>alert(\"x\")</script> ]]> `tick` ``two`` \| pipe *star* _under_ \\boxed{1} [bold]markup[/bold] & --> <!-- nul\x00 c1\x85 del\x7f lone\ud83d surrogate\x07\U0001f600"``` | `"-1000"` | reject | accept (score 1) |
| 4 | false positive | hedge | `"\\boxed{A or B}"` | `"B"` | reject | accept (score 0.75) |
| 5 | self-validation failure | identity | `"\\boxed{say \"hi\"}"` | `"say \"hi\""` | accept | reject (score 0) |
| 6 | fault (worker-death) | identity | `"\\boxed{-1000}"` | `"-1000"` | accept | crash: worker exited with code -9 |
| 7 | fault (timeout) | identity | `"\\boxed{B}"` | `"B"` | accept | error: ValueError: bad "input" \<here\>\\u0000 |
| 8 | fault (repeat) | identity | `"\\boxed{-1000}"` | `"-1000"` | accept | reject (score 0) |

Certificates (why each expected verdict holds):

1. `number-007::unicode.minus`: sympy: \<script\>alert("x")\</script\> \]\]\> \`tick\` \`\`two\`\` \| pipe \*star\* \_under\_ \\boxed{1} \[bold\]markup\[/bold\] & --\> \<!-- nul\\u0000 c1 del\\u007f lone\\ud83d surrogate equals the gold
2. `number-007::unicode.minus`: sympy: \<script\>alert("x")\</script\> \]\]\> \`tick\` \`\`two\`\` \| pipe \*star\* \_under\_ \\boxed{1} \[bold\]markup\[/bold\] & --\> \<!-- nul\\u0000 c1 del\\u007f lone\\ud83d surrogate equals the gold
3. `number-007::near.plus-one`: names two options \| so it commits to no single answer
4. `mc-003::mc.hedge-prev`: names two options \| so it commits to no single answer
5. `string-002::identity`: the response is the gold answer itself
6. `number-007::identity`: the response is the gold answer itself
7. `mc-003::identity`: the response is the gold answer itself
8. `number-007::identity`: the response is the gold answer itself

_A finding is an observed verdict that differs from what the case's certificate requires (or, under a fault check, from the clean run). Rates and the error pattern use the main phase only._
