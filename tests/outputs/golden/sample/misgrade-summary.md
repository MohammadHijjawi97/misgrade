## misgrade: toy

callable grader `toy_rewards:compute_score` · 2 items, 5 cases, 9 grader calls · template `{answer}` · seed 0 · misgrade 0.0.0-test

| Measure | Observed | Rate | 95% CI |
| --- | --- | ---: | --- |
| Self-validation | 2 of 2 gold answers accepted as themselves | 100.0% | 34.2% to 100.0% |
| False positives | 1 of 2 wrong answers accepted | 50.0% | 9.5% to 90.5% |
| False negatives | 1 of 1 equivalent answers rejected | 100.0% | 20.7% to 100.0% |
| Fault checks | 1 of 1 verdicts changed under runtime faults | 100.0% | 20.7% to 100.0% |

**4 findings:** 2 false negatives, 1 false positive, 1 fault.

Categories with findings in the main phase: whitespace (1/1 on 1 item), near-miss (1/1 on 1 item). A category's cases are its operators applied to every item, not independent draws: its 95% interval is conditional on these items and operators (per-operator counts are in the card).

| # | Kind | Category | Response | Gold | Expected | Observed |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | false negative | whitespace | `"42 "` | `"42"` | accept | reject (score 0) |
| 2 | false negative | whitespace | `"42 "` | `"42"` | accept | reject (score 0) |
| 3 | false positive | near-miss | `"43"` | `"42"` | reject | accept (score 1) |
| 4 | fault (repeat) | identity | `"42"` | `"42"` | accept | reject (score 0) |

Certificates (why each expected verdict holds):

1. `number-001::ws.trailing-space`: appended a space
2. `number-001::ws.trailing-space`: appended a space
3. `number-001::number.plus-one`: 43 != 42
4. `number-001::identity`: the response is the gold answer itself

_A finding is an observed verdict that differs from what the case's certificate requires (or, under a fault check, from the clean run). Rates and the error pattern use the main phase only._
