This is the **first and only audit of clip 2**, run on the build exactly as delivered (the same code that was tuned on clip 1; nothing was changed for this clip and nothing was changed after the audit). It is a transfer test, and the result is clearly worse than on clip 1.

**Confirmed by both skeptics (19 of 20 serious claims):**
* **Referee tracked as an athlete (8 frames):** B's solid box and skeleton are on the referee at 81, 312, 347, 348, 349, 351; A's are on the referee at 156 and 363. At 81 the real blue athlete has no B overlay at all; at 349-351 he is outside or mostly outside B's box.
* **Wrong body, A:** 216 (A's colour box covers the blue athlete's back, including the white back panel of his gi); 347 (A's skeleton sits on the blue athlete).
* **B lost while visible (9 frames):** 137, 439-443, 450-452. In the second match the blue athlete is mostly hidden behind the white athlete: the only blue visible is a sliver under 40 px wide (rejected by the minimum-width rule) and the other blue blobs are ad boards.
* 1 claim was refuted (138, B).

**Cause of the referee failures (diagnosed, not fixed):** the athlete filter drops a person only if the torso is mostly dark with no gi colour. The navy suit reads 33% blue / 62% dark on frame 80, which passes as "blue gi", and that candidate earns the blue-class prior while the real blue athlete's box (overlapping the white athlete) is classed `mixed` and earns none. The assignment then hands B to the referee. Colour alone cannot separate a navy suit from a dark blue gi (clip 1's blue gi reads up to 70% dark); a fix needs context, for example that the two athletes are in contact while the referee stands apart. That would change shared assignment logic and needs clip 1 re-validated, so it was not done in this step. The eight referee frames are pinned as strict-xfail tests in `tests/test_clip2_known_failures.py`.

**Unaudited:** 288 of 453 frames were not sampled (the audit covered every third frame from frame 0, plus the flagged frames). The referee error may be more frequent than the 8 sampled frames show: the sheets sample about a third of the frames and the referee stands next to the athletes for long stretches.
