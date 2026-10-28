# Idea 2's follow-ups: the critics' gradients miss λ_s and λ_ξ at their own states and at every training snapshot, linear probes find little of λ_s or λ_ξ in hidden layers beyond z and random features, and unseen road moves braking and corner λ too much for the existing λ to be BC-1's target (exploratory)

Results of 25–28 Oct 2026. **Every number in this note is exploratory.** Each one comes from an
item of [the follow-ups plan](costate-followups-plan.md). The plan was first committed in
`eaf1e80`, before any computation, and has 14 dated amendments, each committed before the numbers
it governs. These items are lane A of [`docs/plan/follow-through.md`](../plan/follow-through.md)
(Sections 7.2 and 7.3, and decisions #8, #9 and #16 of Section 10). None of them is
pre-registered with a tag, none is confirmatory on these 34 networks, and none changes a decision
of `costate-prereg-v1` (`outputs/costate/analyse/decisions.json`, committed in `9b25cc2`). The
confirmatory record is [`costate.md`](costate.md). It is not edited here. Section 3 below says
how each result bears on it.

**Where it ran.** Everything ran on the branch `costate-exploratory`, in a worktree cut from
`c66dcec`. Its `src/`, `scripts/costate.py`, `scripts/_bootstrap.py`, `scripts/fields.py` and the
design document are byte-identical to the tag's. The code for these items is one new script,
`scripts/costate_exploratory.py`, with `tests/test_costate_exploratory.py` (121 tests at
`8f2ad68`).

**What the script checks before every item:**
- the plan, the freeze and the tag (plan R2);
- the sha256 of every bound file it reads, against `decisions.json`'s `inputs` or the record's
  `outputs` (plan R3);
- that it calls the modules' `unlock_trained` only after those checks;
- that every scored row passes one guard. The guard refuses braking- and corner-phase rows, and,
  for λ, any row outside K2b's frozen decision mask (decision #8, plan R5).

**Checked for this note:**
- `costate_exploratory.py verify` reports 0 problems over the 127 bound files and the unbound
  activations, at `bf9b9a1`.
- Every committed summary under [`costate-followups/`](costate-followups/) is byte-identical to its
  stamped local output.
- No committed file holds a local path.

Labels used throughout:
- **Exploratory.** Every number here.
- **Band word** ("in the refute band", "inconclusive"). A frozen band applied to an exploratory
  number describes it and decides nothing (plan R1).
- **Descriptive companion.** A number an amendment added beside an item, with no expected
  direction.
- **Targets only.** BC-a0, K2b on N and CS-14 read no network output, so they may read every
  phase (plan R5).
- **Expected** (E1.1 and so on). A direction written into the plan before the item ran. A miss is
  a miss of the plan's prediction. It is not a failed check.

Sources:
- Each item's stamped `outputs/costate/exploratory/<item>/summary.json` (local).
- The committed copies under [`costate-followups/`](costate-followups/).
- Appendix A maps each key number to its file and JSON path, and Appendix B lists the commits.

## 1. Summary

The confirmatory study refuted C1 in all 9 decision networks. On approach-phase interior states,
the critics' pulled-back value gradients do not match the window problem's λ_s and λ_ξ. It found
C2 mostly inconclusive and C2-sw refuted by its band, and C3-w ruled out a strong association.
Its Section 10 listed follow-ups. Lane A ran the plan's 16 items (CS-1 to CS-15 and BC-a0) in the
plan's order, except CS-13 on T. CS-13 ran only on N, where no pair exists, and CS-14 stopped at
its stop rule. Network outputs were scored against window-problem targets only on approach and
exit rows (decision #8). Network-only descriptives (CS-2) read every interior row, braking and
corner rows included, as R5 allows. Of the 44 expected directions that could be read, 18 held.

**C1: the refute holds everywhere it was checked, and the failure sits in the critic's input
gradient.**
- It holds at the policies' own states:
  - **CS-10:** C1-own on own-run states is in the refute band in 9 of 9, on 11–14 times the
    confirmatory rows. The largest CI upper end is 0.117.
  - **CS-10 (iii):** on N it is in the band in 9 of 9 (largest upper end 0.008).
- **CS-9:** C1 on N is in the refute band in 8 of 9. G3s1 is inconclusive at 0.213, no better
  than its best random-init critic.
- **CS-6:** all 144 training snapshots are in the refute band. The statistic is best at 10M–15M
  timesteps (0.076–0.149) and falls with training in all 9 (ρ with log timesteps −0.55 to −0.98).
- **CS-2:** every one of the 155 value heads is in the refute band, though the heads agree with
  each other (ρ 0.960–0.997).
- **CS-4:** λ is in what the networks see. A ridge on z gives R² 0.354–0.355 for λ_s and
  0.663–0.673 for λ_ξ, and one hidden layer gives 0.641–0.651 and 0.817–0.845. Adding the
  critic's own gradient adds nothing (ΔC1 −0.017 to 0.022).
- **CS-1:** the very negative ξ scores come from the critic's gradients with respect to the near
  half of the edge preview. They sit on the late approach (125–50 m before the apex) to the two
  44.7 m corners, at high tyre utilisation. On 87–94% of tail rows that near half's
  contribution has the sign opposite to λ_ξ. The observation map plays no part. No other
  calibration passes its plants.

**Linear probes find little of λ_s or λ_ξ in a hidden layer beyond z and random features. The
decoy-controlled test of the critic's layers is underpowered, and on the Π targets the seed-0
critics' h3 met the plan's "first sign".**
- **CS-7:** no critic layer or head holds λ_s beyond random-init critics. On λ_ξ, which is not
  decoy-controlled, the critic's h3 gains 0.056 and 0.066 in two networks (G1 and G1s2, CI above
  0), and the actor's h3 gains as much there. The λ_res half has little power: λ_res is not
  decodable by any probe on these rows. Its 14 λ_res,ξ cases above 0.05 with the CI above 0 (h1
  to h3 and the head units) measure how much more random critics' features over-fit, not
  information in the trained layer.
- **CS-5:** 5 critic-h3 gains on the Π targets pass 0.05 with the CI above 0, all in the seed-0
  network of each variant (G1 on Π^out and Π^μ, G2 on Π^μ, G3 on Π^in and Π^out). The plan
  calls that "the first sign that a layer holds a shadow price beyond the input". A layer that V
  is linear in, and that tracks the room the road leaves, need not encode a price, so this is a
  sign, not a finding.
- **CS-8 (c):** with a decoy model that fits geometry 6, C2's CIs are 1.6–8 times narrower than
  f̂_D's on the same rows. The actor's h3 sits below its pool's median in 7 of 9, and no point
  reaches 0.05.
- **CS-12, CS-15 and CS-9:** C2 is inconclusive in all 25 other networks, h1 and h2 give no
  confirm, and C2 on N gives no confirm. C2-sw on N gave one band-word confirm (G3, 5.1
  [0.030, 28.1]); it is read as uninterpretable, because σ^ped barely varies on N.

**C2-sw's rank-1 pattern is real, and consistent with position-and-speed information. There is
no evidence of switching-function information beyond it, but the test for it has low power.**
- **CS-8 (a):** on approach and exit rows the trained h3 beats all 20 random actors in 7 of 9,
  and lies above every host of its own random-only null in 6.
- **CS-8 (b):** once σ^ped is residualised on position and speed, no network's gain has its CI
  above 0. Four networks still rank first, with points larger than before, inside CIs that cover
  0. σ_res keeps 12% of σ^ped's variance, so (b) cannot tell "nothing beyond the decoys" from
  "too little to find".
- **CS-12:** the pattern recurs in 10 of the 22 other generalists.

**The wall price: the policies carry a smaller copy of the optimum's; the critics' width
derivative agrees moderately with the optimum's but not with their own policies'.**
- **CS-11:** in 6 of 9 the policy's own inside-wall price agrees with the optimum's (calibrated
  R² 0.27–0.39, CI above 0), at 37–76% of its median size (28–85% over all 9). The critic's
  derivative agrees with its own policy's price in none of the 9 (R² at most 0.112, every CI
  reaching 0). Against the optimum's Π^in it agrees moderately (R² 0.068–0.350, CI above 0 in 5
  of 9), about as well as the policies' own price does.
- **CS-5:** the critic's h3 holds a little Π^in beyond random critics (0.021–0.056, CI above 0
  in 8 of 9). That need not encode a price: V is linear in that layer, and the value of a state
  depends on the room ahead.
- **CS-3:** nothing links a network's gradient errors to its driving errors (largest |r| 0.109).
- **BC-a0:** the Π targets themselves move by about 0.8–1.8 SD_T with road beyond the preview.

**Braking and corner states (targets only).**
- **BC-a0:** realistic road beyond the preview moves λ by 9–103% of its phase variance per state.
  On corner rows that is about twice as much as K2b's slow second corner. The largest shares are
  corner λ_vx's (U 1.03) and braking λ_s's (0.75). No component-phase cell allows BC-1 to
  pre-register against the existing λ with a bound of 0.1. The recommended route is the
  preview-matched average target (option (ii)), at about 2.3–2.9 times the plan's 150–380 CPU-h
  at W = 400 rates, and more with W = 800 states.
- **CS-14:** q_c cannot be scored under decision #8. Only 134 rows pass its 64-step window
  rule, short of 200.

**Reading.** The follow-ups narrow the confirmatory refute's alternatives but do not change it.
- The gap is not an artefact of:
  - the optimum's states (CS-10, CS-9);
  - the final checkpoint (CS-6);
  - head-level indeterminacy (CS-2);
  - the observation (CS-4);
  - the calibration (CS-1);
  - the optimum-against-policy terms (C1-own on 11–14 times the rows).
- For λ_s and λ_ξ, it is the critics' input gradient that fails to express a costate their
  input carries (CS-4).
- The wall price Π^in is a shadow price of the window problem too, but not one of C1's
  components. There the picture is mixed (CS-11, CS-5). The policies' own price tracks the
  optimum's in 6 of 9. The critics' width derivative tracks the optimum's about as well (CI above
  0 in 5 of 9) but not its own policy's, in any of the 9. The seed-0 critics' h3 holds a little Π
  beyond random critics.
- The confirmatory run's positive C2 points do not survive on approach and exit rows. Under the
  frozen f̂_D, restricting to those rows already makes G1's and G1s1's negative (as `costate.md`
  found on approach rows alone); decoy 2, which fits geometry 6, removes G1s2's and G3s1's
  (CS-8 (c)).
- C2-sw's rank pattern is consistent with position-and-speed information. Residualised on
  position and speed, no gain has its CI above 0, but that test has low power (CS-8 (b)).
- Every claim is still narrow:
  - λ_s and λ_ξ only;
  - approach and exit rows only;
  - synthetic corners;
  - linear probes;
  - one PPO configuration.

## 2. The items against the plan

"Held" counts the item's expected directions that held, out of those that could be read. The
item sections below quote each one beside its result.

| Item | Question | Held | Result | Summary |
|---|---|---|---|---|
| CS-1 | Where does the ξ tail sit; is it the critic or the map; do other calibrations help? | 1 of 4 | Late approach to geometries 38 and 36, near preview, critic not map; Huber and rank fail their plants in 9 of 9; the rule for later items does not fire | [cs-1](costate-followups/cs-1-summary.json) |
| CS-2 | Descriptives: heads, winsorisation, floor, C1 on D, accel term | 3 of 6 | Heads agree (ρ ≥ 0.96) and all 155 refute; C1 on D refutes in 34 of 34; accel term 3.2–5.1% of \|g_vx\| | [cs-2](costate-followups/cs-2-summary.json) |
| CS-3 | C3-int: do gradient errors go with driving errors, network × geometry? | 1 of 1 | No: largest \|r\| 0.109, smallest Holm p 0.130 | [cs-3](costate-followups/cs-3-summary.json) |
| CS-4 | ΔC1, R²_z, the MLP-on-z ceiling | 3 of 4 | λ decodable from z (0.35, 0.67; MLP 0.64, 0.83); ΔC1 −0.017 to 0.022 | [cs-4](costate-followups/cs-4-summary.json) |
| CS-5 | W1 on the Π targets; hidden layers beyond z | 0 of 3 | Π^in less decodable than expected (0.26); actor h3 adds nothing; critic h3 adds 0.021–0.056 on Π^in; 5 critic-h3 cases, all in seed-0 networks, meet the plan's "first sign" (gain above 0.05, CI above 0) | [cs-5](costate-followups/cs-5-summary.json) |
| CS-6 | C1-dev over 16 snapshots | 1 of 2 | 144 of 144 in the refute band; best at 10M–15M, falls with training in 9 of 9 | [cs-6](costate-followups/cs-6-summary.json) |
| CS-7 | Critic-layer and head-unit probes of λ and λ_res | 1 of 3 | Nothing on λ_s or λ_res; λ_ξ gains in G1 and G1s2 matched by the actor's h3 | [cs-7](costate-followups/cs-7-summary.json) |
| CS-8 | C2-sw's random-only null; C2-sw residualised; C2 with decoy 2 | 1 of 3 | Rank 1 of 21 in 7 and above every own-scaling host in 6; no CI above 0 once residualised (low power); decoy 2: 3 refute, 6 inconclusive, h3 below pool median in 7 | [cs-8](costate-followups/cs-8-summary.json) |
| CS-9 | N: C1, C2, C2-sw at the networks' own states, with K2b on N | 2 of 5 | C1 refutes in 8 of 9; C2 6 refute, 3 inconclusive; C2-sw not interpretable on N; K2b on N excludes nothing | [cs-9](costate-followups/cs-9-summary.json) |
| CS-10 | C1-own on own-run states, then on N | 1 of 2 (D); E9.5 on N | Refute band 9 of 9 on D's own runs (11–14× rows) and 9 of 9 on N | [cs-10](costate-followups/cs-10-summary.json), [N](costate-followups/cs-10-iii-summary.json) |
| CS-11 | The policy's own width price Π^{in,own} | 0 of 3 | Critic–own R² ≤ 0.112 in 9 of 9; own–optimum 0.27–0.39 in 6 of 9; on N every agreement higher | [cs-11](costate-followups/cs-11-summary.json), [N](costate-followups/cs-11-n-summary.json) |
| CS-12 | C2 and C2-sw beyond the 9 (primary only) | 1 of 2 | C2 inconclusive in 25 of 25; C2-sw rank 1 in 10 of 22 generalists, 2 of 3 specialists | [cs-12](costate-followups/cs-12-summary.json) |
| CS-13 | C2-pairs at matched decoys | not read | On N no pair exists; on T not run (Section 6) | [N](costate-followups/cs-13-n-summary.json) |
| CS-14 | q_c, the action-projected critic signal | not read | Stopped: 134 rows pass the window rule, under 200; no network evaluated | [cs-14](costate-followups/cs-14-summary.json) |
| CS-15 | FWL check, actor h1 and h2, C2 on D | 1 of 3 | FWL lower by up to 0.076 (raw gain, mostly s); h1 and h2 no confirm; D 7 inconclusive, 2 refute | [cs-15](costate-followups/cs-15-summary.json) |
| BC-a0 | How much unseen road moves braking and corner λ (targets only) | 2 of 3 | Per-state U 0.09–1.03; route (ii) in all 8 cells; Π p90 0.79–1.78 SD_T on approach and exit | [bc-a0](costate-followups/bc-a0-summary.json) |

Some misses were narrow:
- E1.2's clipping part: κ clipping not enriched in 5 of 9, against the 6 needed. Its central
  part did not narrowly miss: no row lies in the 83.3 m/s kink band (speeds stop at 81.0 m/s),
  and the tail's vx is only about 1 m/s higher;
- E4.4: G3s1's s ΔC1 is 0.022, against a bound of 0.02;
- E10.2: 5 of 9, against 6.

Others went the opposite way:
- E1.3: the tail is in the near preview, not the far;
- E5.3: Π^slip and Π^μ are more decodable than Π^in;
- E11.1: the critic agrees less with its own policy's price than with the optimum's, in 9 of 9;
- Ea0.2: λ_s's share U is larger in braking than in the corner, because λ_s varies little within
  braking.

## 3. How they bear on the confirmatory findings

Section numbers in this section are `costate.md`'s. Nothing below changes a decision there. Each
point says how an exploratory result qualifies or extends a confirmatory reading.

**5.2 C1 (refute in all 9).**
- **The state set.** At the policies' own states the refute stands.
  - CS-10 scored own-run approach and exit rows, 2,223–2,298 (s) and 1,814–1,853 (ξ) per
    network. R²_c is at most 0.014 for s and 0.105 for ξ.
  - On N, C1 is in the band in 8 of 9 (CS-9). The exception, G3s1 (ξ R²_c 0.474), is matched by
    a random-init critic on N's narrow states (ξ R²_c 0.45).
- **The checkpoint.** CS-6 shows the final checkpoint is not an unlucky one. Every snapshot is in
  the band. The earliest critics (10M–15M) come closest, with CI lower ends above 0 in 7
  networks, and agreement then falls as ξ's calibration collapses.
- **The observation.** CS-4 rules out an observation that lacks λ: a ridge on z predicts λ_ξ at
  R² 0.663–0.673 and λ_s at 0.354–0.355.
- **The rank agreement.** Section 1's Spearman of 0.34–0.40 recurs at own-run states only weakly
  (CS-10: −0.15 to 0.33, CI above 0 in 5). Over training its trend depends on the variant
  (CS-6).
- **The far-geometry and random-critic nulls** are not rebuilt off T, except the best random
  critic on N (CS-9).

**5.3 C2 (inconclusive in 7, refute in 2).**
- **The positive points.** Section 8 read the inconclusive positive points (G1, G1s1, G1s2 and
  G3s1, 0.030–0.039) as lying above every random-only null host, under G1's scaling. On approach
  and exit rows they do not survive, but much of that is the change of rows, not the decoy.
  - Under the frozen f̂_D on CS-8 (c)'s approach and exit rows, G1 and G1s1 are already negative
    (−0.058 and −0.192). `costate.md` Section 8 found the same on approach rows alone, where all
    four points are negative.
  - Decoy 2 (log R × 3, same θ and direction), which fits geometry 6, removes the remaining two:
    G1s2 goes from 0.066 to 0.023 and G3s1 from 0.043 to −0.010 (CS-8 (c)). G1 and G1s1 become
    −0.005 and −0.027.
  - The actor's h3 sits below its pool's median in 7 of 9.
- **Geometry 6.** It falls from 0.35–0.54 of the base's squared error to 0.055–0.076, which
  confirms design 17.6's reading that its weight is a residualisation artefact.
- **The rows.** CS-8 (c) fits on approach and exit rows, not the confirmatory 3,951 interior
  rows, so its band words (3 refute, 6 inconclusive) are not on C2's rows.
- **Breadth.** C2 is inconclusive in all 25 other networks (CS-12), and h1 and h2 show no more
  than h3 (CS-15).

**5.4 C2-sw (refute by its band in all 9; rank 1 of 21 in 7, descriptive).**
- **The null.** Section 9 noted that C2-sw had no between-network null. CS-8 (a) builds one, on
  the 2,507 approach and exit switching rows.
  - Its bulk sits at 0: median 0.0005, 95th percentile 0.025.
  - The trained h3 lies above every host of its own scaling in 6 of 9.
- **The band word moves on these rows.** The statistic is 0.021–0.049 in 8 of 9, and the band
  word is inconclusive in all 8. On all 2,911 rows the 9 refuted.
- **What the gain is.** The result is consistent with the caveat in Section 8. Residualised on
  position and speed, σ^ped keeps 11.6% of its variance, no layer decodes the rest linearly, and
  every CI covers 0 (CS-8 (b)). Four networks still rank first, with larger points inside those
  CIs, and the test has low power, so it does not show that nothing is left.
- **The reading.** The rank pattern is consistent with the position-and-speed information design
  15.10 named. Design Section 9's switching-function reading still has no support.

**5.6 C3-w and 5.9 W1 (wall price).**
- **The policy's own price.** CS-11 adds the comparison the confirmatory run lacked.
  - In 6 of 9 the policies' own price of the inside wall agrees with the optimum's at a smaller
    size.
  - The critics' derivative agrees with its own policy's price in none of the 9. On D's rows it
    agrees moderately with the optimum's Π^in (R²_c 0.068–0.350, CI above 0 in 5 of 9).
  - So in 6 of 9 W1's near-zero agreement lies in the critic's derivative rather than in the
    policies' valuation. In G1s2, G2s1 and G3s2 the policy's own price also agrees poorly with
    the optimum's (R²_c below 0.08).
- **C3-int.** CS-3 finds no within-network link between gradient errors and driving errors, the
  link C3-w could not test (largest |r| 0.109).
- **The target.** BC-a0 measures how much Π moves with road beyond the preview: p90 |ΔΠ|/SD_T is
  0.79–1.02 on approach rows and 0.96–1.78 on exit rows. That qualifies Π^in as the reference for
  W1, C3-w's x and CS-5. Part of any disagreement with Π^in may be Π^in's own sensitivity to road
  no network sees.

**5.7 C1-own (refute in all 9, on about 162 rows per component).**
- CS-10 gives an exploratory, better-powered analogue on own-run approach and exit states
  (11–14 times the rows, with CIs 0.001–0.19 times as wide), which also refutes. These are the
  kept runs' every-5th states, not T's restarts.
- CS-10 (iii) gives the same analogue on N. The refute band holds in 9 of 9 both times.
- Section 8 said that C1-own rests on few rows with wide CIs. The confirmatory C1-own's caveat
  stands for its own rows; CS-10 does not change its CIs.

**5.9 C1-heads.**
- CS-2 extends it to every head's C1. All 155 heads are in the refute band.
- They follow the main V's R²_c only where it is near 0. Where the ξ tail drags the main V far
  below 0, the heads lie 0.1–1.3 below it.

**7.1 The ξ tail.** CS-1 answers Section 10's first item.
- **Where.** Geometries 38 and 36, 125–50 m before the apex, at tyre utilisation near 0.9.
- **What.** The near half of the edge preview: 77–88% of |g_ξ|, with the sign opposite to λ_ξ on
  87–94% of tail rows.
- **Not the pullback.** No observation-clip, preview-spacing or tyre-channel kink of the pullback
  is involved.
- **Calibrations.** Neither the rank calibration nor design Section 6's original Huber fit passes
  its plants.
- **Over training.** CS-6 suggests that the tail grows with training, since ξ's R²_c turns
  negative between 22M and 49M timesteps. It does not locate the tail at each snapshot.

**8 What it means.**
- **"Values fit, gradients do not."** The follow-ups sharpen this reading.
  - **The gradient.** The heads' gradients agree with each other (ρ 0.960–0.997), so the
    mismatch is not head-level indeterminacy (CS-2). It is not the policy's own costate either
    (CS-10), and on ξ it worsens with training (CS-6).
  - **The values.** CS-6 did not measure value accuracy over training.
- **Design Section 9's two readings.** Neither is established.
  - "The regression does not determine the gradient" cannot be tested through the heads, since
    they share a trunk.
  - "A determined gradient, but of V^π" is not supported. At the policies' own states the
    gradient does not match the deterministic policy's costate either (CS-10). λ^own is still
    not exactly the costate of the V^π the critic estimates (`costate.md` Section 8).

**9 What it does not mean.**

Limits these items narrow:
- the optimum's states (CS-10, CS-9);
- final checkpoints only (CS-6);
- other forms of λ in the critic: layers and heads (CS-7), per-head R²_c (CS-2);
- calibration dependence (CS-1);
- geometry 6 and the decoy (CS-8 (c));
- the C2-sw band without a null (CS-8 (a));
- C2's h3-only probe (CS-15);
- the deferred analyses: C3-int, C1-dev, ΔC1 and N (CS-3, CS-6, CS-4, CS-9).

Limits that stand:
- **The decision set.** λ_s and λ_ξ only, approach and exit rows only. Decision #8 keeps braking
  and corner rows for BC-1, and BC-a0 measured only targets.
- **q_c.** Not scorable under #8 (CS-14).
- **The probes.** Linear probes of hidden layers. The only MLP is CS-4's, on z alone.
- **The setting.** Synthetic A2 corners, one PPO configuration, and nothing about R_TEST.

**10 What next.**
- Item 1 is done (CS-1).
- Item 2 is done except C2-pairs on T: C1-dev (CS-6), ΔC1 (CS-4), C3-int (CS-3). C2-pairs is not
  scorable on N.
- Item 3 is done (CS-8 (a) and (c)).
- Item 4 has its pilot (BC-a0). Its route recommendation goes to the lead under decision #6.
- Item 5, the intervention, is outside this lane.

## 4. What it means (exploratory)

- **The critics' input gradients do not express λ_s or λ_ξ, at any state set or snapshot
  tried, and the failure is specific.**
  - What the networks see carries λ: z predicts λ_ξ at R² about 0.67 linearly and 0.83 with one
    hidden layer.
  - The critics' gradients do not carry it. What they carry on ξ is a large, wrong-sign response
    to the near edge preview on the late approach to the 44.7 m corners. On ξ the early
    critics, at 10M–15M timesteps, come closer than the final ones.
- **The wall price is a different case.** The policies' own Π^{in,own} tracks the optimum's in
  6 of 9 (CS-11). The critics' width derivative agrees moderately with the optimum's Π^in
  (calibrated R² up to 0.35 on D, CI above 0 in 5 of 9), about as well as the policies' price
  does. What it fails to match, in all 9, is its own policy's price.
- **Linear probes find little of λ_s or λ_ξ in a hidden layer beyond z and random features.**
  - Critic h1–h3 and the head units (CS-7). Their only λ gains, on λ_ξ in G1 and G1s2, are not
    decoy-controlled and are matched by the actor's h3. The decoy-controlled half (λ_res) has
    little power, since λ_res is not decodable by any probe on these rows.
  - Actor h3 under decoy 2, which fits geometry 6 (CS-8 (c)), and h1 and h2 under f̂_D (CS-15).
  - The exception is the critic's h3 on the Π targets (CS-5). It holds 0.02–0.06 of Π^in beyond
    random critics in 8 of 9, and 5 gains in the seed-0 networks meet the plan's "first sign"
    (above 0.05, CI above 0). It need not encode a price: V is linear in that layer, and the
    value of a state depends on the room ahead.
- **The actor's switching-step gain over random actors is real, and consistent with
  position-and-speed information.** It ranks first of 21 in 7 of 9 and clears the random-only
  null. Once σ^ped is residualised on position and speed, no gain has its CI above 0, but that
  test has low power, so there is no evidence of switching-function information beyond position
  and speed, and no evidence against it either.
- **BC-1 needs a new target, and it costs more than planned.** The existing λ in braking and
  corner states depends on unseen road by 9–103% of its phase variance, mostly as a common shift
  from the stored straight continuation. A preview-matched average target (option (ii)) is the
  only route the pilot leaves open, at about 2.3–2.9 times the planned cost at W = 400 rates,
  and more with W = 800 states. Its W = 800 states need a solver that reaches full amplitude.

## 5. What it does not mean (limits)

- **Exploratory only.** No result here is a test. The items report dozens of statistics, and only
  CS-3 corrects for multiplicity (Holm). The expected directions were written in advance, but a
  held direction is not a confirmed hypothesis, and 26 of 44 missed.
- **What was seen before the plan's details were fixed.**
  - The amendments fix operational details. Amendments 4–6 and 12–14 were written after a
    development run or a stopped run had been seen, and each records what it saw:
    - amendments 4–6: CS-11's wiring gaps;
    - amendments 12–14: BC-a0's costs and failure counts.
  - Amendment 9 was committed before CS-9's code, and no CS-9 number had been seen. CS-9's
    development runs came after it, before the canonical run. Three of their outcomes were seen:
    a G2s1 development key moving λ_vx by 0.30 SD_T, G1's 'secondary' C1-own status, and CS-13
    finding no pair. No amendment records them; the CS-9 section discloses them.
  - Partial development runs, mostly on G1, came before most canonical runs. They were written
    outside `outputs/costate`. Only their wiring and timings, and the outcomes each item
    section names, were read.
- **The rows.** Decision #8 keeps every comparison of a network output with a target off braking
  and corner rows (CS-2's network-only descriptives read them). So nothing
  here says whether the critics hold λ where most of the policies' states lie.
  - CS-10's own runs hold 3,248–3,657 braking and 2,555–2,931 corner states per network, only
    counted.
  - λ_n and λ_vx are never scored against a network output.
- **N is narrow.** One state per key, at about 183 m before the apex and 85 m/s.
  - λ there varies mainly across geometries. So random gradients calibrate on ξ, f̂_D adds
    variance, and σ^ped varies little.
  - C2-sw on N cannot be read as an R² gain.
  - N's higher wall-price agreements (CS-11 on N) probably reflect the state set.
- **The controls.** Random-init critics and actors are the only controls for the hidden-layer
  gains (CS-5, CS-7, CS-8). A gain over them says a layer holds more than random features of z,
  not that it encodes a price. The random-init critics' h3 equals the pool actors' h3 (same
  trunk architecture, same seeds), so the critic's and the actor's h3 are compared with one
  shared random-feature control, and their gains in CS-5 and CS-7 are contrasts against it.
- **The Π targets move with unseen road** (BC-a0, p90 about 0.8–1.8 SD_T). So CS-5's and
  CS-11's comparisons against Π^in are against a target that the road beyond the preview moves.
- **BC-a0 is a pilot.**
  - Its shares rest on 21 keys and 43 states (26 keys before the drops), after 5 of the 6
    W = 800 keys were dropped for failed re-solves.
  - The dropped keys are mostly tight corners, so the shares probably understate the movement.
  - Its costs were measured on a shared machine.
- **The probe machinery.** CS-15's FWL check differs from the joint ridge by up to 0.076 on the
  raw gain. The likely cause, the base's ridge penalty, was not checked (α_B was not recorded).
- **The setting.** Synthetic corners, mostly 12 m wide. One PPO configuration at 200M steps.
  R_TEST and `outputs/sealed/` were never read.

## 6. Not done, and questions for the lead

Not done:
- **CS-13 on T** (dropped by the user on 28 Oct 2026; C2-pairs at matched decoys, about 30 min, stored arrays only). The plan's order
  puts it with CS-6 and CS-12, but no step of this lane ran it. The follow-through rates it low
  value, since it qualifies a C2 confirm and C2 never confirmed. On N it is not scorable: no N
  state has a matched T state.
- **CS-14 beyond its stop rule.** q_c, its check against σ and K2b for q_c were never computed
  (amendment 10 item 4).
- **CS-7's λ_res half on decoy 2.** On C1's rows the frozen λ_res is not decodable by any probe,
  so this half has little power. A rerun on decoy 2 needs an amendment.
- **Decoy 2 co-reported for C2 on N and in CS-12.** It also needs an amendment.
- **CS-15's chosen base penalty α_B.** It was not recorded, so the FWL gap's explanation is
  unchecked.

Questions for the lead (plan Section 6, and from the results):
1. **Random-init hosts under #8.** May C2-sw's random-only null be computed on all 2,911
   confirmatory rows? Those are the rows its trained statistic was decided on.
2. **Π under #8.** The plan treats the Π targets as window-problem targets. The confirmatory W1
   already scored Π^in on braking and corner rows, and BC-a0 shows Π moves with unseen road.
3. **C2-sw on N.** Should it be reported only as "not interpretable on N" (σ^ped's SD on N is a
   quarter of T's)?
4. **N.** Does N's narrow state set warrant any further N analysis?
5. **BC-1's route** under decision #6. BC-a0 recommends option (ii) in all 8 cells, at about
   2.3–2.9 times the planned CPU-h at W = 400 rates, and more with W = 800 states. Should q_c
   (CS-14) be part of BC-1's design?
6. **CS-13 on T.** Run it, or drop it as the follow-through's Section 8 allows?

## 7. The items

The item sections follow in the order they ran, each with its run, rows, wiring checks, results,
expected directions and a closing reading. Each was written when its item finished and is kept as
written, except that numbers found misquoted against the summaries at the close of the lane were
corrected in place, and the CS-5 and CS-7 sections gained a note on their shared control. The
CS-5 section's open |ΔΠ|/SD_T was later answered in the BC-a0 section.

## CS-1. The ξ tail (exploratory)

**The run.**
- `scripts/costate_exploratory.py cs-1` at `70979ee`, under the plan with amendment 1 (`320ef22`,
  CS-1's operational details, committed before any CS-1 number).
- The 9 decision networks, 125 s on the Mac (the stamps' start to end), niced, no pool.
- Outputs: `outputs/costate/exploratory/cs-1/summary.json` (committed copy:
  [`costate-followups/cs-1-summary.json`](costate-followups/cs-1-summary.json)) and `rows.npz`
  (per row, local).
- The stamp records an empty `changed_vs_head`, no partial flag, and the sha256 of all 39 inputs:
  the world, the gates' K4 record, the 9 networks' critic, K3 and activations files, and their
  checkpoints through the record's pins.
- A partial development run on G1 alone came first, written outside `outputs/costate`. It
  showed that `probes.wspearman` returns a round-off value, not "undefined", for a constant g.
  So the rank calibration's zero and direction-only plants had taken the rank route instead of
  amendment 1's fallback. The code was fixed to the amendment's text before the canonical run.
  No definition changed.

**Rows.** λ_ξ's 1,581 decision rows (approach phase, interior). λ_s's 1,643 rows enter only
where the calibrations are compared. No braking or corner row is scored (decision #8, plan
R5).

**Wiring checks (all passed).**
- **The tail.** Its counts equal findings Section 7.1's in all 9 (G1 286, G1s1 247, G1s2 259,
  G2 283, G2s1 315, G2s2 229, G3 240, G3s1 336, G3s2 294). The table's other columns agree to
  their printed rounding.
- **The frozen C1.** It reproduces `decisions.json` (`r2c`, `stat`, `ci`, the per-fold `a` and
  `b`) to at most 3.6·10⁻¹⁵. The rebuilt decision masks have the recorded sha256.
- **The fresh VJP.** Its `critic.pullback` equals the stored g bit for bit. The channel groups
  sum to the stored g_ξ within 5.3·10⁻¹² relative, and the accel term is at most 1.1·10⁻¹².

### Where the tail sits

| Network | Geometries holding half the tail's squared error (share) | Geometry 38: share, row enrichment | κ-clipped rows (tail / bulk), enrichment | s − s_apex, median (m), tail / bulk | max tyre util, median (mean), tail / bulk |
|---|---|---|---|---|---|
| G1 | 36 (0.37), 38 (0.11), 31 (0.07) | 0.11, 3.2 | 14 / 40, 1.58 | −75 / −98 | 0.94 (0.84) / <0.001 (0.25) |
| G1s1 | 38 (0.16), 36 (0.15), 0 (0.12), 8 (0.07), 28 (0.07) | 0.16, 2.0 | 14 / 40, 1.89 | −67 / −99 | 0.91 (0.84) / <0.001 (0.28) |
| G1s2 | 38 (0.27), 31 (0.15), 36 (0.13) | 0.27, 3.1 | 11 / 43, 1.31 | −74 / −99 | 0.94 (0.86) / <0.001 (0.26) |
| G2 | 36 (0.43), 38 (0.09) | 0.09, 4.6 | 8 / 16, 2.29 | −64 / −101 | 0.94 (0.85) / <0.001 (0.24) |
| G2s1 | 38 (0.22), 36 (0.18), 31 (0.13) | 0.22, 3.1 | 7 / 16, 1.76 | −72 / −100 | 0.95 (0.84) / <0.001 (0.23) |
| G2s2 | 38 (0.46), 28 (0.15) | 0.46, 5.0 | 0 / 23, 0 | −80 / −96 | 0.90 (0.81) / <0.001 (0.29) |
| G3 | 36 (0.32), 38 (0.32) | 0.32, 4.7 | none | −76 / −97 | 0.95 (0.85) / <0.001 (0.27) |
| G3s1 | 38 (0.26), 36 (0.18), 34 (0.12) | 0.26, 2.9 | none | −77 / −99 | 0.93 (0.81) / <0.001 (0.24) |
| G3s2 | 36 (0.32), 38 (0.27) | 0.27, 3.7 | none | −81 / −95 | 0.87 (0.77) / <0.001 (0.28) |

The geometries are (R, θ): 0 (6.0 m, 90°), 8 (9.4 m, 180°), 28 (22.9 m, 180°), 31 (35.8 m, 90°),
34 (40.0 m, 180°), 36 (44.7 m, 90°) and 38 (44.7 m, 180°). Medians and means are key-weighted.
"Max tyre util" is max(util_f, util_r) as the observation caps it. It is a descriptive companion
added by amendment 1, with no expected direction.

- **Two geometries.** Two to five geometries hold half of the tail's squared error. Geometry 38
  is among them in all 9 networks. Geometry 36, the other 44.7 m corner (θ 90° rather than 180°),
  is among them in 8.
- **The late approach.** 84–99% of the tail's squared error lies 125–50 m before the apex, the
  bins of s − s_apex from −125 to −50 m. The tail's median s − s_apex is −64 to −81 m, against −95
  to −101 m on the bulk.
- **High tyre utilisation.** On the tail rows the larger tyre utilisation has a key-weighted
  median of 0.87–0.95, against below 0.001 on the bulk. No row on either side is at the
  observation's tyre clip (util ≥ 2).
- **No preview-spacing kink.** On these rows vx lies between 30.1 and 81.0 m/s, so no row lies
  within 1 m/s of 25 or 83.3 m/s, and both kink flags are never set. The tail's median vx is about
  1 m/s above the bulk's in all 9 (56.3–56.8 against 55.6 m/s). The weighted means go either way:
  the tail's is lower in 4.
- **Clipping is κ clipping only.** On these rows every clipped channel is a preview κ channel; no
  channel that ξ moves is ever clipped. κ clipping touches 23–54 rows in 6 networks and none in
  G3, G3s1 and G3s2. Its row enrichment exceeds 1.5 in 4 networks, but those rows carry at most
  3.9% of the tail's squared error.
- **No tyre kink.** As amendment 1 anticipated, the tyre-kink flag is never set: channels
  212–215 do not read ξ.

### The critic against the map

g_ξ is split by a fresh VJP of the main V into channel groups. Shares are of |g_ξ| and signed:
a group that pulls against the total has a negative share, and the shares sum to 1. The
lateral channel 204 contributes nothing, since J_z[204, ξ] = 0, so the third column is the ξ
channel 205 alone.

| Network | Tail: near / far half of the edge poses / ξ channel | Bulk: near / far / ξ channel | ‖∂V_c/∂z‖ on ξ's channels, tail / bulk | ‖J_z e_ξ‖, tail / bulk | median \|g_ξ\|, tail / bulk | near group's sign opposite λ_ξ, tail rows |
|---|---|---|---|---|---|---|
| G1 | 0.82 / 0.00 / 0.18 | 0.44 / 0.34 / 0.22 | 3.19 | 0.97 | 90.9 / 2.7 | 0.90 |
| G1s1 | 0.86 / 0.01 / 0.13 | 0.74 / −0.09 / 0.35 | 3.83 | 0.96 | 108.2 / 5.1 | 0.94 |
| G1s2 | 0.88 / 0.04 / 0.09 | 0.74 / −0.14 / 0.41 | 2.42 | 0.97 | 62.0 / 4.2 | 0.91 |
| G2 | 0.84 / 0.00 / 0.16 | 0.41 / 0.42 / 0.17 | 2.84 | 0.94 | 72.2 / 3.3 | 0.92 |
| G2s1 | 0.82 / 0.01 / 0.18 | 0.86 / −0.08 / 0.22 | 3.07 | 0.96 | 108.8 / 3.2 | 0.91 |
| G2s2 | 0.82 / 0.04 / 0.14 | 0.78 / 0.04 / 0.18 | 2.54 | 0.98 | 64.2 / 3.6 | 0.91 |
| G3 | 0.82 / 0.04 / 0.15 | 0.82 / −0.04 / 0.22 | 2.45 | 0.97 | 64.6 / 4.0 | 0.91 |
| G3s1 | 0.82 / 0.02 / 0.16 | 1.37 / −0.34 / −0.02 | 3.40 | 0.98 | 81.7 / 3.3 | 0.92 |
| G3s2 | 0.77 / 0.04 / 0.20 | 1.30 / −0.31 / 0.01 | 3.08 | 1.00 | 100.5 / 3.7 | 0.87 |

The size columns are ratios of key-weighted medians, tail over bulk. The near half is preview
points 1–20; the far half is points 21–40.

- **The critic, not the map.** On tail rows the critic's gradient on ξ's channels is 2.4–3.8
  times its size on bulk rows, while the map's ξ column is 0.94–1.00 times. |g_ξ| itself is 15–34
  times larger. Most of the rest is alignment: the median |cosine| between the critic's gradient
  and the map's ξ column is 0.11–0.13 on the tail, against 0.010–0.019 on the bulk (from
  `rows.npz`).
- **The near preview, not the far.** On tail rows the near half of the edge-pose channels
  carries 77–88% of |g_ξ|, the far half 0–4%, and the ξ channel 9–20%.
- **The wrong sign comes from the near preview.** On 87–94% of tail rows, the near half's
  contribution has the sign opposite to λ_ξ.

### Other calibrations

R²_c is R²_wd for ξ. The rows, weights, folds and bootstrap are the frozen ones, and CIs are
95%. The bulk and tail splits are in the summary.

| Network | ξ: frozen | ξ: Huber [CI] | ξ: rank [CI] | s: frozen | s: Huber | s: rank |
|---|---|---|---|---|---|---|
| G1 | −0.002 | −0.030 [−0.138, 0.006] | −0.664 [−1.125, −0.431] | 0.021 | 0.019 | −0.768 |
| G1s1 | −1.351 | −0.000 [−0.099, 0.037] | −0.274 [−0.771, 0.050] | −0.102 | −0.034 | −0.932 |
| G1s2 | −2.706 | −0.020 [−0.129, 0.018] | −0.268 [−0.621, −0.029] | −0.101 | −0.026 | −0.875 |
| G2 | 0.006 | −0.032 [−0.139, 0.007] | −0.588 [−0.885, −0.458] | −0.103 | −0.052 | −0.955 |
| G2s1 | −4.984 | −0.011 [−0.116, 0.024] | 0.043 [−0.276, 0.252] | 0.029 | 0.020 | −0.701 |
| G2s2 | −0.320 | −0.024 [−0.131, 0.012] | −0.175 [−0.399, −0.043] | −0.015 | −0.035 | −0.854 |
| G3 | −2.679 | −0.026 [−0.130, 0.012] | −0.154 [−0.480, 0.070] | −0.038 | 0.027 | −0.495 |
| G3s1 | −5.003 | 0.002 [−0.094, 0.036] | −0.399 [−0.793, −0.172] | −0.147 | 0.007 | −0.709 |
| G3s2 | −2.709 | −0.032 [−0.138, 0.006] | −0.213 [−0.601, 0.014] | −0.063 | −0.019 | −0.807 |

- **The plants.**
  - The frozen calibration passes its plants in all 9: each plant scores −0.015 on s and −0.011
    on ξ.
  - Huber fails in all 9. Every plant scores −0.055 on ξ, below −0.02, because its intercept is
    the Huber location of the skewed λ_ξ rather than its mean. That is why design Section 13
    moved C1's calibration off Huber.
  - Rank fails in all 9. Its noise plant scores −0.49 on s and −0.40 on ξ: a chance positive
    Spearman in a training fold maps noise onto λ's whole distribution. Its other plants score
    −0.015 and −0.011.
- **Huber's ξ is near 0 because it barely reads g.** Its ξ R²_c is −0.032 to 0.002 in all 9. That
  is above the frozen one in the 7 networks with large slopes, because its per-fold ξ slopes are
  only 0.0007–0.0046, against fold-mean slopes of 0.0035–0.039 for the frozen calibration. So its
  predictions stay near its intercepts. Its ξ scores lie 0.02–0.06 above its own zero plant.
- **Rank is far below.** It scores −0.66 to 0.04 on ξ and −0.96 to −0.50 on s. With weak rank
  agreement, the quantile map hands the predictions λ's full spread.
- **The tail is worse under every calibration.** On the tail rows ξ scores −0.12 to −0.65 under
  Huber, −1.0 to −3.1 under rank, and −0.74 to −88 under the frozen calibration.

### Expected directions against the results

- **E1.1** ("at least half of its squared error lies in at most 6 of the 48 geometries, with
  geometry 38 among them in at least 6 of the 9 networks"): **holds.** Two to five geometries hold
  half, and geometry 38 is among them in 9 of 9.
- **E1.2** ("its rows have higher vx than the bulk, and the kink band around 83.3 m/s is enriched
  (ratio > 1.5) in at least 6 of 9. Neither observation clipping nor the tyre kinks are enriched
  (ratio < 1.5)"): **does not hold.**
  - The tail's median vx is higher in 9 of 9, but only by about 1 m/s, and the means go either
    way.
  - The 83.3 m/s band holds no row: these rows' speeds stop at 81.0 m/s.
  - κ clipping, the only clipping on these rows, is enriched above 1.5 in 4 of 9, so "not
    enriched" holds in 5 of 9, one short of 6. It carries at most 3.9% of the tail's error.
  - The tyre-kink flag is never set.
- **E1.3** ("the far half of the edge-pose channels carries more than half of |g_ξ| on tail rows,
  and less than half on bulk rows. ‖∂V_c/∂z‖ on the ξ channels is larger on tail rows than on bulk
  rows (ratio > 2), while ‖J_z e_ξ‖ differs less (ratio < 2)"): **does not hold, in its preview
  part.**
  - The critic-against-map part holds in 9 of 9: critic ratio 2.4–3.8, map ratio 0.94–1.00.
  - The far half carries more than half on tail rows in 0 of 9: it is the near half, at 77–88%.
- **E1.4** ("Under the Huber and rank calibrations, ξ's R²_c rises toward the bulk's (at least
  −0.05 in all 9) but stays below 0.2 in all 9. s moves by less than 0.03"): **does not hold** for
  either calibration.
  - Huber: ξ is at least −0.05 and below 0.2 in 9 of 9, but its s moves by 0.03 or more in 6 of 9.
  - Rank: ξ is at least −0.05 in 1 of 9, and s moves by 0.03 or more in 9 of 9.

**The rule for later items** (plan CS-1): it **does not fire.** Neither alternative passes its
plants in any network. Huber would otherwise have raised ξ's R²_c by more than 0.1 in 7 of 9. So
the later items report R²_c under the frozen calibration only.

### What CS-1 says (exploratory)

- **The tail is in the critics' gradients, not in the observation map.**
  - The map's ξ column has the same size on tail and bulk rows.
  - None of ξ's channels is clipped on these rows, no row is near a preview-spacing kink, and
    none is at the tyre clip.
- **Where it is.** It sits on the late approach to the two 44.7 m corners (about 125–50 m before
  the apex), at states with high tyre utilisation.
- **What carries it.** The critics' gradients with respect to the near half of the edge preview
  are larger there, much better aligned with the map's ξ column, and of the sign opposite to λ_ξ.
- **The refute stands.** No calibration that passes its plants brings any network's ξ R²_c near
  0.2: only the frozen one passes, and its bulk R²_c is at most 0.092 (findings Section 7.1).
  CS-1 changes no decision.

## CS-2 to CS-5: the runs (exploratory)

- `scripts/costate_exploratory.py cs-2`, `cs-3`, `cs-4` and `cs-5` at `b62ead3`, one after another,
  niced, under the plan with amendment 2 (`f95456f`: CS-2 to CS-5's operational details,
  committed before any of their numbers). CS-4 and CS-5 ran in pools of 4 workers × 2 XLA
  threads, BLAS pinned to one thread. The Mac was shared with another workflow throughout
  (load average 30–60, swap nearly full), so the wall times below are long.
- Every stamp records an empty `changed_vs_head`, no partial flag, the plan's sha256 with 2
  amendments, and the sha256 of every input: CS-2 104 files (the world, all 34 networks' critic
  and K3 files and their checkpoints through the record's pins), CS-3 146 (CS-2's, plus the 29
  step 2 rows files, the optimum's rows and the 12 step 1 run tables), CS-4 and CS-5 38 each.
- **Wiring checks (all passed).** In every item and network the frozen C1 was recomputed first
  and reproduced `decisions.json` (r2c, stat, ci, the per-fold a and b of s and ξ) to at most
  3.6·10⁻¹⁵; the 9 decision networks' rebuilt masks have the recorded sha256; CS-2's heads give
  C1-heads' D of g_s and g_ξ to within 10⁻¹⁶; CS-4's probes agree with `analysis.delta_c1` exactly;
  CS-5's pool actor of seed 0, rebuilt from its PPOConfig, gives its stored h3 exactly.
- **Development runs first.** Partial runs, written outside `outputs/costate`, came before the
  canonical ones: CS-2, CS-4 and CS-5 on G1 alone, and CS-3 on the first 3 networks. They
  found no fault. After them only two things changed, neither a definition: `--limit` for CS-2
  and CS-3, and a filter for a harmless numpy warning in CS-3.
- **No calibration but the frozen one.** CS-1's rule for later items did not fire, so every
  R²_c below is the frozen calibration's.

## CS-2. The descriptives pack (exploratory)

**Run.** 381 s (the stamps' start to end), all 34 networks. Output: `outputs/costate/exploratory/cs-2/summary.json`
(committed copy: [`costate-followups/cs-2-summary.json`](costate-followups/cs-2-summary.json)).
**Rows.** C1's rows (λ_s 1,643 and λ_ξ 1,581 rows in the 9 decision networks), D's 477 rows, and
for the accel term (no target) the interior.

| Network | s: R²_c / unwinsorised | ξ: R²_wd / raw R² / floor | ρ_s / ρ_ξ | heads' C1 statistic (largest CI upper end) | heads − main V | C1 on D [CI] | SD over folds, s / ξ | accel share of g_vx |
|---|---|---|---|---|---|---|---|---|
| G1 | 0.021 / 0.021 | −0.002 / 0.341 / 0.335 | 0.990 / 0.990 | −0.001 to 0.014 (0.034) | −0.01 to 0.00 | −0.017 [−0.112, −0.007] | 0.20 / 0.04 | 0.051 |
| G1s1 | −0.102 / −0.101 | −1.351 / −0.547 / 0.335 | 0.960 / 0.985 | −1.136 to −0.938 (−0.519) | −0.41 to −0.21 | 0.057 [−0.081, 0.123] | 0.08 / 0.98 | 0.042 |
| G1s2 | −0.101 / −0.100 | −2.706 / −1.438 / 0.335 | 0.988 / 0.989 | −1.858 to −1.514 (−0.655) | −0.45 to −0.11 | 0.103 [−0.005, 0.161] | 0.26 / 1.53 | 0.039 |
| G2 | −0.103 / −0.102 | 0.006 / 0.346 / 0.335 | 0.996 / 0.996 | −0.062 to −0.053 (0.015) | −0.01 to −0.00 | −0.017 [−0.112, −0.007] | 0.19 / 0.05 | 0.041 |
| G2s1 | 0.029 / 0.029 | −4.984 / −2.937 / 0.335 | 0.975 / 0.973 | −2.934 to −2.612 (−1.186) | −0.46 to −0.13 | 0.067 [−0.095, 0.140] | 0.19 / 2.37 | 0.039 |
| G2s2 | −0.015 / −0.015 | −0.320 / 0.131 / 0.335 | 0.987 / 0.992 | −0.181 to −0.086 (−0.016) | −0.01 to 0.08 | −0.017 [−0.112, −0.007] | 0.16 / 0.32 | 0.044 |
| G3 | −0.038 / −0.037 | −2.679 / −1.421 / 0.335 | 0.994 / 0.994 | −1.718 to −1.507 (−0.442) | −0.36 to −0.15 | 0.063 [−0.050, 0.115] | 0.38 / 2.10 | 0.047 |
| G3s1 | −0.147 / −0.145 | −5.003 / −2.949 / 0.335 | 0.966 / 0.969 | −3.879 to −2.690 (−1.207) | −1.30 to −0.12 | −0.025 [−0.134, 0.011] | 0.58 / 2.60 | 0.032 |
| G3s2 | −0.063 / −0.062 | −2.709 / −1.440 / 0.335 | 0.991 / 0.994 | −1.703 to −1.606 (−0.591) | −0.32 to −0.22 | −0.030 [−0.150, 0.036] | 0.13 / 1.98 | 0.038 |

ξ's unwinsorised R² equals its R²_wd in every network: λ_ξ lies inside its winsorisation bounds
on all 1,581 rows. The floor is a target-only prediction on rows shared by the 9, so it is the
same 0.335 in each. "Accel share" is the key-weighted mean of the per-state share on C1's rows.

- **The heads agree with each other.** ρ_k is 0.960–0.997 for s and 0.969–0.996 for ξ over all
  31 generalists (ξ within direction: 0.974–0.997 in the 9), so √ρ_k ≥ 0.98. No head's
  |corr(g_k, λ_k)| comes near it: the main V's is −0.05 to 0.22 for s and 0.16 to 0.24 for ξ in
  the 9, and no head's exceeds 0.25 anywhere. On the whole interior, ρ of all 8 components is
  at least 0.959.
- **Every head refutes.** All 155 heads' C1 statistics have CI upper ends below 0.2 (the largest
  is 0.034). But the heads follow the main V only where the main V is near 0. Where the ξ tail
  drives the main V far below 0, the heads lie 0.1 to 1.3 below it. 53 of the 155 heads are
  within 0.1 of their main V.
- **Winsorisation hardly matters.** Unwinsorised R² differs from R²_c by −0.0004 to 0.0014 for s
  over all 34 networks, and not at all for ξ.
- **Raw R² of ξ.** It sits on the floor (0.341 and 0.346 against 0.335) only in G1 and G2, the
  two networks whose ξ R²_c is near 0. In the other 7 it is 0.20 to 3.3 below it: their
  calibrated ξ predictions spread wider than the direction means do.
- **The folds.** The SD over the 6 folds is 0.08–0.58 for s and 0.04–2.60 for ξ; for ξ it
  tracks the size of the tail. G3s1's per-fold ξ R²_wd runs from −9.8 to −1.6.
- **C1 on D.** All 34 networks are in the refute band (CI upper ends 0.161 at most among the 9,
  0.177 among the others). In G1, G2 and G2s2 every fold's slope on D is clamped at 0 for both s
  and ξ, so they score the intercept-only fit's −0.017. The interior-fitted calibration scored on
  D does worse: −0.030 to 0.007 for s and −0.142 to −0.012 for ξ.
- **The accel term is small.** On C1's rows it carries 3.2–5.1% of |g_vx| per state on average in
  the 9 (2.6–5.7% in the others), 6.9–10.0% on the rows with t ≥ 1 (D's reset rows carry none),
  and 5.5–7.9% on the whole interior.
- **The accel-channel sensitivity rises with tyre utilisation.** On C1's rows its weighted
  Spearman with util_max is 0.22 to 0.67 in the 9, and its median in the top util bin (≥ 1) is
  1.3 to 9.2 times that in the bottom one. On the whole interior, braking and corner rows
  included, the Spearman is weaker (−0.13 to 0.37).
- **The specialists** (no heads): R²_c of s −0.03 to −0.02 and of ξ 0.04 to 0.16; C1 on D 0.015
  to 0.111, all in the refute band.

**Expected directions against the results.**
- **E2.1** ("ρ_k ≥ 0.9 for s and ξ in every generalist ... So the ceiling √ρ_k does not bind"):
  **holds**, in 31 of 31.
- **E2.2** ("Every head's R²_c is in the refute band (CI upper end < 0.2), and the heads' R²_c
  agree with the main V's to within 0.1"): **does not hold.** The refute-band part holds for all
  155 heads; the agreement part for 53 of them.
- **E2.3** ("Unwinsorised R² is within 0.05 of R²_c for s, and lower than R²_c for ξ"): **does not
  hold**, in 0 of 9. The s part holds in all 34 networks. ξ's is equal, never lower, because no
  ξ decision row is winsorised.
- **E2.4** ("On ξ, raw R² lies near the direction-only floor (within 0.1)"): **does not hold**:
  2 of 9 (G1, G2), and 4 of the other 25.
- **E2.5** ("C1 on D is in the refute band"): **holds**, in 9 of 9 and in all 25 others.
- **E2.6** ("The accel term carries under 10% of |g_vx| on C1's rows"): **holds**, in 9 of 9 and in
  all 25 others.

**What CS-2 says (exploratory).** The five heads agree on their gradients (ρ about 0.99) and
none of them agrees with λ. The refute is not head-level indeterminacy: in design Section 6's
words, the gradient is determined but wrong, given the trunk. The heads do differ from the main
V in how far the ξ tail drags R²_c below 0, but not in the band. The accel term is too small to
matter for g_vx, and the critic's sensitivity to the accel channels grows with tyre
utilisation.

## CS-3. C3-int on s and ξ (exploratory)

**Run.** 174 s. Outputs: `outputs/costate/exploratory/cs-3/summary.json` (committed copy:
[`costate-followups/cs-3-summary.json`](costate-followups/cs-3-summary.json)) and `tables.npz`
(the residual and behaviour tables and the null draws, local).

**Rows and tables.**
- The residuals: every network's 48 geometries hold C1 rows of both s and ξ.
- Step 2's G − 1 and B: the 29 generalists; 327–419 of T's keys per network survive step 2's own
  exclusions, covering 45–48 geometries.
- Step 1's measures: the 9 generalist finals (and S1–S3); 471 keys each. The entry speed covers
  all 48 geometries. The brake onset covers 37 (445 keys): keys without a sustained brake have
  none.
- The specialists crash on most synthetic keys (289–438 of 471 deterministic runs), the
  generalists on 1–24.

**Results.** The correlation r is Pearson over the cells both double-centred tables hold. p is
two-sided against 10,000 blocked permutations; "null" is the permutation null's 2.5% and 97.5%
quantiles.

| Pair | Networks | Cells | r | p | Holm p | Null |
|---|---|---|---|---|---|---|
| s, G − 1 | 29 | 1,364 | 0.081 | 0.018 | 0.130 | [0.028, 0.079] |
| s, B | 29 | 1,364 | −0.027 | 0.234 | 0.720 | [−0.042, 0.005] |
| s, entry-speed ratio − 1 | 9 | 432 | 0.109 | 0.016 | 0.130 | [−0.025, 0.103] |
| s, brake-onset residual | 9 | 333 | −0.034 | 0.619 | 1 | [−0.127, 0.048] |
| ξ, G − 1 | 29 | 1,364 | −0.078 | 0.127 | 0.720 | [−0.091, −0.030] |
| ξ, B | 29 | 1,364 | 0.033 | 0.120 | 0.720 | [−0.010, 0.043] |
| ξ, entry-speed ratio − 1 | 9 | 432 | −0.078 | 0.165 | 0.720 | [−0.109, 0.016] |
| ξ, brake-onset residual | 9 | 333 | 0.003 | 0.960 | 1 | [−0.042, 0.119] |
| *With S1–S3 (descriptive):* | | | | | | |
| s, entry-speed ratio − 1 | 12 | 576 | 0.034 | 0.157 | – | [−0.036, 0.054] |
| s, brake-onset residual | 12 | 444 | 0.051 | 0.189 | – | [−0.051, 0.085] |
| ξ, entry-speed ratio − 1 | 12 | 576 | −0.018 | 0.495 | – | [−0.058, 0.032] |
| ξ, brake-onset residual | 12 | 444 | −0.047 | 0.183 | – | [−0.070, 0.069] |

The residual is (calibrated prediction − λ), σ-folded for ξ, so a positive s residual is a
calibrated s-gradient above λ_s. The brake-onset residual is s_b − s_b*, negative for earlier
braking.

- **No pair is sizeable.** The largest |r| is 0.109. After Holm's adjustment no pair has p ≤ 0.05;
  the smallest adjusted p is 0.130.
- **Two unadjusted p below 0.05**, both on s: G − 1 (r 0.081, p 0.018) and the entry speed (r
  0.109, p 0.016). Where a network's calibrated s-gradient lies unusually above λ_s, its G and
  its entry speed are a little higher than usual for it.
- **The nulls are not centred on 0.** The permutation keeps each geometry within its R tercile ×
  θ class block, so an association at the level of the blocks survives into the null. s with
  G − 1 has a null band of [0.028, 0.079], and its r of 0.081 lies just above it.
- **With the specialists** the entry-speed pairs shrink (|r| 0.034 and 0.018) and the
  brake-onset pairs stay small (|r| ≤ 0.051), though both grow; every p ≥ 0.157.

**Expected direction against the result.**
- **E3.1** ("Every |r| < 0.15, and no pair has a Holm-adjusted p ≤ 0.05"): **holds.**

**What CS-3 says (exploratory).** Within a network, the geometries where its critic's
calibrated s- or ξ-gradient is unusually far from λ are not, to any useful degree, the ones
where its line, buffer, entry speed or braking point is unusually far from the optimum's. This
says nothing about n and vx, the pairs the theory signs, which have no decision rows (decision
#8).

## CS-4. ΔC1, R²_z and the MLP-on-z ceiling (exploratory)

**Run.** 653 s, 18 units (network × component) on 4 workers; each unit's MLP took 78–171 s.
Outputs: `outputs/costate/exploratory/cs-4/summary.json` (committed copy:
[`costate-followups/cs-4-summary.json`](costate-followups/cs-4-summary.json)) and `preds.npz`
(the out-of-fold predictions, local). **Rows.** C1's rows (λ_s 1,643, λ_ξ 1,581), C1's weights,
partition 0's folds.

| Network | s: R²_z / R²_MLP / R²(z ⊕ g) | s: ΔC1 [CI] | s: R²_c / R²_MLP | ξ: R²_z / R²_MLP / R²(z ⊕ g) | ξ: ΔC1 [CI] | ξ: R²_c / R²_MLP |
|---|---|---|---|---|---|---|
| G1 | 0.355 / 0.645 / 0.350 | −0.005 [−0.028, 0.012] | 0.033 | 0.663 / 0.830 / 0.660 | −0.003 [−0.023, 0.010] | −0.002 |
| G1s1 | 0.355 / 0.641 / 0.359 | 0.004 [−0.005, 0.021] | −0.159 | 0.668 / 0.819 / 0.668 | −0.000 [−0.000, 0.000] | −1.648 |
| G1s2 | 0.355 / 0.642 / 0.355 | −0.000 [−0.028, 0.033] | −0.157 | 0.669 / 0.821 / 0.668 | −0.001 [−0.002, 0.001] | −3.295 |
| G2 | 0.354 / 0.647 / 0.374 | 0.020 [−0.006, 0.065] | −0.160 | 0.671 / 0.838 / 0.671 | 0.001 [−0.003, 0.005] | 0.007 |
| G2s1 | 0.354 / 0.651 / 0.337 | −0.017 [−0.071, 0.014] | 0.045 | 0.671 / 0.832 / 0.668 | −0.003 [−0.013, 0.005] | −5.991 |
| G2s2 | 0.354 / 0.647 / 0.350 | −0.004 [−0.013, 0.003] | −0.023 | 0.671 / 0.817 / 0.672 | 0.001 [−0.012, 0.013] | −0.392 |
| G3 | 0.354 / 0.645 / 0.353 | −0.000 [−0.013, 0.012] | −0.059 | 0.673 / 0.832 / 0.671 | −0.001 [−0.011, 0.009] | −3.219 |
| G3s1 | 0.354 / 0.641 / 0.376 | 0.022 [−0.012, 0.061] | −0.229 | 0.673 / 0.829 / 0.666 | −0.007 [−0.072, 0.035] | −6.033 |
| G3s2 | 0.354 / 0.644 / 0.342 | −0.012 [−0.039, 0.004] | −0.097 | 0.673 / 0.845 / 0.668 | −0.005 [−0.019, 0.010] | −3.206 |

R² is R²_wd for ξ. g is the main V's 8 pulled-back components.

- **λ_s and λ_ξ are decodable from z.** A ridge on z gives 0.354–0.355 for s (CI about [−0.02,
  0.57]) and 0.663–0.673 for ξ ([0.47, 0.81]). The values hardly differ between networks
  because the probes z-score each feature on the training fold, so a network's obs_rms changes
  only which channels clip at ±10 (as K4's PR-7 intends).
- **The MLP ceiling is higher.** One hidden layer of 64 tanh units on z reaches 0.641–0.651 for s
  (CIs from about 0.35 to 0.86) and 0.817–0.845 for ξ (0.64 to 0.95): 0.29 above the ridge for s
  and 0.15–0.17 for ξ, CIs above 0 in every network.
- **The critic's gradient adds nothing to z.** ΔC1 is −0.017 to 0.022 for s and −0.007 to 0.001
  for ξ, every CI covering 0.
- **The normalised fidelity is near or below 0.** R²_c / R²_MLP is −0.23 to 0.045 for s and −6.0 to
  0.007 for ξ.

**Expected directions against the results.**
- **E4.1** ("R²_z is at least 0.3 for ξ and at least 0.2 for s"): **holds**, in 9 of 9 for both.
- **E4.2** ("The MLP ceiling is above R²_z by at least 0.05, and above 0.4 for ξ"): **holds**, in 9
  of 9.
- **E4.3** ("R²_c/R²_MLP ≤ 0.1 in every network"): **holds**, in 9 of 9 for both components.
- **E4.4** ("ΔC1 ≤ 0.02 in every network, with a CI that covers 0"): **does not hold**, by a
  hair: G3s1's s ΔC1 is 0.022 (CI [−0.012, 0.061]). Every other network and component holds;
  G2's s is 0.0198.

**What CS-4 says (exploratory).** The refute is not about the observation. What the networks
see determines λ_ξ well (R² about 0.67 linearly, 0.83 with one hidden layer) and λ_s moderately
(0.35, 0.65), on the same rows where the critics' calibrated gradients score near or below 0.
Adding the critic's own gradient to z does not improve a probe of λ. The failure is in the
critic's gradient: λ is there to be read from the input, and the critic's input gradient does
not carry it.

## CS-5. W1 on the Π targets (exploratory)

**Run.** 1,195 s, 9 units (one per network) on 4 workers, about 7 min each. Outputs:
`outputs/costate/exploratory/cs-5/summary.json` (committed copy:
[`costate-followups/cs-5-summary.json`](costate-followups/cs-5-summary.json)) and `preds.npz`
(local). **Rows.** T's interior rows of the approach and exit phases, 1,581 + 72 = 1,653, the
same in every network, under the key weights renormalised on them. The targets are winsorised
at their frozen bounds and scored by raw R².

**Decodability from z** (the same in every network to the third decimal, as in CS-4):

| Target | R²_z [CI] | Spearman, z | R²(z ⊕ actor h3) | R²(z ⊕ critic h3) | Pool's median raw gain over z, actor / critic |
|---|---|---|---|---|---|
| Π^in | 0.256 [0.036, 0.387] | 0.53 | 0.841–0.888 | 0.892–0.931 | 0.612–0.621 / 0.612–0.621 |
| Π^out | 0.419 [0.106, 0.557] | 0.62 | 0.724–0.798 | 0.756–0.872 | 0.361–0.378 / 0.361–0.378 |
| Π^slip | 0.742 [0.594, 0.817] | 0.89 | 0.876–0.899 | 0.889–0.915 | 0.150–0.160 / 0.150–0.160 |
| Π^μ | 0.529 [0.331, 0.652] | 0.79 | 0.760–0.822 | 0.784–0.864 | 0.270–0.283 / 0.270–0.283 |

The pool's columns are the median over each network's 20 controls of R²(z ⊕ control h3) −
R²_z: the 20 pool actors for the actor, 20 random-init critics for the critic. Ranges are over
the 9 networks. *Note added at the close of the lane:* the two control sets give bit-identical
h3 gains in all 36 (network, target) cells. A random-init critic and a pool actor share the
trunk architecture and the seed, so their h3 is the same. The actor's and the critic's h3 are
therefore compared with one shared random-feature control.

**Gains beyond z, against the controls** (the trained h3's gain minus the controls' median;
two-level CI; rank among the 21):

| Network | actor: Π^in | Π^out | Π^slip | Π^μ | critic: Π^in | Π^out | Π^slip | Π^μ |
|---|---|---|---|---|---|---|---|---|
| G1 | 0.012 | −0.003 | −0.006 | −0.020 | 0.050 [0.030, 0.079] 1 | **0.079 [0.043, 0.129] 1** | 0.011 | **0.052 [0.026, 0.097] 1** |
| G1s1 | −0.001 | −0.039 | 0.002 | −0.016 | 0.039 [0.013, 0.070] 1 | 0.045 [0.006, 0.104] 1 | 0.009 | 0.032 [0.002, 0.064] 2 |
| G1s2 | −0.027 | −0.008 | −0.006 | −0.022 | 0.023 | 0.030 [0.012, 0.072] 2 | 0.015 [0.001, 0.036] 2 | 0.033 |
| G2 | −0.032 | −0.044 | −0.002 | −0.041 | 0.034 [0.012, 0.062] 1 | 0.021 | −0.007 | **0.052 [0.013, 0.092] 1** |
| G2s1 | −0.016 | −0.036 | −0.012 | −0.022 | 0.037 [0.013, 0.069] 1 | −0.018 | 0.015 | 0.010 |
| G2s2 | −0.011 | −0.030 | 0.001 | −0.021 | 0.021 [0.000, 0.047] 1 | −0.025 | −0.006 | −0.016 |
| G3 | 0.010 | 0.018 | −0.019 | 0.023 | **0.056 [0.037, 0.085] 1** | **0.092 [0.048, 0.152] 1** | 0.006 | 0.043 |
| G3s1 | −0.019 | −0.056 | −0.004 | −0.023 | 0.038 [0.016, 0.068] 1 | 0.044 [0.000, 0.097] 1 | 0.023 [0.004, 0.049] 1 | 0.041 |
| G3s2 | 0.007 | 0.012 | −0.013 | 0.007 | 0.026 [0.010, 0.051] 1 | 0.021 | 0.016 | 0.025 |

CIs and ranks are shown where the CI's lower end is at or above 0; bold marks a statistic above
0.05 with its CI above 0. The summary holds every CI and rank.

- **Π is decodable from z, and more so nonlinearly.** z alone gives 0.26 for Π^in, 0.42 for
  Π^out, 0.74 for Π^slip and 0.53 for Π^μ. Adding any 256-unit hidden layer computed from z,
  trained or random, raises this a great deal (to 0.84–0.93 for Π^in): most of that is the
  random features' nonlinearity, since the controls gain almost as much.
- **The actor's h3 adds nothing beyond its controls.** Its gain over the pool's median is −0.056
  to 0.023; no CI lies above 0, and it ranks 18th to 21st of 21 in 4–6 networks per target.
- **The critic's h3 adds a little, consistently, on Π^in.** Beyond the random-init critics it
  gains 0.021–0.056 on Π^in, CI above 0 in 8 of 9 and first of 21 in 8. On Π^out it gains
  −0.025 to 0.092 (CI above 0 in 5), on Π^μ −0.016 to 0.052 (3), and on Π^slip −0.007 to 0.023
  (2).
- **Five gains pass 0.05 with their CI above 0**, all in the critic and all in the three seed-0
  networks: G1 on Π^out (0.079) and Π^μ (0.052), G2 on Π^μ (0.052), G3 on Π^in (0.056) and
  Π^out (0.092).
- **W1 on these rows.** The calibrated R² of the critic's inside-width derivative against Π^in is
  −0.046 to 0.176, every CI's lower end below 0 (upper ends 0.07–0.29); its Spearman is 0.37 to
  0.53. With the winsorised target it is the same. The confirmatory W1 on the whole interior
  (braking and corner rows included) was not re-scored here (decision #8).
- **|ΔΠ|/SD_T** awaits BC-a0 (amendment 2). K2b was never measured for the Π targets, so how much
  unseen road moves them on these rows is not yet known.
  - *Added at the close of the lane:* BC-a0 has since measured it. Its p90 is 0.79–1.02 on
    approach rows and 0.96–1.78 on exit rows (BC-a0 section, 'For CS-5 and CS-14').

**Expected directions against the results.**
- **E5.1** ("Π^in and Π^out are decodable from z (R² ≥ 0.3)"): **does not hold.** Π^out is, in 9
  of 9 (0.42); Π^in is not, in 0 of 9 (0.26).
- **E5.2** ("Neither hidden layer adds more than 0.05 beyond z for any Π target"): **does not
  hold.** The actor's h3 holds it for every target in 9 of 9. The critic's h3 exceeds 0.05 in 5
  network-target cases (above), each with its two-level CI above 0.
- **E5.3** ("Π^slip and Π^μ are less decodable than Π^in"): **does not hold**, in 0 of 9: they are
  more decodable (0.74 and 0.53 against 0.26).

**What CS-5 says (exploratory).**
- **The actor's h3 extends C2's result** from λ to the road and grip prices: no linear
  shadow-price information beyond what random-init actors' features of z give.
- **The critic's h3 is the exception, and it is small.** Five network-target cases pass the plan's
  "first sign" (a gain above 0.05 with its CI above 0), all in the critic and all in the seed-0
  network of each variant. On Π^in its gain is smaller (0.02–0.06) but consistent, and its CI is
  above 0 in 8 of 9.
- **What that does and does not show.** V is linear in the critic's h3, and the value of a state
  depends on how much room the road ahead leaves, so a layer that correlates with Π^in beyond z
  need not encode a price. It is still the first layer seen to hold information about a
  window-problem target that z and random features do not.
- **Two caveats.** The random-init critics are the only control. And Π's sensitivity to unseen
  road is unmeasured until BC-a0 reports |ΔΠ|/SD_T.
- **The calibrated agreement stays low.** On these rows the critic's own width derivative still
  agrees poorly with Π^in (calibrated R² at most 0.18, every CI reaching below 0), as W1 found on
  the interior.

## CS-2 to CS-5: where they leave the refute (exploratory)

- **The refute is about the critic's gradient.** It is not about head-level indeterminacy (CS-2:
  ρ ≈ 0.99), the observation (CS-4: λ_ξ is decodable from z at R² ≈ 0.67, 0.83 nonlinearly), the
  winsorisation (CS-2), the accel term (CS-2: under 6% of |g_vx|) or the decision states alone
  (CS-2: C1 on D refutes in all 34).
- **Nothing ties a network's specific gradient errors to its specific driving errors** (CS-3).
- **The critic's last hidden layer holds a little information about the wall and grip prices
  beyond z and random features** (CS-5), most consistently for Π^in, while its input gradient
  does not express it (W1 on these rows). That is the one result here that points away from "no
  shadow price anywhere". It is exploratory, and CS-7 (the critic's layers on λ) is the item that
  tests the same reading for the costate itself.
- None of this changes a decision of `costate-prereg-v1`.

## CS-10 and CS-11: the runs (exploratory)

- **CS-10 (ii).** `scripts/costate_exploratory.py cs-10` at `705c5c2`, under the plan with
  amendment 3 (`c9b79db`: CS-10's and CS-11's operational details on D, committed before any of
  their numbers). It ran 36 units (network × quarter of D's keys) on 4 workers × 2 XLA threads,
  niced, in 15 min (worker time 372–454 s per network). The plan's estimate was about 45 min
  plus 4 min per network. The cost had been measured on a random-init stand-in (amendment 3).
- **CS-11 on D.** `cs-11` at `b340a50`, under the plan with amendments 3–6, with the same pool,
  in 29 min (worker time 738–879 s per network). The plan's estimate was 15–50 min. The cost was
  measured on G1 first, in a development run written outside `outputs/costate` (amendment 4).
  Three earlier canonical runs stopped on wiring round-off and wrote nothing. Amendments 4–6
  (committed before each rerun) widened two cross-program tolerances to K3's 10⁻⁶, rescaled the
  dV_dwin check, and made the units stop at the first failure. Each records the single gap that
  run showed.
- **Stamps.** Both outputs record an empty `changed_vs_head`, no partial flag, the plan's sha256
  with its amendment count (3 for CS-10, 6 for CS-11), and the sha256 of every input:
  - CS-10, 47 files: the world and its summary, the 9 networks' critic and K3 files, their own and
    K3b files, and their checkpoints through the record's pins;
  - CS-11, 29 files: the same without the own and K3b files.

  They were re-verified before this note:
  - each stamp's script sha256 is the script's blob at its HEAD, and its plan sha256 is the plan
    at `c9b79db` (CS-10) or `d06f28e` (CS-11);
  - the `rows.npz` sha256s and the summaries' content sha256s were recomputed and match;
  - every input still has its stamped sha256, and the script's `verify` reports 0 problems.

  CS-10's code is unchanged from `705c5c2` to `b340a50`: amendments 4–6 touched CS-11's checks
  only.
- **The runs.** From each of D's 477 keys (x_0, a_{−1} = 0, accel 0), each network's
  deterministic policy drives for 800 steps.
  - 1–24 runs per network terminate within 800 steps and are dropped from both items. The counts
    are the same in both items: G1 9, G1s1 8, G1s2 24, G2 24, G2s1 17, G2s2 20, G3 1, G3s1 8,
    G3s2 12.
  - Every other run lasts the full 800 steps (none reaches the strip's end) and passes
    s_c1 + 50 m. At both of CS-11's horizons it keeps a positive inside buffer (median B
    0.96–1.55 m).
  - So CS-11's common rows, its rows kept at H and its rows kept at W are the same 453–476 per
    network, and its two companions repeat the primary.
- **Wiring checks (all passed).**
  - **CS-10.**
    - The confirmatory C1-own, recomputed on T's restarts, reproduces `decisions.json` exactly
      (gap 0 in r2c, stat, ci and the per-fold a and b of s and ξ).
    - The pullback at every kept run's state 0 equals the stored g at the key's D row bit for bit.
    - The restarts at ⌊0.25 H⌋ and ⌊0.6 H⌋ reproduce the sweep's λ^own and λ^own_held to at most
      1.5·10⁻¹⁰ of the row's largest |λ| (G2s2; tolerance 10⁻⁹).
  - **CS-11.**
    - The recomputed dV_dwin is within 9.0·10⁻¹⁵ of the stored one, as a share of the network's
      largest |dV_dwin| (tolerance 10⁻⁹).
    - The width loop's J matches `own_return`'s to at most 4.2·10⁻¹⁰ relative (G1s2; tolerance
      10⁻⁶).
    - The JVP matches the VJP's contraction to at most 5.6·10⁻⁸ (G2s2; tolerance 10⁻⁶), on 36 keys
      (one per unit).
    - **Descriptive.** On those 36 keys, Richardson differences of J_env at h = 10⁻³ m differ from
      the AD value by a median of 2·10⁻⁷ relative, at most 8.7·10⁻³ (one G2s1 key). Every perturbed
      run stays alive.
- **No calibration but the frozen one** (CS-1's rule for later items did not fire).

## CS-10. C1-own on own-run states (exploratory)

**Outputs.** `outputs/costate/exploratory/cs-10/summary.json` (committed copy:
[`costate-followups/cs-10-summary.json`](costate-followups/cs-10-summary.json)) and `rows.npz`
(per state, local).

**Rows.** The scored states are the kept runs' every-5th states up to each run's own t_end, in
the approach and exit phases, under the own interior flag: 4,079–4,445 per network.
- λ_s is scored on the 2,223–2,298 of them where s is observable. That is 13.6–14.1 times the
  confirmatory C1-own's 162–163 s rows.
- λ_ξ is scored on the approach rows, 1,814–1,853, which is 11.2–11.5 times its 161–162 ξ rows.
- No braking or corner row is scored (decision #8, plan R5). The own-run tables hold
  3,248–3,657 braking and 2,555–2,931 corner states, which are only counted.

| Network | runs kept (terminated) | rows s / ξ | R²_c s / ξ | C1-own statistic [CI] | CI width / confirmatory's | ξ Spearman [CI] | confirmatory C1-own [CI] |
|---|---|---|---|---|---|---|---|
| G1 | 468 (9) | 2,269 / 1,838 | −0.004 / −0.003 | −0.004 [−0.036, −0.002] | 0.130 | −0.05 [−0.16, 0.07] | −0.122 [−0.313, −0.048] |
| G1s1 | 469 (8) | 2,298 / 1,849 | −0.010 / −0.003 | −0.006 [−0.023, −0.004] | 0.001 | 0.19 [0.10, 0.26] | −0.054 [−31.190, −0.034] |
| G1s2 | 453 (24) | 2,238 / 1,820 | 0.003 / −0.009 | −0.003 [−0.040, 0.013] | 0.005 | 0.12 [0.01, 0.23] | −0.047 [−11.153, −0.034] |
| G2 | 453 (24) | 2,223 / 1,814 | −0.044 / −0.024 | −0.034 [−0.084, −0.021] | 0.194 | −0.12 [−0.25, 0.01] | 0.029 [−0.284, 0.046] |
| G2s1 | 460 (17) | 2,228 / 1,814 | −0.004 / −0.017 | −0.010 [−0.054, −0.008] | 0.081 | −0.04 [−0.17, 0.09] | 0.026 [−0.425, 0.146] |
| G2s2 | 457 (20) | 2,247 / 1,833 | −0.005 / −0.017 | −0.011 [−0.055, −0.007] | 0.044 | −0.15 [−0.24, −0.04] | −0.150 [−0.979, 0.115] |
| G3 | 476 (1) | 2,258 / 1,853 | −0.006 / −0.025 | −0.016 [−0.131, −0.012] | 0.180 | 0.21 [0.09, 0.31] | −0.126 [−0.624, 0.035] |
| G3s1 | 469 (8) | 2,270 / 1,849 | −0.017 / −0.038 | −0.028 [−0.100, 0.005] | 0.105 | 0.10 [0.01, 0.19] | −0.075 [−0.878, 0.113] |
| G3s2 | 465 (12) | 2,235 / 1,840 | 0.014 / 0.105 | 0.060 [−0.0005, 0.117] | 0.135 | 0.33 [0.19, 0.45] | −0.031 [−0.716, 0.152] |

The statistic is C1-own's (`analysis.c1_own`: λ^own raw, the frozen calibration, partition 0's
folds, key weights, the 48-geometry bootstrap). The last column is the confirmatory C1-own on
T's restarts (`decisions.json`, reproduced here).

- **The refute holds at the states the policies reach.** Every network's statistic is in the
  refute band; the largest CI upper end is 0.117 (G3s2). R²_c is −0.044 to 0.014 for s and
  −0.038 to 0.105 for ξ. The CIs are 0.001–0.19 times as wide as the confirmatory C1-own's.
- **Rank agreement is weak and changes sign.** ξ's Spearman is −0.15 to 0.33: its CI is above 0
  in 5 networks (0.10–0.33) and below 0 in G2s2 (−0.15). s's Spearman is −0.08 to 0.18, with its
  CI above 0 in 3 (G1s2, G2, G3s2).
- **Companions (descriptive).**
  - With λ^own winsorised at T's frozen bounds and the predictions clipped there, every network
    is still in the refute band (statistic −0.044 to 0.056, CI upper ends at most 0.107).
  - The accel-held form repeats the primary. On s, n and ξ the held λ^own and g equal the
    on-manifold ones to round-off (at most 5·10⁻¹³ on the scored rows); they differ only from vx
    on.
- **Where the own runs are.**
  - The own interior covers almost every run state (median share 1.0); contact touches 0–4 kept
    states in the scored phases.
  - The own-run ξ rows sit closer to the apex than T's: key-weighted median s − s_apex is −82.0 to
    −80.0 m, against T's −89.7 m. Their median vx is 54.7–55.6 m/s, against T's 55.6 m/s.

**Expected directions against the results.**
- **E10.1** ("C1-own on own-run states is in the refute band in all 9, on at least 10 times
  C1-own's rows, with CIs at most half as wide"): **holds**, each part in 9 of 9. The rows are
  13.6–14.1 times (s) and 11.2–11.5 times (ξ) the confirmatory C1-own's, and the CI widths are
  0.001–0.19 times theirs.
- **E10.2** ("ξ's Spearman is positive (CI lower end > 0) in at least 6 of 9"): **does not
  hold**: 5 of 9 (G1s1, G1s2, G3, G3s1, G3s2). G2s2's is negative with its CI below 0, and G1,
  G2 and G2s1's CIs include 0.

**What CS-10 says (exploratory).**
- **The gap is in the critic's gradient itself** (the plan's reading of a refute here). The
  plan offered two other places for it:
  - T's restarts from the optimum's states. Here the critic is scored where its own policy
    drives, on 11–14 times the rows.
  - The optimum-versus-policy terms of design Section 3.3. Here the target is the policy's own
    costate.

  Neither rescues it. On approach and exit rows no network's R²_c exceeds 0.105, and no CI's
  upper end reaches 0.12.
- **Ranks.** On ξ the critic's gradient carries a weak positive rank signal in 5 networks (all
  three v3 seeds among them). Its sign is not stable across seeds.
- **Scope.** Approach and exit rows only (decision #8). The braking and corner rows, where most
  of the own runs' states lie, are untouched and stay open for BC-1. N's C1-own (CS-10 (iii))
  waits for CS-9.

## CS-11. The policy's own width price Π^{in,own} (exploratory)

**Outputs.** `outputs/costate/exploratory/cs-11/summary.json` (committed copy:
[`costate-followups/cs-11-summary.json`](costate-followups/cs-11-summary.json)) and `rows.npz`
(per key, local; it holds the inside-width fields ∂J/∂w_in(s_j) at both horizons).

**Rows.** D's 477 rows are all approach-phase interior rows of T, with no multi-basin row. The
scored rows are the kept keys' rows, 453–476 per network, under key weights renormalised on
them.

**What is compared.**
- Π^{in,own} is the derivative of the policy's discounted return with respect to a uniform inside
  widening. It is taken at H = W for the comparison with the optimum (W is 400 or 800 at D) and
  at H = 800 for the comparison with the critic.
- dV_dwin is the critic's stored derivative at D (σ_R-scaled), and Π^in is the optimum's price
  (winsorised at its frozen bounds as a target).
- Every quantity is in reward units per metre.

| Network | rows | critic–own R²_c [CI] | critic–optimum R²_c [CI] | own–optimum R²_c [CI] | Spearman: critic–own / critic–optimum / own–optimum | b_gl, Π^{in,own} on Π^in [CI] | medians: Π^{in,own} (H = W) / Π^in / dV_dwin |
|---|---|---|---|---|---|---|---|
| G1 | 468 | 0.047 [−0.045, 0.159] | 0.145 [−0.047, 0.214] | 0.268 [0.062, 0.399] | 0.48 / 0.42 / 0.51 | 0.61 [0.43, 0.82] | 0.143 / 0.230 / 0.306 |
| G1s1 | 469 | 0.112 [−0.0001, 0.215] | 0.217 [−0.012, 0.327] | 0.390 [0.195, 0.531] | 0.48 / 0.54 / 0.62 | 0.75 [0.53, 0.98] | 0.174 / 0.229 / 0.319 |
| G1s2 | 453 | 0.008 [−0.024, 0.032] | 0.239 [0.042, 0.322] | 0.079 [−0.200, 0.251] | 0.34 / 0.50 / 0.49 | 0.41 [0.06, 0.81] | 0.198 / 0.233 / 0.412 |
| G2 | 453 | 0.031 [−0.078, 0.114] | 0.068 [−0.123, 0.144] | 0.291 [0.109, 0.402] | 0.43 / 0.39 / 0.57 | 0.77 [0.49, 1.13] | 0.127 / 0.233 / 0.280 |
| G2s1 | 460 | 0.019 [−0.165, 0.054] | 0.280 [0.105, 0.378] | 0.076 [−0.130, 0.168] | 0.40 / 0.54 / 0.39 | 0.89 [0.38, 1.57] | 0.065 / 0.231 / 0.278 |
| G2s2 | 457 | −0.007 [−0.202, 0.021] | 0.129 [−0.093, 0.229] | 0.282 [0.043, 0.436] | 0.26 / 0.46 / 0.61 | 0.82 [0.45, 1.23] | 0.176 / 0.232 / 0.415 |
| G3 | 476 | −0.030 [−3.824, −0.024] | 0.263 [0.094, 0.338] | 0.275 [0.067, 0.414] | 0.45 / 0.55 / 0.56 | 2.55 [0.67, 6.30] | 0.092 / 0.228 / 0.277 |
| G3s1 | 469 | 0.037 [−0.212, 0.147] | 0.334 [0.132, 0.445] | 0.343 [0.162, 0.470] | 0.60 / 0.58 / 0.60 | 1.09 [0.62, 1.78] | 0.085 / 0.229 / 0.265 |
| G3s2 | 465 | 0.013 [−0.102, 0.102] | 0.350 [0.134, 0.463] | 0.049 [−0.199, 0.235] | 0.39 / 0.60 / 0.35 | 0.54 [0.16, 0.97] | 0.090 / 0.230 / 0.365 |

The R²_c is the calibrated R² under C1's rule, out of fold (`analysis.calibrated_agreement`;
no row went uncalibrated). b_gl is the pooled WLS slope of Π^{in,own} on Π^in. Medians are
key-weighted.

- **The critic agrees with its own policy's price least.**
  - Critic–own R²_c is −0.030 to 0.112, with every CI's lower end at or below 0 (upper ends at
    most 0.215).
  - On the same rows, critic–optimum is 0.068–0.350 (CI above 0 in 5 of 9) and own–optimum is
    0.049–0.390 (CI above 0 in 6 of 9).
  - All three Spearmans are moderate (0.26–0.62).
- **The policy's price has the optimum's shape at a smaller size.**
  - In the 6 networks where own–optimum is 0.27–0.39 with its CI above 0, the policy's price
    tracks the optimum's at least as well as the critic's derivative tracks either (in G3 and
    G3s1 by only 0.01).
  - The policy's median price is 28–85% of Π^in's in 9 of 9 (0.065–0.198 against 0.228–0.233).
    The critic's median dV_dwin (0.265–0.415) is larger than both.
- **The horizon hardly matters.** Π^{in,own}'s medians at H = W and H = 800 differ by at most
  0.006, and no kept row has Π^{in,own} = 0.

**Expected directions against the results.**
- **E11.1** ("The critic agrees better with its own policy's price than with the optimum's
  (calibrated R² critic–own above critic–optimum) in at least 6 of 9"): **does not hold**: 0 of 9.
  By Spearman, critic–own is higher in 3 (G1, G2, G3s1).
- **E11.2** ("The policy's price is smaller than the optimum's: the pooled slope of Π^{in,own} on
  Π^in is below 1 with its CI below 1, in at least 6 of 9"): **does not hold**: 4 of 9 (G1,
  G1s1, G1s2, G3s2).
  - The point slope is below 1 in 7. G3 (2.55 [0.67, 6.30]) and G3s1 (1.09) are above.
  - The medians are smaller in 9 of 9.
- **E11.3** ("The own-policy–optimum agreement is low (calibrated R² < 0.2)"): **does not hold**:
  3 of 9 (G1s2 0.079, G2s1 0.076, G3s2 0.049). In the other 6 it is 0.27–0.39 with its CI above
  0.

**What CS-11 says (exploratory).**
- **The plan's first reading is not supported at the level of the policy.** That reading was
  "the networks do not value the inside wall the way the optimum does", which needed E11.1 with
  E11.3, and neither holds. In 6 of 9 networks the policy's own price of the inside wall agrees
  with the optimum's in rank and, moderately, after calibration (R²_c 0.27–0.39). It is smaller,
  as a policy that keeps a buffer would have it.
- **The plan's second reading applies.** In all 9 the critic disagrees with its own policy's
  price (R²_c at most 0.11), more than with the optimum's. So the critic misprices the wall even
  relative to what its own driving earns.
  - In 6 of 9 the mismatch W1 found lies in the critic's derivative, not in the policy's
    valuation.
  - In G1s2, G2s1 and G3s2 the policy's price also agrees poorly with the optimum's (R²_c below
    0.08).
- **Caveats.**
  - D only (477 reset starts on the approach).
  - One widening direction, the uniform inside widening.
  - Π^{in,own} reaches the wall only through the policy's observation: the env's termination
    test is not differentiated, and kept runs never terminate.
  - The critic's dV_dwin is the derivative of a learned value at γ = 0.995, compared with the
    800-step return's.
  - N's CS-11 waits for CS-9.

## CS-10 and CS-11: where they leave the refute (exploratory)

- **On the policies' own states, the critics' gradients still do not express their policies'
  costate** on approach and exit rows (CS-10). This holds with 11–14 times the rows and much
  narrower CIs than the confirmatory C1-own.
- **In 6 of 9 the policies price the inside wall roughly as the optimum does**, only smaller
  (Spearman 0.51–0.62; calibrated R² 0.27–0.39, CI above 0). **The critics price it unlike their
  own policies in all 9** (CS-11).
- There the networks' driving carries a smaller copy of the wall's price, and the critic's input
  gradient is what fails to express it.
- None of this changes a decision of `costate-prereg-v1`.

## CS-7 and CS-8: the runs (exploratory)

- **Plan and code.** Both ran at `2056a3d` (code and tests), under the plan with amendment 7
  (`3d41566`: their operational details, committed before any of their numbers) and amendment 8
  (`79c19d6`: two corrections to amendment 7's text, also before any number).
- **CS-7.** `scripts/costate_exploratory.py cs-7`: 72 units ((network, component) × 4 chunks)
  on 4 workers × 2 XLA threads, niced, in 94 min. Worker time was 1,112–1,399 s per (network,
  component). The plan estimated about 30 min pooled at 5 workers. Each of the 106 probe fits
  per (network, component) takes 9–12 s, as measured on synthetic data before the run
  (amendment 7).
- **CS-8.** `cs-8`, with the same pool, in 30 min (its three pools are described under CS-8).
- **Development runs.** Before the canonical runs, partial runs on G1 were written outside
  `outputs/costate` to check the wiring: CS-8 for G1's scaling and (c), and CS-7 for G1's s
  component. Only their structure, wiring gaps and timings were read, plus one outcome of
  (c)'s choice: candidate 1 fails and candidate 2 passes. That choice reads targets only, and
  the canonical run repeats it.
- **Stamps.** Both outputs record an empty `changed_vs_head`, no partial flag, the plan's sha256
  with 8 amendments, and the sha256 of every input:
  - CS-7, 20 files: the world and its summary, the 9 networks' activations, and their checkpoints
    through the record's pins;
  - CS-8, 21 files: the same, with `gates/c2_null.json`.

  They were re-verified before this note:
  - each stamp's script sha256 is the script's blob at `2056a3d`, and its plan sha256 is the plan
    at `79c19d6`;
  - the `preds.npz` sha256s and the summaries' content sha256s were recomputed and match;
  - the script's `verify` reports 0 problems.
- **Wiring checks (all passed).**
  - The frozen λ_res's interior residual shares equal the record's exactly (gap 0 in all four
    components).
  - In every unit, the pool actor of seed 0 rebuilds its stored h3 exactly.
  - CS-7's random-init critic of seed 0 matches `Critic.apply` to at most 1.8·10⁻¹⁵ (tolerance
    10⁻¹⁰), and every random critic has 5 heads.
  - CS-8's cross-fit scheme reproduces `states.cross_fit` exactly.
  - f̂_D's geometry-6 shares reproduce design 17.6: 11.78% and 9.00% of λ's sum of squares, 70.45%
    and 64.66% of λ_res's.
  - Every probe's targets are, bit for bit, λ and λ_res's own-fold residual (CS-7), or σ^ped and
    σ_res's (CS-8).
- **No calibrated R²** is reported by either item, so CS-1's rule for later items does not apply.

## CS-7. Critic-layer and head-unit probes (exploratory)

**Run** as above. Outputs: `outputs/costate/exploratory/cs-7/summary.json` (committed copy:
[`costate-followups/cs-7-summary.json`](costate-followups/cs-7-summary.json)) and `preds.npz`
(local). **Rows.** C1's rows (λ_s 1,643, λ_ξ 1,581) under C1's weights. **Targets.** λ_k
winsorised, and the frozen λ_res,k (f̂_D, k = 8) by fold. Both of ξ's are scored by R²_wd.

**Decodability** (ranges over the 9 networks; R² of the probe on z alone and on z ⊕ each layer):

| Target | R²(z) | z ⊕ critic h3 | z ⊕ head units | z ⊕ actor h3 | Random critics' median gain (h3) |
|---|---|---|---|---|---|
| λ_s | 0.354–0.355 | 0.51–0.61 | −0.40 to 0.46 | 0.54–0.61 | 0.22–0.24 |
| λ_ξ | 0.663–0.673 | 0.80–0.89 | 0.06–0.82 | 0.83–0.88 | 0.15–0.17 |
| λ_res,s | −1.17 | −0.67 to −0.11 | −0.91 to −0.01 | −0.79 to −0.06 | 0.91–1.00 |
| λ_res,ξ | 0.017–0.024 | −0.09 to 0.02 | −0.64 to 0.01 | −0.37 to −0.14 | −0.20 to −0.16 |

**Gains beyond the controls.** Each cell gives the statistic's range over the 9 networks: the
trained layer's gain over z minus the median of its 20 controls' gains (random-init critics'
same layer, or the pool actors for the actor's h3). In brackets: the networks whose two-level CI
lies above 0 / below 0. *Note added at the close of the lane:* as in CS-5, the random-init
critics' h3 equals the pool actors' h3 (same trunk, same seeds; their control gains are
bit-identical), so the critic-h3 and actor-h3 columns are contrasts against one shared
random-feature control.

| Target | Critic h1 | Critic h2 | Critic h3 | Head units | Actor h3 |
|---|---|---|---|---|---|
| λ_s | −0.082 to 0.041 (0 / 1) | −0.050 to 0.049 (1 / 1) | −0.085 to 0.010 (0 / 3) | −0.983 to −0.094 (0 / 8) | −0.043 to 0.012 (0 / 1) |
| λ_ξ | −0.054 to 0.023 (0 / 1) | −0.038 to 0.045 (0 / 0) | −0.023 to 0.066 (2 / 0) | −0.787 to −0.028 (0 / 4) | 0.000 to 0.059 (2 / 0) |
| λ_res,s | −0.105 to 0.033 (0 / 0) | −0.109 to 0.051 (0 / 0) | −0.465 to 0.122 (0 / 0) | −0.785 to 0.129 (0 / 4) | −0.588 to 0.159 (0 / 1) |
| λ_res,ξ | 0.018 to 0.150 (2 / 0) | −0.079 to 0.179 (4 / 0) | 0.063 to 0.196 (5 / 0) | −0.488 to 0.166 (3 / 0) | −0.191 to 0.024 (0 / 1) |

On ξ, per network. Bold marks a statistic above 0.05 with its two-level CI above 0. The last
column is R²(z ⊕ critic h3) on λ_res,ξ.

| Network | λ_ξ: critic h3 | λ_ξ: actor h3 | λ_res,ξ: h1 | h2 | h3 | heads | actor h3 | R²(z ⊕ h3), λ_res,ξ |
|---|---|---|---|---|---|---|---|---|
| G1 | **0.056 [0.003, 0.116]** | 0.056 [0.009, 0.113] | 0.057 | 0.032 | **0.156** | 0.066 | −0.020 | 0.003 |
| G1s1 | 0.022 | 0.025 | 0.045 | 0.021 | 0.089 | −0.488 | −0.014 | −0.046 |
| G1s2 | **0.066 [0.004, 0.165]** | 0.059 [0.020, 0.110] | 0.115 | **0.121** | 0.128 | 0.137 | −0.085 | −0.011 |
| G2 | 0.005 | 0.017 | 0.018 | 0.030 | **0.139** | 0.077 | −0.012 | −0.010 |
| G2s1 | 0.020 | 0.003 | 0.063 | **0.169** | **0.162** | **0.166** | 0.024 | −0.002 |
| G2s2 | −0.023 | 0.007 | 0.047 | −0.011 | 0.065 | −0.098 | −0.091 | −0.075 |
| G3 | 0.021 | 0.000 | **0.133** | −0.079 | 0.063 | −0.286 | −0.018 | −0.085 |
| G3s1 | 0.021 | 0.001 | 0.070 | **0.169** | **0.183** | **0.142** | −0.017 | 0.005 |
| G3s2 | 0.023 | 0.030 | **0.150** | **0.179** | **0.196** | **0.138** | −0.191 | 0.018 |

- **λ_s: no critic layer holds it beyond random critics' features.**
  - The critic's h3 gains −0.085 to 0.010, with its CI below 0 in 3.
  - The head units do worse than their random counterparts in all 9 (−0.98 to −0.09, CI below 0
    in 8). With 320 units they over-fit: R²(z ⊕ head units) reaches −0.40 in G3.
  - The actor's h3 is no different (−0.043 to 0.012).
- **λ_ξ: the critic's h3 gains a little in G1 and G1s2, and so does the actor's.**
  - The critic's h3 gains 0.056 and 0.066 beyond its random critics, with CIs above 0, first of
    21 in both. In the other 7 it gains −0.023 to 0.023.
  - The actor's h3 gains as much in the same two networks (0.056 and 0.059, CIs above 0). The
    gain is not specific to the critic.
- **λ_res,s is not decodable on these rows.** R²(z) is −1.17, and every layer, trained or random,
  leaves the probe below 0 (z ⊕ critic h3: −0.67 to −0.11). The statistic is noise: no CI lies
  above 0. This is geometry 6's residualisation error (CS-8 (c)).
- **λ_res,ξ is not decodable either, and that is where the large gains come from.**
  - R²(z) is 0.02. No trained layer lifts the probe above 0.02: R²(z ⊕ critic h3) is −0.09 to
    0.02.
  - The random critics' layers make the probe worse (their median gains are −0.20 to −0.14). The
    trained critic's make it worse by much less (h3: −0.10 to 0.00).
  - That gap is the statistic, 0.063–0.196 for h3, above 0.05 with its CI above 0 in 5. It
    measures how much more random critics' features over-fit, not information in the trained
    layer.
  - The actor's h3 over-fits like its pool (−0.19 to 0.02).
- **Means over s and ξ** (descriptive): on λ, the critic's h3 gains −0.032 to 0.017 with every CI
  covering 0, and the head units −0.51 to −0.06 with every CI below 0.

**Expected directions against the results.**
- **E7.1** ("No critic layer gains more than 0.05 beyond z on λ or λ_res with its CI above 0"):
  **does not hold as computed.** 16 cases are named, the same under the geometry-only CI.
  - 2 are on λ_ξ: G1's and G1s2's h3 (0.056 and 0.066).
  - 14 are on λ_res,ξ, where no layer predicts the target (R²(z ⊕ h) at most 0.02). There the
    statistic exceeds 0.05 because the controls over-fit.
- **E7.2** ("The critic's h3 gains at least as much as the actor's h3"): **does not hold.** It
  holds in 1 of 9 on λ_s, 5 on λ_ξ, 4 on λ_res,s and 9 on λ_res,ξ. The 9 on λ_res,ξ come only from
  the actor's h3 over-fitting as much as its pool.
- **E7.3** ("The head units gain no more than the critic's h3"): **holds**, in 9, 9, 8 and 7 of
  9.

**What CS-7 says (exploratory).**
- **The plan's reading of a gain above 0.05** ("λ is linearly present in the critic even though
  ∇V does not match it") **does not apply to the cases found.**
  - On λ_res,ξ the named gains carry no prediction of the target.
  - On λ_ξ the two networks whose critic h3 passes are the two whose actor h3 passes too.
  - λ itself is not decoy-controlled. So those gains may be position-and-speed information that
    random features of z do not express linearly.
- **No critic layer holds λ_s, or λ_res on either component,** beyond what random critics'
  features of z give.
- **So the gains near 0 extend C2's result to the critic** (the plan's reading). As far as
  linear probes on these rows can tell, the costate is no more in the critic's layers than in
  its input gradient. This differs from CS-5, where the critic's h3 held a little Π^in beyond z.
- **A limit.** On C1's rows the frozen λ_res is not decodable by any probe (R²(z) −1.17 on s,
  0.02 on ξ), so CS-7's λ_res half has little power. CS-8 (c)'s decoy 2 would give a better
  λ_res for it. Rerunning on it would need an amendment, and is not done here.
- None of this changes a decision of `costate-prereg-v1`.

## CS-8. C2-sw's random-only null, C2-sw residualised, and C2 with a decoy that fits geometry 6 (exploratory)

**Run.** 1,779 s in three pools on 4 workers:
- G1's scaling first: 4 units, 69 s.
- R6's rule: the other 8 scalings were projected at 8 × 69 s = 553 s, well under 3 h, so the
  null also ran per network (32 units, 519 s).
- (c): 18 units, 1,104 s.

The plan estimated 20 min for G1's scaling and 2.7 h per network. Fitting each scaling's base and
pool probes once (amendment 7) brought (a) and (b) together under 10 min. Outputs:
`outputs/costate/exploratory/cs-8/summary.json` (committed copy:
[`costate-followups/cs-8-summary.json`](costate-followups/cs-8-summary.json)) and `preds.npz`
(local).

**Rows.**
- (a) and (b) use T's pedal-switching rows in the approach and exit phases: 976 + 1,531 = 2,507,
  under the key weights renormalised on them. The confirmatory C2-sw used all 2,911, 404 of them
  braking or corner rows. So these numbers are not on its rows.
- (c) fits its probes on T's interior approach and exit rows (1,653) and scores each component
  on C2's decision rows (s 1,643, ξ 1,581). The confirmatory C2 fitted on all 3,951 interior
  rows, so (c) carries its own reference: f̂_D on the same rows (amendment 7 item 3).

### (a) C2-sw's random-only null

The trained networks on the 2,507 rows. Δsw is the trained h3's gain over z minus the pool's
median gain, with the two-level CI. "Hosts ≥" is the share of the 20 null hosts whose statistic
is at or above the network's, under G1's scaling and under the network's own.

| Network | Δsw [two-level CI] | Trained gain | Pool median | Rank of 21 | Hosts ≥, G1's | Hosts ≥, own | Confirmatory Δsw (2,911 rows), rank |
|---|---|---|---|---|---|---|---|
| G1 | 0.049 [0.022, 0.116] | 0.186 | 0.137 | 1 | 0.00 | (G1's) | 0.028, 1 |
| G1s1 | 0.043 [0.023, 0.103] | 0.177 | 0.133 | 1 | 0.00 | 0.00 | 0.030, 1 |
| G1s2 | −0.113 [−0.469, 0.031] | 0.024 | 0.137 | 19 | 0.85 | 0.90 | 0.025, 1 |
| G2 | 0.027 [0.008, 0.091] | 0.178 | 0.151 | 1 | 0.05 | 0.00 | 0.021, 1 |
| G2s1 | 0.028 [0.002, 0.080] | 0.176 | 0.148 | 1 | 0.05 | 0.05 | 0.009, 3 |
| G2s2 | 0.040 [0.016, 0.087] | 0.193 | 0.153 | 1 | 0.00 | 0.00 | 0.029, 1 |
| G3 | 0.021 [0.002, 0.057] | 0.186 | 0.165 | 3 | 0.10 | 0.15 | 0.006, 3 |
| G3s1 | 0.047 [0.018, 0.114] | 0.203 | 0.156 | 1 | 0.00 | 0.00 | 0.021, 1 |
| G3s2 | 0.039 [0.013, 0.112] | 0.193 | 0.154 | 1 | 0.00 | 0.00 | 0.028, 1 |

R²(z) on σ^ped is 0.738 in all 9.

- **The null under G1's scaling** (hosts 20–39 against the pool 0–19).
  - Its bulk sits at 0: median 0.0005, 95th percentile 0.025, largest 0.030.
  - Its mean is −0.055 and its SD 0.151, because four hosts lie far below: −0.645, −0.191,
    −0.121 and −0.094.
  - In each of the four, the most influential geometry is 17 (R 14.6 m, θ 45°). Without it, each
    lies within 0.01 of 0.
  - Ruled decisions: 11 refute and 9 inconclusive; none confirms. Two CIs lie wholly above 0.
- **Under each network's own scaling** the nulls are alike: means −0.014 to −0.021, SDs 0.042 to
  0.064, medians −0.002 to 0.010, largest 0.026 to 0.034.
- **The trained h3 on approach and exit rows.**
  - In 8 of 9 its statistic is 0.021–0.049, with the two-level CI above 0. On all 2,911 rows it
    was 0.006–0.030.
  - The CIs' upper ends are 0.057–0.116, so all 8 carry the inconclusive band word. On 2,911 rows
    all 9 refuted.
  - It ranks first of 21 in 7: the confirmatory 7, with G2s1 in place of G1s2.
  - Five networks lie above every G1-scaling host: G1, G1s1, G2s2, G3s1 and G3s2. G2 and G2s1
    lie above 19 of the 20, G3 above 18.
  - Against their own scaling's hosts, 6 of 9 lie above every one: those five and G2.
  - In G1, 4 of the 48 leave-one-geometry-out sets reach the confirm band word. The full set's
    point is 0.049, just under 0.05.
- **G1s2 is the exception.** Its trained h3 gains only 0.024 over z, against the pool's median
  0.137. Its most influential geometry is also 17: without it, its statistic is 0.021.

**E8a.1** ("the null's points centre near or below 0, with an SD of about 0.01–0.02 … The 7
networks that rank first of 21 lie above every host"): **does not hold.**
- The mean is below 0 (−0.055).
- The SD is 0.151, outside [0.005, 0.025): one geometry's outliers carry it.
- Of the 7 networks, 5 lie above every host. G2 (0.027) lies below the largest host (0.030), and
  G1s2 (−0.113) below 17 of the 20.

### (b) C2-sw on σ^ped residualised on position and speed

| Network | Δ on σ_res [two-level CI] | Rank of 21 | Trained gain | Pool median | Δsw on σ^ped, (a) |
|---|---|---|---|---|---|
| G1 | −0.002 [−0.153, 0.112] | 11 | −0.126 | −0.125 | 0.049 |
| G1s1 | 0.050 [−0.071, 0.183] | 1 | −0.083 | −0.132 | 0.043 |
| G1s2 | 0.005 [−0.162, 0.220] | 8 | −0.125 | −0.130 | −0.113 |
| G2 | −0.031 [−0.191, 0.087] | 18 | −0.190 | −0.159 | 0.027 |
| G2s1 | 0.004 [−0.100, 0.091] | 11 | −0.152 | −0.156 | 0.028 |
| G2s2 | 0.085 [−0.020, 0.248] | 1 | −0.075 | −0.160 | 0.040 |
| G3 | 0.079 [−0.026, 0.224] | 1 | −0.097 | −0.176 | 0.021 |
| G3s1 | 0.011 [−0.091, 0.059] | 8 | −0.174 | −0.185 | 0.047 |
| G3s2 | 0.117 [−0.001, 0.281] | 1 | −0.071 | −0.188 | 0.039 |

- **f̂_D's decoys carry most of σ^ped.** On these rows σ_res keeps 11.6% of σ^ped's variance.
- **What is left is not linearly decodable.**
  - R²(z) on σ_res is −0.077 in all 9.
  - Every hidden layer makes the probe worse, trained or random: the trained gains are −0.071 to
    −0.190, and the pool's medians −0.125 to −0.188.
  - The statistic (−0.031 to 0.117) therefore measures how much less a trained h3 over-fits than
    random ones, not a signal any layer finds.
- **Every two-level CI covers 0.** G3s2's lower end is −0.001; its geometry-only CI is
  [0.001, 0.277]. Four networks rank first: G1s1, G2s2, G3 and G3s2.

**E8b.1** ("the gain shrinks to at most half of its unresidualised value in at least 6 of 9 …
the trained h3 ranks first of 21 in at most 4 of 9"): **does not hold.**
- The first part holds in 4 of 8: G1, G2, G2s1 and G3s1. G1s1, G2s2, G3 and G3s2 are larger on
  σ_res, inside CIs that cover 0. G1s2's statistic on σ^ped is negative, so it is not counted.
- The second part holds: first of 21 in 4 of 9.
- **The rank-1 pattern does not survive.** No network ranks first with its CI above 0, against
  the plan's 7 of 9.

### (c) C2 with a decoy that fits geometry 6

**The choice** (targets only, before any probe was fitted; amendment 7 item 3). The thresholds
are twice geometry 6's share of λ's sum of squares: 23.6% for s and 18.0% for ξ.

| Decoy | Geometry 6's share of λ_res's sum of squares, s | ξ | Interior residual share, s | ξ | Kept |
|---|---|---|---|---|---|
| f̂_D (frozen) | 70.4% | 64.7% | 0.224 | 0.248 | – |
| 1. Same θ and direction | 70.4% | 64.1% | 0.224 | 0.248 | no |
| 2. × 3 on log R | 3.0% | 5.5% | 0.204 | 0.154 | **yes** |

- Candidate 3 was not tried.
- In both candidates the search fell back to the turn direction only for the 72 θ 60° rows,
  wherever fold 2 was predicted (432 predictions in the 36 fits).
- Under candidate 2 the power floor still passes s and ξ only (n 0.060, vx 0.063).
- **Why candidate 1 does not help.** Geometry 6's held-out neighbours under f̂_D were already of
  its θ and direction: 71% from the 22.9 m, 90° geometry, 26% from 8.4 m. Weighting log R by 3
  draws them from the nearest radii instead: 81% from 8.4 m and 19% from 10.5 m. (A count of
  D₁'s neighbours, made for this note; it reads no target.)

**C2 under the chosen decoy**, against f̂_D on the same rows:

| Network | Δ2 [two-level CI], decoy 2 | Rank of 21 | Gain s, ξ | Geometry 6's share of the base's error, s / ξ | Δ2 [two-level CI], f̂_D, same rows | Its geometry 6 share, s / ξ | CI width ratio | Band word |
|---|---|---|---|---|---|---|---|---|
| G1 | −0.005 [−0.220, 0.062] | 14 | 0.013, −0.024 | 0.059 / 0.071 | −0.058 [−0.619, 0.028] | 0.350 / 0.505 | 0.44 | inconclusive |
| G1s1 | −0.027 [−0.180, 0.029] | 20 | −0.007, −0.047 | 0.057 / 0.068 | −0.192 [−0.690, 0.020] | 0.376 / 0.531 | 0.29 | refute |
| G1s2 | 0.023 [−0.069, 0.133] | 4 | 0.063, −0.018 | 0.057 / 0.076 | 0.066 [−0.364, 1.328] | 0.367 / 0.541 | 0.12 | inconclusive |
| G2 | −0.168 [−0.756, −0.018] | 21 | −0.014, −0.322 | 0.057 / 0.075 | 0.250 [−0.069, 1.773] | 0.375 / 0.533 | 0.40 | refute |
| G2s1 | −0.045 [−0.206, 0.013] | 21 | −0.013, −0.077 | 0.058 / 0.072 | 0.028 [−0.216, 0.236] | 0.354 / 0.525 | 0.48 | refute |
| G2s2 | −0.051 [−0.351, 0.034] | 21 | −0.021, −0.082 | 0.058 / 0.070 | 0.185 [−0.111, 1.516] | 0.366 / 0.536 | 0.24 | inconclusive (full set: refute) |
| G3 | 0.005 [−0.168, 0.075] | 10 | 0.004, 0.006 | 0.057 / 0.075 | −0.114 [−0.714, 0.015] | 0.358 / 0.523 | 0.33 | inconclusive |
| G3s1 | −0.010 [−0.203, 0.060] | 15 | 0.044, −0.063 | 0.056 / 0.072 | 0.043 [−0.164, 0.385] | 0.370 / 0.527 | 0.48 | inconclusive |
| G3s2 | −0.074 [−0.455, 0.036] | 21 | 0.018, −0.165 | 0.055 / 0.073 | −0.126 [−0.637, 0.136] | 0.366 / 0.513 | 0.63 | inconclusive (full set: refute) |

"CI width ratio" is the decoy-2 CI's width over f̂_D's on the same rows. "(full set: refute)"
marks a refute that the single-geometry rule withdrew: in G2s2 4 of the 48 sets, and in G3s2 3,
do not refute.

- **With f̂_D on approach and exit rows, C2 is unstable.**
  - Geometry 6 still holds 0.35–0.38 of the base's squared error on s and 0.51–0.54 on ξ.
  - The statistic swings from −0.192 to 0.250, with two-level CIs 0.45–1.84 wide.
- **Decoy 2 removes geometry 6's weight.**
  - It holds 0.055–0.059 of the base's error on s and 0.068–0.076 on ξ, about its share of the
    weight.
  - Geometry 6 is the most influential geometry in no network (31 in 4, 8 in 2).
- **The CIs are 1.6 to 8 times narrower** (0.20–0.74 wide).
- **The trained h3 sits below the pool's median in 7 of 9.** It ranks 14th to 21st in 7, and last
  of 21 in 4. Only G1s2 (4th) and G3 (10th) rank above the median.
- **The base probe no longer predicts λ_res,s.** Its R² is −0.26 on s and 0.11–0.13 on ξ. What
  is left of λ_s after decoy 2 is not linear in z and m on these rows.
- **Against the confirmatory C2** (all interior rows, f̂_D): its CIs were 0.24–0.50 wide. Decoy
  2's are narrower by a third in only 2 networks (G1s1 and G1s2), and wider in G1, G2 and G3s2.

**E8c.1** ("Geometry 6's share of the base's squared error falls below 0.3. C2's two-level CIs
narrow by at least a third. No point reaches 0.05"): **holds**, in 9 of 9 on each part.
- Shares 0.055–0.076.
- Width ratios 0.12–0.63 against f̂_D on the same rows.
- Largest point 0.023.

### What CS-8 says (exploratory)

- **C2-sw's pool rank carries over to approach and exit rows, larger.** The trained h3's extra
  gain over random h3s is 0.021–0.049 in 8 of 9, with CIs above 0. It ranks first of 21 in 7,
  and 6 networks lie above every host of their own null.
- **C2-sw's own null has a heavy lower tail, from one geometry.** Its median and 95th percentile
  (0.000 and 0.025) describe it; its mean and SD (−0.055, 0.151) do not. Geometry 17, a θ 45°
  corner, also drags G1s2's statistic below its null.
- **The rank-1 gain does not survive residualisation on position and speed.** After f̂_D's
  decoys, nothing linear in z is left for any layer to find. So the gain in (a) is consistent
  with the position-and-speed information that design 15.10 named. It gives no evidence of
  switching-function information beyond it.
  - (b) is weak: σ_res keeps 12% of σ^ped's variance on 2,507 rows. It cannot tell "nothing
    beyond the decoys" from "too little to find".
- **A decoy that fits geometry 6 makes C2 more precise and leaves it without a confirm.**
  - On the same rows its CIs are 1.6–8 times narrower than f̂_D's.
  - The trained h3 sits below the pool's median in 7 of 9.
  - The band words are 3 refute and 6 inconclusive. In 2 of the 6 the full set refutes, but
    removing one of 3–4 geometries does not, so the single-geometry rule withdrew the refute.
  - Geometry 6's weight was a residualisation artefact, as design 17.6 read it. With it removed,
    C2 shows no sign of costate information in h3 beyond z, m and the decoys.
- None of this changes a decision of `costate-prereg-v1`.

## CS-7 and CS-8: where they leave the refute (exploratory)

- **No layer of the critic or actor shows the costate beyond z, the actor's output and
  position-and-speed decoys.**
  - CS-7: the critic's layers and heads hold no more of λ_s, λ_res,s or λ_res,ξ than random
    critics' features. Their small gains on λ_ξ, in two networks, match the actors'.
  - CS-8 (c): with a decoy that fits geometry 6, the actor's h3 sits below its random pool's
    median in 7 of 9, with CIs 1.6–8 times narrower than f̂_D's on the same rows.
- **C2-sw's rank-1 pattern is real but not shown to be the switching function.** On approach and
  exit rows the trained h3 adds more than every random actor in 7 of 9, and more than every null
  host in 5–6. Once σ^ped is residualised on position and speed, no layer finds anything (CS-8
  (b)). Design Section 9's switching-function reading therefore still needs a confirm it does not
  have.
- **Two measurement lessons for later items.**
  - f̂_D's residuals are unstable on approach and exit rows (geometry 6). Decoy 2 (log R × 3,
    same θ and direction) fixes it on targets alone.
  - C2-sw's null has a heavy lower tail from one geometry (17, θ 45°), so its median and upper
    percentiles are the robust comparison.
  - CS-9's and CS-12's C2 inherit both points, and the lead may want decoy 2 co-reported there
    (an amendment).
- None of this changes a decision of `costate-prereg-v1`.

## CS-9 with CS-10 (iii), N's CS-11 and N's CS-13: the runs (exploratory)

- **Plan and code.** `scripts/costate_exploratory.py cs-9` at `6563b28` (code and tests), under
  the plan with amendment 9 (`4541068`: the operational details of CS-9, CS-10 (iii), N's CS-11
  and N's CS-13, committed before any of their numbers). It runs in four stages, each stamped
  on its own, on 4 workers × 2 XLA threads, niced:
  - **build** (targets only): 103 units (a stimulus and a block of at most 128 of its keys), 8
    min;
  - **k2b** (targets only): 27 units (26 keys and the wiring key), 14 min;
  - **networks**: 48 units (an agent and a quarter of its keys), 77 min (worker time 25–41 min
    per generalist, 4–12 min per specialist);
  - **analyse**: 12 units (one per agent), 15 min.

  The plan estimated 45–60 min pooled at 5 workers, plus the build, plus about 53 min for K2b
  on N.
- **Development runs.** Partial runs were written outside `outputs/costate` before the canonical
  ones:
  - the build on 6 stimulus blocks (350 keys), K2b on 3 keys of that partial world, and the
    networks stage on G1's 19 rows in it;
  - on the canonical N, the networks stage on a quarter of G1's rows (175), then the analyse
    stage on those rows.

  Their structure, wiring gaps and timings were read. Two outcomes were also seen:
  - the 3 development K2b keys' rows. One of them, a G2s1 key outside the canonical picks,
    moved λ_vx by 0.30 SD_T and λ_s by 0.23 under the slow second corner;
  - from the partial analysis: G1's C1-own status ('secondary'), and that CS-13 finds no pair.
    A count on states alone then showed that no N state of any agent has a matched T state
    (item 7 below).
- **The first networks run stopped.** It computed all 48 units (about 72 min), then the parent's
  re-check of the width checks stopped on S2 and wrote nothing. Every S2 run terminates, so no S2
  unit kept a key and none ran a JVP check, and the re-check required one. That is a code fault,
  not a failed check: amendment 9 item 4 asks for the check on each unit's first kept key.
  `950918c` lets an agent with no kept key through and records why, with a test. The rerun used
  that commit. The only number seen from the stopped run is S2's (no kept key).
- **Stamps.** Every output records an empty `changed_vs_head`, no partial flag, and the plan's
  sha256 with 9 amendments. The build and K2b ran at `6563b28`; the networks stage and the
  analyses at `950918c`. The inputs, by sha256:
  - the build, 111: the world and its summary, all 34 networks' critic and K3 files, the 9
    decision networks' own, K3b and activations files, `gates/k0.json` and `gates/k2b.json`, and
    the 12 agents' N key sets (content hashes);
  - K2b, 112: the same with N's world;
  - the networks stage, 112: the checked groups, N's world and the 12 checkpoints through the
    record's pins;
  - the analyses, 115: those, with the build's, K2b's and the networks stage's outputs.

  They were re-verified before this note:
  - each stamp's script sha256 is the script's blob at its HEAD, and its plan sha256 the plan at
    that HEAD (unchanged since `4541068`);
  - the npz sha256s and the summaries' content sha256s were recomputed and match;
  - the script's `verify` reports 0 problems.
- **Wiring checks (all passed).**
  - **Build.** The build's code at T's steps reproduces the bound world's rows (every column),
    Inputs and ObsJac on T's first two keys exactly (gap 0; tolerance 10⁻¹²). The jac and cheap
    routes' λ_1 agree to at most 1.0·10⁻¹² (median 7·10⁻¹⁷), descriptively.
  - **K2b.** The generalised K2b code at T's rows reproduces `gates/k2b.json`'s first key
    (reach, s2 and both severities' rows) exactly (gap 0; tolerance 10⁻⁶ SD_T).
  - **Networks**, per agent on T's first key, from the bound arrays: the critic's pullback, the
    random critic of seed 0 and dV_dwin reproduce the stored ones exactly (gap 0) for all 12
    agents. λ^own at T's first interior restart reproduces `post/own` exactly for the 9 (gap 0).
  - **The width loop.** Its J matches `own_return`'s to at most 3.0·10⁻¹¹ relative (G1s1;
    tolerance 10⁻⁶). The JVP matches the VJP's contraction to at most 5.5·10⁻⁸ (G2s1; tolerance
    10⁻⁶), on 40 keys: one per unit that kept a key (S2's 4 units kept none, S1's and S3's 4 of
    8). Descriptive: on those keys, Richardson differences of J_env at h = 10⁻³ m differ from the
    AD value by a median of 4.7·10⁻⁶ relative, at most 9.7·10⁻³; every perturbed run stays
    alive.
  - **Analyses.**
    - The frozen λ_res's interior residual shares equal the record's exactly.
    - The external cross-fit on T's own rows reproduces T's by-fold residuals exactly.
    - On T's first 64 rows, the trained actor's h3 and the pool actor of seed 0 reproduce the
      bound `actor_h3` and `random0_h3` exactly (the 9).
    - CS-13's union fit reproduces the fit on T's rows alone to at most 8.3·10⁻¹² of T's
      predictions (tolerance 10⁻⁹).
- **No calibration but the frozen one** (CS-1's rule for later items did not fire).

### N

- **The keys.** The index's lists equal the record's key sets, and the key files its content
  hashes, for all 12 agents. The 6 K0 keys are dropped (2 of G1's, 2 of G2's, 1 of G3's, 1 of
  G1s1's). No key is multi-basin and no re-polish fails. That leaves 8,130 states: G1 699, G1s1
  709, G1s2 705, G2 720, G2s1 742, G2s2 723, G3 750, G3s1 760, G3s2 759, S1 766, S2 404, S3
  393.
- **Where they are.** Every one of the 8,130 states is an approach-phase interior state, with no
  contact. Their x_1 lie 213–173 m before the apex (5th–95th percentiles; median 183 m), at
  67–87 m/s (median 85 m/s). 93–100% are pedal-switching. The generalists' states cover 26–29
  of T's geometries, and the specialists' 30. All are θ 90° or 180° geometries.
- **So N scores λ_s and λ_ξ on approach rows only.** It has no exit row, so s is scored only on
  the approach phase, where it is also scored on T. Every N state is a decision row of both s
  and ξ wherever s is observable.

### K2b on N (targets only)

- **The keys.** 26 picks over the 6 strata N's geometries form (R tercile × θ class 90° or 180°;
  4–5 per stratum), from 10 agents (G3s2 4; G1, G1s1, G1s2, G2, G2s2 and S1 3 each; G3 2; S2
  and S3 1). Every re-solve succeeds at both severities.
- **The second corner sits much nearer than on T.** On T it starts 20 m beyond the furthest
  preview point of all of a key's T states (beyond s_c1 + 50 m). On N it starts 20 m beyond
  x_1's own preview, 1,149–1,414 m along the strip, well inside the window problem's
  400-step plan (which ends at 2,080–2,502 m).
- **|Δλ_k| / SD_T at x_1, approach phase (26 rows):**

  | Severity | s | n | ξ | vx |
  |---|---|---|---|---|
  | fast (R 150 m, θ 60°): p90 | 0.0011 | 0.0001 | 0.0001 | 0.0013 |
  | slow (R 30 m, θ 90°): p90 | 0.070 | 0.006 | 0.005 | 0.091 |
  | slow: largest | 0.085 | 0.008 | 0.013 | 0.099 |

- **The rule excludes nothing on N.** Every p90 is below 0.1, so N's own exclusions are empty.
  N's decision exclusions are T's frozen ones (approach: n and vx). vx's slow p90 (0.091) is
  close to the line. One of the 3 development keys, not among the 26, moved λ_vx by 0.30.
- **E9.4** ("K2b on N excludes n and vx on approach, as on T"): **does not hold.** N's own rule
  excludes no component on approach. With T's exclusions in the union, the decision rows are the
  same components as on T.

### C1 on N (CS-9 step 2)

**Outputs.** `outputs/costate/exploratory/cs-9/summary.json` (committed copy:
[`costate-followups/cs-9-summary.json`](costate-followups/cs-9-summary.json); the build, K2b and
networks stages' records beside it as `cs-9-build.json`, `cs-9-k2b.json` and
`cs-9-networks.json`) and `rows.npz` (per row, local).


**Rows.** Each agent's N states, all of them approach-phase interior rows. λ_s and λ_ξ are
scored on 699–760 rows per decision network (s is observable on every one). That is 0.43–0.46
times C1's 1,643 s rows on T and 0.44–0.48 times its 1,581 ξ rows. The bootstrap is over the
26–29 geometries present.

| Network | rows | R²_c s / ξ | C1 statistic [CI] | band | ξ Spearman [CI] | best random critic: ξ R²_c; statistic minus its [CI] | T's confirmatory C1 [CI] |
|---|---|---|---|---|---|---|---|
| G1 | 699 | −0.025 / −0.084 | −0.054 [−0.711, −0.025] | refute | 0.17 [−0.10, 0.43] | 0.59; −0.34 [−0.78, −0.24] | 0.010 [−0.060, 0.025] |
| G1s1 | 709 | −0.035 / −0.043 | −0.039 [−0.527, −0.002] | refute | 0.24 [−0.08, 0.55] | 0.10; −0.07 [−0.47, 0.00] | −0.726 [−1.354, −0.389] |
| G1s2 | 705 | −0.026 / −0.055 | −0.041 [−0.614, −0.027] | refute | 0.08 [−0.15, 0.29] | 0.03; −0.04 [−0.37, −0.03] | −1.403 [−2.818, −0.593] |
| G2 | 720 | −0.044 / −0.034 | −0.039 [−0.360, 0.043] | refute | 0.35 [−0.06, 0.67] | 0.54; −0.28 [−0.52, −0.12] | −0.049 [−0.179, 0.012] |
| G2s1 | 742 | −0.018 / −0.152 | −0.085 [−0.313, −0.057] | refute | 0.21 [0.02, 0.41] | 0.17; −0.16 [−0.32, −0.05] | −2.477 [−4.802, −1.081] |
| G2s2 | 723 | −0.027 / −0.073 | −0.050 [−0.505, −0.040] | refute | 0.08 [−0.16, 0.33] | 0.04; −0.06 [−0.30, −0.03] | −0.168 [−0.475, −0.036] |
| G3 | 750 | −0.021 / −0.098 | −0.060 [−0.366, −0.017] | refute | 0.16 [−0.09, 0.34] | 0.18; −0.13 [−0.34, −0.09] | −1.359 [−2.932, −0.364] |
| G3s1 | 760 | −0.048 / 0.474 | 0.213 [−0.023, 0.290] | inconclusive | 0.62 [0.36, 0.83] | 0.45; 0.03 [−0.10, 0.10] | −2.575 [−5.035, −1.054] |
| G3s2 | 759 | −0.043 / −0.024 | −0.034 [−0.221, −0.007] | refute | 0.25 [0.09, 0.41] | 0.41; −0.21 [−0.33, −0.14] | −1.386 [−2.815, −0.480] |

The statistic is C1's, with the frozen calibration, T's partition-0 folds, key weights and the
single-geometry rule. Its one null is the best of the 20 random-init critics: the critic with the
largest statistic, whose ξ R²_c is shown. The value-fit, QSS, barrier and far-geometry nulls are
not rebuilt on N (plan CS-9). The last column is `decisions.json`'s C1 on T. No decision changes
under the single-geometry rule.

- **8 of 9 refute; G3s1 does not.** Every other network's CI upper end is below 0.05 (G2's 0.043
  is the largest). G3s1's ξ R²_c is 0.474, its statistic 0.213 with CI [−0.023, 0.290]:
  inconclusive. No single geometry carries it: removing any one of its 29 keeps the set
  inconclusive, with the statistic between 0.17 and 0.26.
- **G3s1's ξ is no better than a random critic's.**
  - On N, the best random-init critic's ξ R²_c is 0.03–0.59 across the 9. G3s1's scores 0.45,
    and G3s1's statistic exceeds that critic's by 0.03 [−0.10, 0.10].
  - In 8 of the 9 networks the trained statistic lies below the best random critic's, with the
    CI below 0 in 7.
  - The likely reason is the state set. N holds one state per key, all at about the same
    distance before the apex (median 183 m) and speed (85 m/s). λ_ξ there varies mainly across
    geometries, and its SD on N is an eighth of T's approach rows' (0.18 against 1.42). Any
    gradient that differs across geometries through J_z can then calibrate to it.
- **s calibrates to nothing.** s's R²_c is −0.048 to −0.018 in all 9. G3s1's fitted s slope is
  clamped at 0 in every fold (intercepts only).
- **Ranks.** ξ's Spearman is 0.08–0.62. Its CI is above 0 in 3 (G2s1 0.21, G3s1 0.62, G3s2
  0.25), and every other CI covers 0. s's Spearman is −0.40 to 0.10; G3s2's is negative with its
  CI below 0.
- **Specialists (descriptive).** S1 refutes (0.012 [−0.205, 0.098]). S2 (−0.036 [−0.701, 0.222])
  and S3 (0.160 [−0.125, 0.267]) are inconclusive. Their best random critics reach ξ R²_c
  0.50–0.67, above their own (−0.04 to 0.33).

### C2 and C2-sw on N (CS-9 steps 3 and 4)

**Rows.** C2 is fitted on each agent's N stratum (all its rows, every one an approach interior
row) and scored on s and ξ there: 699–760 rows per decision network. C2-sw is scored on the
agent's pedal-switching rows, 699–760 (93–100% of N). λ_res is f̂_D cross-fitted on the agent's
own N rows.

| Network | C2: statistic [two-level CI] | band | h3 rank of 21 | C2-sw: statistic [two-level CI] | band | h3 rank of 21 | C2-sw's R²(z) | T's C2; C2-sw |
|---|---|---|---|---|---|---|---|---|
| G1 | 0.001 [−0.001, 0.005] | refute | 4 | 4.5 [−0.07, 25.8] | inconclusive | 1 | −1.41 | inconclusive; refute |
| G1s1 | 0.001 [−0.002, 0.001] | refute | 8 | −0.002 [−0.057, 0.051] | inconclusive | 11 | 0.59 | inconclusive; refute |
| G1s2 | −0.001 [−0.003, 0.021] | refute | 16 | 42.2 [−0.07, 291.3] | inconclusive | 5 | −27.2 | inconclusive; refute |
| G2 | 0.056 [−0.171, 0.175] | inconclusive | 1 | 0.008 [−0.014, 0.021] | refute | 2 | 0.62 | inconclusive; refute |
| G2s1 | −0.027 [−0.402, 0.056] | inconclusive | 21 | −26.9 [−187.4, 0.002] | refute | 19 | −7.91 | inconclusive; refute |
| G2s2 | −0.000 [−0.005, 0.002] | refute | 20 | 0.001 [−0.003, 0.008] | refute | 10 | 0.54 | refute; refute |
| G3 | 0.017 [−0.016, 0.121] | inconclusive | 1 | 5.1 [0.030, 28.1] | confirm | 1 | −9.21 | inconclusive; refute |
| G3s1 | −0.038 [−0.105, 0.007] | refute | 14 | 0.19 [−0.10, 1.33] | inconclusive | 9 | 0.92 | inconclusive; refute |
| G3s2 | −0.004 [−0.015, 0.004] | refute | 16 | 9.0 [−0.0004, 49.2] | inconclusive | 1 | −15.2 | refute; refute |

- **C2: no confirm, 6 refute.** G2 (0.056), G2s1 and G3 are inconclusive. G2's point is above
  0.05, but its two-level CI is [−0.171, 0.175]. No decision moves under the single-geometry
  rule.
- **f̂_D residualises badly on N.**
  - λ_res's share of λ's variance on N is 0.43–1.38 for s and 0.13–0.94 for ξ, against 0.224 and
    0.248 on T's interior.
  - Above 1 the decoy model adds variance. N's narrow states (one approach distance, one speed
    band) are predicted from neighbours at other distances.
  - So C2's targets on N hold more variance that no network can explain than on T.
- **C2-sw on N cannot be read as an R² gain.** σ^ped's SD on N is a quarter of T's (0.017
  against 0.066 on T's approach and exit switching rows). The z-only probe's out-of-fold R² then
  ranges from −27 to 0.92 across the 9 networks, and the statistics are differences of such
  numbers (−26.9 to 42.2).
  - G3's 'confirm' (5.1 [0.030, 28.1]) has a z-only probe at R² −9.2, against the full probe's
    0.60 and the pool's median −4.5. h3 repairs an extrapolation failure of z. That is not
    evidence of switching-function information.
  - Where the z-only probe holds up (R²(z) 0.54–0.92: G1s1, G2, G2s2, G3s1), the statistics are
    −0.002 to 0.19, three of them within ±0.01.
  - The co-reported Spearman ρ(m_pedal − 1, σ^ped) is 0.06–0.94.
- **Specialists (descriptive).** C2: S1 refutes; S2 and S3 are inconclusive (their z-and-m-only
  R² on ξ are −76 and −32). C2-sw: S1 refutes; S2 is inconclusive under the single-geometry rule
  (its full set refutes); S3 is inconclusive.

### CS-10 (iii): C1-own on N (CS-9 step 5)

**Outputs.** `outputs/costate/exploratory/cs-9/cs10-iii.json` (committed copy:
[`costate-followups/cs-10-iii-summary.json`](costate-followups/cs-10-iii-summary.json)).

**Rows.** λ^own_1 comes from each N state over W − 1 = 399 steps of the policy's own loop, with
the agent's own a_0 as prev_action. A run that terminates within them is dropped: G1s2 6, G2 12,
G2s2 13, none in the other 6. The scored rows are 699–760 per decision network.

| Network | rows | R²_c s / ξ | C1-own statistic [CI] | ξ Spearman [CI] | confirmatory C1-own [CI] |
|---|---|---|---|---|---|
| G1 | 699 | −0.039 / −0.028 | −0.034 [−0.123, −0.015] | 0.20 [−0.02, 0.41] | −0.122 [−0.313, −0.048] |
| G1s1 | 709 | −0.006 / −0.001 | −0.004 [−0.548, −0.003] | 0.01 [−0.14, 0.18] | −0.054 [−31.190, −0.034] |
| G1s2 | 699 | −0.007 / −0.031 | −0.019 [−0.095, −0.011] | 0.03 [−0.20, 0.27] | −0.047 [−11.153, −0.034] |
| G2 | 708 | −0.152 / −0.189 | −0.171 [−0.410, −0.125] | 0.03 [−0.16, 0.19] | 0.029 [−0.284, 0.046] |
| G2s1 | 742 | −0.032 / −0.035 | −0.034 [−0.598, −0.019] | −0.08 [−0.23, 0.10] | 0.026 [−0.425, 0.146] |
| G2s2 | 710 | −0.037 / 0.011 | −0.013 [−0.237, 0.008] | 0.31 [0.09, 0.48] | −0.150 [−0.979, 0.115] |
| G3 | 750 | −0.048 / −0.128 | −0.088 [−0.225, −0.069] | −0.11 [−0.29, 0.10] | −0.126 [−0.624, 0.035] |
| G3s1 | 760 | −0.022 / −0.079 | −0.050 [−0.711, −0.034] | −0.16 [−0.34, 0.02] | −0.075 [−0.878, 0.113] |
| G3s2 | 759 | −0.090 / −0.105 | −0.098 [−0.287, −0.062] | −0.07 [−0.29, 0.16] | −0.031 [−0.716, 0.152] |

- **9 of 9 refute.** The largest CI upper end is 0.008 (G2s2). All 9 have 'secondary' status
  (K3b's trained analogue and value accuracy passed).
- **Companions (descriptive).** Winsorised at T's bounds, all 9 still refute (largest CI upper
  end 0.008). The accel-held form repeats the primary.
- **Ranks.** ξ's Spearman is −0.16 to 0.31, with its CI above 0 only in G2s2.
- **G3s1 against its own policy.** Its ξ calibrates against λ* on N (R²_c 0.474) but not against
  its own policy's λ^own (−0.079).
- **Specialists (descriptive).** Their runs mostly crash from N's fast approaches: S1 keeps 200
  of 766 runs, S2 none of 404, S3 20 of 393. S1 refutes (−0.135 [−0.910, −0.099]); S2 is not
  testable; S3's 20 rows give −11.5.

### N's CS-11: the policy's own width price at N (CS-9 step 6)

**Outputs.** `outputs/costate/exploratory/cs-9/cs11-n.json` (committed copy:
[`costate-followups/cs-11-n-summary.json`](costate-followups/cs-11-n-summary.json)).

**Rows.** N's approach-phase interior rows whose run is kept at both horizons (W − 1 = 399 and
H = 800). They are the same 699–760 rows as CS-10 (iii)'s: every run that survives also keeps a
positive inside buffer. No Π^{in,own} is 0.

| Network | rows | critic–own R² [CI] | own–optimum R² [CI] | own–optimum b_gl [CI] | critic–optimum R² [CI] | on D: critic–own; own–optimum; critic–optimum |
|---|---|---|---|---|---|---|
| G1 | 699 | 0.083 [−0.192, 0.243] | 0.323 [0.127, 0.667] | 0.43 [0.25, 0.69] | 0.339 [0.102, 0.741] | 0.047; 0.268; 0.145 |
| G1s1 | 709 | 0.277 [0.073, 0.493] | 0.478 [0.226, 0.712] | 0.64 [0.39, 0.95] | 0.462 [0.238, 0.679] | 0.112; 0.390; 0.217 |
| G1s2 | 699 | 0.037 [−0.230, 0.183] | 0.380 [0.076, 0.642] | 0.61 [0.29, 1.04] | 0.472 [0.249, 0.650] | 0.008; 0.079; 0.239 |
| G2 | 708 | 0.264 [0.002, 0.395] | 0.585 [0.377, 0.748] | 0.74 [0.47, 1.05] | 0.591 [0.402, 0.693] | 0.031; 0.291; 0.068 |
| G2s1 | 742 | −0.008 [−0.109, 0.069] | 0.213 [−0.067, 0.409] | 0.74 [0.27, 1.34] | 0.581 [0.375, 0.664] | 0.019; 0.076; 0.280 |
| G2s2 | 710 | 0.368 [0.172, 0.678] | 0.545 [0.285, 0.867] | 0.99 [0.61, 1.45] | 0.436 [0.159, 0.788] | −0.007; 0.282; 0.129 |
| G3 | 750 | 0.108 [−0.122, 0.329] | 0.407 [0.169, 0.624] | 0.82 [0.45, 1.34] | 0.406 [0.204, 0.577] | −0.030; 0.275; 0.263 |
| G3s1 | 760 | 0.184 [−0.046, 0.375] | 0.673 [0.506, 0.785] | 0.76 [0.53, 1.04] | 0.442 [0.137, 0.618] | 0.037; 0.343; 0.334 |
| G3s2 | 759 | 0.316 [−0.144, 0.623] | 0.308 [−0.175, 0.629] | 0.58 [0.25, 0.92] | 0.817 [0.718, 0.865] | 0.013; 0.049; 0.350 |

The comparisons are W1's (C1's calibration, raw targets), bootstrapped over N's geometries. The
last column is CS-11 on D (above), on D's 453–476 rows.

- **At N every comparison is higher than at D.**
  - The critic's width derivative against the optimum's Π^in reaches 0.34–0.82 (CI above 0 in
    all 9), against 0.07–0.35 at D.
  - The policy's own price against Π^in reaches 0.21–0.67 (CI above 0 in 7), against 0.05–0.39.
  - The likely reason is again the state set. N's states share one approach distance and speed
    band, so Π^in at them varies mainly with the geometry (median 0.48), and a width signal that
    tracks the geometry calibrates well. D's rows spread over T's approach starts.
- **The critic still agrees better with the optimum than with its own policy.** critic–own is
  −0.008 to 0.368, below critic–optimum in all 9.
- **The policies price the inside wall below the optimum.** b_gl (Π^{in,own} on Π^in) is
  0.43–0.99, with its CI below 1 in 3 (G1, G1s1, G3s2). The median Π^{in,own} is 0.02–0.26,
  against Π^in's 0.48.
- **Specialists (descriptive).** S1 (200 rows) and S3 (20) have negative calibrated R² in every
  comparison; S2 has no kept run.

### N's CS-13: C2-pairs between N and T (CS-9 step 7)

**Outputs.** `outputs/costate/exploratory/cs-9/cs13-n.json` (committed copy:
[`costate-followups/cs-13-n-summary.json`](costate-followups/cs-13-n-summary.json)).

- **Not scorable: no pair exists.** No N state of any agent has a T plan state of the same
  stimulus within 2.5 m of s − s_apex and 1 m/s of vx.
  - N's x_1 lie 173–213 m before the apex at 67–87 m/s (5th–95th percentiles), on step 1's fast
    approaches (its 'B' cells).
  - T's states on the A2 keys' approaches are slower: T's stratum is below 77 m/s at its 95th
    percentile.
- The probes were fitted and passed their wiring, but no Spearman is computed. E13.1 and E13.2
  cannot be read on N.

### Expected directions against the results

- **E9.1** ("C1 on N is in the refute band in all 9"): **does not hold**: 8 of 9. G3s1 is
  inconclusive (0.213 [−0.023, 0.290]), at the level of its best random critic.
- **E9.2** ("C2 on N is inconclusive or in the refute band in all 9"): **holds**: 6 refute and 3
  inconclusive; none is 'not testable'.
- **E9.3** ("C2-sw on N is in the refute band", read as at least 6 of 9): **does not hold**: 3 of
  9 (G2, G2s1, G2s2). G3 is 'confirm' and 5 are inconclusive, on a statistic that N's small σ^ped
  variance makes uninterpretable (above).
- **E9.4** ("K2b on N excludes n and vx on approach, as on T"): **does not hold**: N's own rule
  excludes nothing (vx's p90 0.091, s's 0.070).
- **E9.5** ("C1-own on N is in the refute band"): **holds**: 9 of 9.
- On N's CS-11 rows (amendment 9 item 5, descriptions): E11.1 holds in 0 of 9, E11.2 in 3 of 9,
  and E11.3 in 0 of 9 (own–optimum's R² is 0.21–0.67).
- E13.1 and E13.2 cannot be read on N (no pair).

### What CS-9 says (exploratory)

- **At the networks' own states the critics' gradients still do not calibrate to λ.** C1 refutes
  in 8 of 9, and C1-own (against the policy's own costate) in 9 of 9. G3s1's ξ calibrates to λ*
  on N, but no better than its best random-init critic, and not to its own policy's λ^own.
- **N is a narrow state set.** It holds one state per key, at about 183 m before the apex and
  85 m/s. On it λ varies mainly with the geometry, so:
  - random gradients calibrate on ξ;
  - f̂_D adds variance rather than removing it;
  - σ^ped varies little.

  These properties, not the networks, decide C2-sw's numbers on N. They probably also raise W1's
  comparisons above D's.
- **The width prices.** At N the critic's inside-width derivative tracks the optimum's Π^in
  across geometries (R² 0.34–0.82), better than it tracks its own policy's price. The policies
  value the inside wall at about half the optimum's price or less (medians 0.02–0.26 against
  0.48).
- **The plan's reading of a score above 0.2** ("the critics' gradients are better at the states
  their own policies reach") does not apply: G3s1's 0.213 is inconclusive and matched by random
  critics.
- **Scope.** Approach rows only: N has no exit, braking or corner state. K2b on N placed its
  second corner just beyond x_1's preview, much nearer than on T, and still excluded nothing.
- None of this changes a decision of `costate-prereg-v1`.

## CS-6, CS-12, CS-14 and CS-15: the runs (exploratory)

- **Plan and code.** All four ran at `937b712` (code and tests), under the plan with amendment 10
  (`5cf5d2d`: their operational details, committed alone before any of their numbers).
- **CS-6.** `scripts/costate_exploratory.py cs-6`, in one process, niced, in 637 s: about 4.4 s
  per snapshot, as amendment 10 measured on a random-init stand-in.
- **CS-12.** `cs-12`: 50 units ((network, statistic)) on 4 workers × 2 XLA threads, niced, in
  33 min (C2 about 195 s and C2-sw about 70 s of worker time per network). The plan estimated
  45–60 min.
- **CS-14.** `cs-14`: 15 units (blocks of at most 32 of T's keys), same pool, in about 30 s (31 s
  from the stamps' start to end; `seconds` 28.9).
- **CS-15.** `cs-15`: 36 units ((network, part)), same pool, in 32 min (about 200–220 s of worker
  time per unit). The plan estimated about 1 h.
- **Development runs.** Before the canonical runs, partial runs were written outside
  `outputs/costate` to check the wiring and the timings: CS-6 on G1's last and first snapshots,
  CS-12 on S2, CS-14 on its first block of 32 keys, and CS-15 on G1. Only their structure,
  wiring gaps and timings were read.
- **Stamps.** Every output records an empty `changed_vs_head`, no partial flag, HEAD `937b712`,
  the plan's sha256 with 10 amendments, and the sha256 of every input:
  - CS-6, 164 files: the world and its summary, the 9 networks' critic and K3 files, and the 144
    snapshots through the record's pins (the last of each is the network's own pin);
  - CS-12, 53 files: the world and its summary, the 25 networks' activations (each read only
    after its stamp passed R3's rule) and checkpoints, and CS-8's summary;
  - CS-14, 3 files: the world, its summary and `gates/k2b.json`;
  - CS-15, 21 files: the world and its summary, the 9 networks' activations and checkpoints, and
    CS-8's summary.

  They were re-verified before this note: each stamp's script sha256 is the script's blob at
  `937b712` and its plan sha256 is the plan at `5cf5d2d`; the npz sha256s and the summaries'
  content sha256s were recomputed and match; the script's `verify` reports 0 problems.
- **Wiring checks (all passed).**
  - **CS-6.** At each network's last snapshot, the pullback equals the stored g exactly (gap 0 on
    all 8 components; tolerance 10⁻¹⁰ of each row's largest |g|); the decision mask is
    `decisions.json`'s; and C1 (R²_c of s and ξ, the statistic, its CI, the per-fold a and b)
    reproduces `decisions.json` to at most 3.6·10⁻¹⁵ (tolerance 10⁻⁸).
  - **CS-12.** The frozen λ_res's interior residual shares equal the record's exactly. In every
    unit, the 20 pool actors rebuilt on the network's z give the stored controls of seeds 0–4
    (h3 and m) exactly.
  - **CS-14.** At every T row, the re-polished plan's s_t equals T's column exactly, and so do its
    phase, s_b and W.
  - **CS-15.** The primary C2 assembled from CS-15's own probes equals CS-8 (c)'s f̂_D reference
    record exactly in all 9 (gap 0 on the statistic, its CIs, the gains, R² and the pool rank).
    In every unit the pool actor of seed 0 rebuilds its stored h3 exactly.
- **Calibrated R².** CS-6 reports R²_c under the frozen calibration only: CS-1's rule for later
  items did not fire.

## CS-6. C1-dev over the 16 snapshots (exploratory)

**Run** as above. Outputs: `outputs/costate/exploratory/cs-6/summary.json` (committed copy:
[`costate-followups/cs-6-summary.json`](costate-followups/cs-6-summary.json)) and `g.npz` (per
snapshot, the main V's g and the decision mask; local). **Rows.** C1's rows: λ_s 1,643 and λ_ξ
1,581 at every snapshot. s's observability, read under each snapshot's own obs_rms, leaves the
same 1,643 s rows at all 144 snapshots. No braking or corner row is scored (decision #8, plan R5).

**C1's statistic over training** (the mean of R²_c over s and ξ, C1 without nulls; 95% CI):

| Network | 10M | Best snapshot | Last (200M) | Snapshots with CI above 0 | ρ with log timesteps |
|---|---|---|---|---|---|
| G1 | 0.110 [0.035, 0.137] | 10M | 0.010 [−0.060, 0.025] | 10M, 12M | −0.55 |
| G1s1 | 0.051 [−0.032, 0.074] | 0.076 at 12M | −0.726 [−1.354, −0.389] | none | −0.96 |
| G1s2 | 0.013 [−0.055, 0.025] | 0.078 at 15M | −1.403 [−2.818, −0.593] | none | −0.96 |
| G2 | 0.149 [0.076, 0.175] | 10M | −0.049 [−0.179, 0.012] | 10M, 12M | −0.71 |
| G2s1 | 0.091 [0.005, 0.118] | 10M | −2.477 [−4.802, −1.081] | 10M | −0.98 |
| G2s2 | 0.049 [−0.029, 0.075] | 0.087 at 12M | −0.168 [−0.475, −0.036] | 12M | −0.77 |
| G3 | 0.086 [0.004, 0.115] | 10M | −1.359 [−2.932, −0.364] | 10M | −0.95 |
| G3s1 | 0.093 [−0.002, 0.123] | 0.114 at 12M | −2.575 [−5.035, −1.054] | 12M, 15M | −0.95 |
| G3s2 | 0.036 [−0.040, 0.053] | 0.091 at 15M | −1.386 [−2.815, −0.480] | 12M, 15M | −0.94 |

(12M and 15M are the 12.2M and 14.9M snapshots; the replicates' last snapshot is at 199.9M.)

- Every one of the 144 snapshots is in the refute band. The largest CI upper end is 0.175 (G2 at
  10M).
- The statistic falls with training in all 9 (ρ −0.55 to −0.98). ξ drives the fall:
  - ξ's R²_c is 0.04–0.31 at 10M. It turns negative between 22M and 49M timesteps and ends at
    −5.0 to 0.006 (its ρ is −0.62 to −0.98 in all 9);
  - s's R²_c stays between −0.49 and 0.03 at every snapshot, with no common direction (ρ −0.73
    to 0.42).
- The early lift is small: no snapshot passes 0.15. 7 of 9 networks have an early snapshot (10M
  to 15M) whose CI lies above 0; no snapshot after 15M has one.
- The calibration's slope is clamped at 0 in some fold of s at 60 of the 144 snapshots, and of ξ
  at 3.

**Rank agreement over training** (the weighted Spearman of g against λ on C1's rows):

| Network | ξ: 10M | ξ: last | ξ: ρ with log timesteps | s: 10M | s: last | s: ρ |
|---|---|---|---|---|---|---|
| G1 | 0.378 | 0.050 | −0.72 | 0.182 | 0.197 | −0.49 |
| G1s1 | 0.267 | 0.336 | 0.36 | 0.182 | 0.100 | −0.70 |
| G1s2 | 0.080 | 0.342 | 0.06 | 0.122 | 0.155 | 0.33 |
| G2 | 0.416 | −0.040 | −0.74 | 0.165 | 0.138 | 0.02 |
| G2s1 | 0.375 | 0.346 | −0.24 | 0.235 | 0.192 | −0.38 |
| G2s2 | 0.214 | 0.024 | −0.28 | 0.239 | 0.006 | −0.89 |
| G3 | 0.303 | 0.388 | 0.39 | 0.174 | 0.321 | 0.87 |
| G3s1 | 0.323 | 0.398 | 0.11 | 0.021 | 0.223 | 0.80 |
| G3s2 | 0.166 | 0.363 | 0.07 | 0.085 | 0.158 | 0.43 |

- ξ's Spearman is positive at most snapshots (its CI lies above 0 at 119 of 144). It is 0.08–0.42
  at 10M and −0.04 to 0.40 at the last snapshot, and it does not rise with training as a rule:
  its trend is positive in v3's three seeds, in two of v1's (G1s1, G1s2), and in none of v2's.
- s's rank trend is positive in all three v3 seeds, and mixed in v1 and v2.

**Expected directions against the results** (amendment 10 item 2's reading).

| Expected | Result | Holds? |
|---|---|---|
| E6.1: no snapshot's CI upper end reaches 0.2 | All 144 below 0.2 (largest 0.175) | yes |
| E6.2: ξ's Spearman trend ρ > 0 in at least 2 of 3 seeds in each variant | v1 2 of 3, v2 0 of 3, v3 3 of 3 | no (2 of 3 variants) |
| E6.3: R²_c's own trend not predicted | ρ −0.55 to −0.98: it falls in all 9 | (reported) |

**What CS-6 says (exploratory).**
- The critics' calibrated agreement with λ is best early in training and falls as training goes
  on. In every network the 10M–15M critics score highest, and by 200M ξ's calibration has
  collapsed. This looks like CS-1's heavy ξ tail growing with training, not like a gradient
  that approaches λ and then stalls. (CS-6 does not locate the tail at each snapshot.)
- The ordering does not improve as a rule either: the gradient's rank agreement with λ_ξ is
  about as good at 10M as at 200M, and its trend depends on the variant. So of the plan's two
  readings, "better ordering, not shape" does not fit, and "set early" fits the ordering only:
  the shape gets worse with training.
- Even at its best (G2 at 10M, 0.149) no snapshot leaves the refute band.

## CS-12. C2 and C2-sw beyond the 9 decision networks (exploratory)

**Run** as above. Output: `outputs/costate/exploratory/cs-12/summary.json` (committed copy:
[`costate-followups/cs-12-summary.json`](costate-followups/cs-12-summary.json)). **Networks.**
The 22 generalists outside the decision set and S1–S3. **Rows.** C2 is fitted on T's interior
approach and exit rows (1,653) and scored on s's 1,643 and ξ's 1,581 decision rows (every
network keeps all of them); C2-sw on the 2,507 approach and exit pedal-switching rows, as CS-8.
**Controls.** The 20 pool actors, computed on each network's own z. Decisions are the ruled ones
(after the single-geometry rule); the unruled one is given where it differs.

| Network | C2 [two-level CI] | C2 decision | C2 rank | C2-sw [two-level CI] | C2-sw decision | C2-sw rank |
|---|---|---|---|---|---|---|
| v1 s3 | 0.229 [−0.170, 1.652] | inconclusive | 3 | 0.054 [0.029, 0.124] | inconclusive (confirm before the rule) | 1 |
| v1 s4 | −0.027 [−0.740, 0.126] | inconclusive | 16 | 0.042 [0.025, 0.084] | inconclusive | 1 |
| v2 s3 | 0.006 [−0.352, 0.244] | inconclusive | 13 | 0.042 [0.015, 0.092] | inconclusive | 1 |
| v2 s4 | 0.061 [−0.290, 0.371] | inconclusive | 9 | 0.024 [0.003, 0.074] | inconclusive | 2 |
| v3 s3 | 0.022 [−0.290, 0.335] | inconclusive | 9 | −0.017 [−0.053, 0.005] | refute | 16 |
| v3 s4 | 0.119 [−0.069, 1.088] | inconclusive | 2 | 0.046 [0.020, 0.100] | inconclusive | 1 |
| LOO f1 s0 | −0.014 [−0.780, 0.195] | inconclusive | 11 | 0.029 [−0.000, 0.083] | inconclusive | 2 |
| LOO f1 s1 | −0.005 [−0.877, 0.897] | inconclusive | 9 | 0.030 [0.013, 0.065] | inconclusive | 2 |
| LOO f1 s2 | −0.077 [−0.442, 0.084] | inconclusive | 17 | −0.008 [−0.038, 0.018] | refute | 15 |
| LOO f1 s3 | −0.038 [−0.564, 0.129] | inconclusive | 13 | 0.009 [−0.006, 0.040] | refute | 6 |
| LOO f2 s0 | 0.041 [−0.132, 0.365] | inconclusive | 8 | 0.018 [−0.006, 0.054] | inconclusive | 3 |
| LOO f2 s1 | −0.159 [−1.130, 0.070] | inconclusive | 21 | 0.026 [0.007, 0.057] | inconclusive | 1 |
| LOO f2 s2 | −0.144 [−0.584, 0.024] | inconclusive (refute before the rule) | 21 | 0.018 [−0.009, 0.034] | refute | 2 |
| LOO f2 s3 | 0.303 [−0.033, 1.860] | inconclusive | 2 | 0.038 [0.019, 0.076] | inconclusive | 1 |
| real-width v1 s0 | 0.136 [0.0003, 0.650] | inconclusive (confirm before the rule) | 3 | 0.012 [−0.012, 0.036] | refute | 7 |
| real-width v1 s1 | 0.086 [−0.017, 0.448] | inconclusive | 2 | 0.025 [−0.010, 0.060] | inconclusive | 2 |
| real-width v1 s2 | −0.105 [−0.674, 0.202] | inconclusive | 21 | 0.033 [−0.007, 0.090] | inconclusive | 1 |
| real-width v1 s3 | 0.014 [−0.187, 0.385] | inconclusive | 11 | 0.037 [0.021, 0.140] | inconclusive | 1 |
| real-width v3 s0 | 0.080 [−0.093, 0.573] | inconclusive | 2 | 0.008 [−0.013, 0.027] | refute | 8 |
| real-width v3 s1 | −0.016 [−0.378, 0.110] | inconclusive | 13 | 0.061 [0.027, 0.129] | inconclusive (confirm before the rule) | 1 |
| real-width v3 s2 | 0.180 [−0.223, 1.668] | inconclusive | 2 | 0.027 [0.007, 0.061] | inconclusive | 1 |
| real-width v3 s3 | −0.001 [−0.407, 0.222] | inconclusive | 10 | 0.031 [0.011, 0.064] | inconclusive | 2 |
| S1 | −0.131 [−1.119, 0.038] | inconclusive (refute before the rule) | 20 | 0.029 [0.007, 0.079] | inconclusive | 1 |
| S2 | 0.277 [−0.005, 1.705] | inconclusive | 4 | 0.046 [0.021, 0.101] | inconclusive | 1 |
| S3 | −0.136 [−1.138, 0.470] | inconclusive | 20 | −0.218 [−0.838, 0.028] | refute | 21 |

(v1 s3 is `runs/dr-pilot-200M-s3`, v2 and v3 the `tilt30` and `tilt50a2` replicates,
real-width the `dr-realw-v1` and `dr-realw-v3` runs, LOO f1 and f2 the `dr-realw-loo-f1` and
`-f2` runs. "Rank" is the trained network's rank of 21 among its pool's own gains.)

**C2.**
- All 25 are inconclusive after the rule. Before it, real-width v1 s0 would confirm (0.136, its
  CI's lower end 0.0003) and LOO f2 s2 and S1 would refute; the rule moves each, on 24, 3 and 24
  geometries respectively.
- The two-level CIs are wide, as on the 9 (CS-8 (c)): their upper ends are 0.02–1.86. The s
  component drives both the width and the large points (gains on λ_res,s up to 0.59), where the
  base R² on λ_res,s is far below 0 (CS-7).
- The trained h3 ranks first of 21 in none of the 25; it is second or third in 7.

**C2-sw.**
- The trained h3 ranks first of 21 in 10 of the 22 generalists and second in 6 more. Its
  two-level CI lies above 0 in 12 of 22. Two networks (v1 s3, real-width v3 s1) would confirm
  before the rule; geometry 17 (R 14.6 m, θ 45°) moves each to inconclusive.
- By recipe: first in 4 of the 6 baseline replicates (v1, v2, v3 s3 and s4), 2 of 4 real-width
  v1, 2 of 4 real-width v3, 2 of 4 LOO f2, and none of the 4 LOO f1.
- The specialists split: S1 and S2 rank first (CIs above 0), S3 last (refute).
- With the 9 decision networks on the same rows (CS-8 (a): first in 7 of 9), 17 of the 31
  generalists rank first.

**Expected directions against the results** (amendment 10 item 3's reading).

| Expected | Result | Holds? |
|---|---|---|
| E12.1: C2 inconclusive or refute in at least 22 of 25, confirm in none | 25 of 25 inconclusive; no confirm (1 before the rule) | yes |
| E12.2: C2-sw first of 21 in at least 15 of 22 generalists, at most 1 of 3 specialists | 10 of 22; 2 of 3 specialists | no |

**What CS-12 says (exploratory).**
- C2's pattern holds beyond the 9: no network shows costate information in h3 beyond z, m and
  f̂_D's decoys, and none can be refuted either. The CIs are too wide on these rows.
- C2-sw's rank-1 pattern is weaker beyond the 9: about half the other generalists, and two of
  the three specialists, rank first. So the trained h3's extra C2-sw gain over random actors is
  common but not general, and Hockenheim training does not remove it. On the 9, CS-8 (b) found
  that this gain does not survive residualising σ^ped on position and speed, so it is
  consistent with the decoys design 15.10 names. CS-12 did not test that variant here (primary
  only, R6).
- Geometry 17 (θ 45°) again decides where C2-sw's band word would confirm, as in CS-8's null.

## CS-14. q_c, the action-projected critic signal (exploratory)

**Run** as above. Output: `outputs/costate/exploratory/cs-14/summary.json` (committed copy:
[`costate-followups/cs-14-summary.json`](costate-followups/cs-14-summary.json)). Targets only: no
network was evaluated, and K2b for q_c was not run (amendment 10 item 4).

**The window rule** (plan step 4). A T interior row passes when the plan states x_t … x_{t+64}
all lie in its own phase.

| Phase | Interior rows | Pass | Fail: window leaves the phase | Fail: t + 64 > W |
|---|---|---|---|---|
| Approach | 1,581 | 62 | 1,519 | 0 |
| Exit | 72 | 72 | 0 | 0 |
| Total | 1,653 | 134 | 1,519 | 0 |

- The passing rows come from 110 of the 477 keys. A 64-step window spans 65–276 m of road (median
  176 m; 124–276 m on the passing rows), so it reaches the brake onset from almost every approach
  row.
- The bounds read before amendment 10 (36 to 113 approach rows) bracket the 62.

**The stop rule** (plan step 8). 134 rows is fewer than 200, so CS-14 is **not scorable under #8**
and stops. K2b for q_c can only remove rows, so it could not change this, and was not run (about
53 min of solves saved). E14.1–E14.3 are not read.

**What CS-14 says (exploratory).** Under decision #8, the action-projected critic signal cannot be
scored on T. With GAE's 64-step window, almost every approach state's advantage already reads the
braking phase. q_c can only be scored with windows that reach braking and corner rows, which #8
reserves for BC-1. Whether q_c belongs in BC-1's design is the lead's call.

## CS-15. The FWL check, actor h1 and h2 probes, C2 on D (exploratory)

**Run** as above. Outputs: `outputs/costate/exploratory/cs-15/summary.json` (committed copy:
[`costate-followups/cs-15-summary.json`](costate-followups/cs-15-summary.json)) and `preds.npz`
(step 1's base, full and FWL predictions; local). **Rows.** Steps 1 and 2: T's interior approach
and exit rows (1,653), scored on s's 1,643 and ξ's 1,581 decision rows. Step 3: D's 477 rows,
every one a decision row for both s and ξ. λ_res is the frozen f̂_D. The step 1 record is CS-8
(c)'s f̂_D reference, reproduced exactly.

**Step 1: FWL against the one-stage nested ridge.** "Raw gain" is the trained h3's own
R²(full) − R²(base), the mean over s and ξ; "statistic" is C2's (trained minus the pool's
median). Differences are FWL minus one-stage, with paired CIs (geometry-only for the raw gain,
two-level for the statistic).

| Network | Raw gain, one-stage | FWL | Difference [CI] | s | ξ | Statistic, one-stage | FWL | Difference [CI] |
|---|---|---|---|---|---|---|---|---|
| G1 | −0.030 | −0.070 | −0.040 [−0.344, −0.000] | −0.073 | −0.006 | −0.058 | −0.045 | 0.013 [−0.023, 0.115] |
| G1s1 | −0.227 | −0.296 | −0.069 [−0.480, −0.012] | −0.097 | −0.040 | −0.192 | −0.224 | −0.032 [−0.247, −0.004] |
| G1s2 | 0.017 | 0.033 | 0.016 [−0.033, 0.164] | 0.006 | 0.026 | 0.066 | 0.119 | 0.053 [0.006, 0.361] |
| G2 | 0.164 | 0.158 | −0.005 [−0.043, 0.032] | −0.003 | −0.008 | 0.250 | 0.279 | 0.029 [−0.003, 0.274] |
| G2s1 | −0.044 | −0.091 | −0.047 [−0.429, 0.006] | −0.100 | 0.006 | 0.028 | 0.020 | −0.008 [−0.144, 0.055] |
| G2s2 | 0.137 | 0.141 | 0.005 [−0.019, 0.094] | 0.003 | 0.007 | 0.185 | 0.233 | 0.048 [0.006, 0.443] |
| G3 | −0.162 | −0.181 | −0.019 [−0.174, 0.027] | −0.033 | −0.004 | −0.114 | −0.116 | −0.002 [−0.015, 0.181] |
| G3s1 | −0.113 | −0.142 | −0.029 [−0.282, 0.012] | −0.048 | −0.009 | 0.043 | 0.045 | 0.002 [−0.082, 0.087] |
| G3s2 | −0.203 | −0.279 | −0.076 [−0.592, −0.015] | −0.119 | −0.034 | −0.126 | −0.155 | −0.030 [−0.220, 0.000] |

- The two fits agree within 0.01 on the raw gain in 2 of 9 (G2, G2s2). FWL's gain is lower in 7
  of 9, by up to 0.076, and its CI lies below 0 in 3.
- s carries the difference (−0.119 to 0.006); on ξ the two differ by −0.040 to 0.026.
- The statistic moves by −0.032 to 0.053. The FWL statistic carries no single-geometry rule, so
  its band word is compared with the one-stage word before the rule: they agree in 8 of 9. In G3
  FWL's CI upper end is 0.052, so the one-stage refute becomes inconclusive.
- A likely reason: both stages use the full model's base penalty α_B, and the two-stage fit
  equals the joint ridge only as α_B → 0 (`probes`' docstring), while the base [z, m] has 224
  columns penalised at whatever α_B the inner CV picks. CS-15 did not record the chosen α_B, so
  this reading is not checked here.

**Step 2: C2 with h1 or h2 in place of h3** (statistic [two-level CI], and the trained layer's
rank of 21 among the pool's same layer):

| Network | h3 (step 1) | h1 | h2 | D (step 3, h3) |
|---|---|---|---|---|
| G1 | −0.058 [−0.619, 0.028], 17 | 0.105 [−0.373, 1.215], 5 | −0.164 [−0.953, −0.061], 21 | −0.051 [−0.469, 0.001], 15 |
| G1s1 | −0.192 [−0.690, 0.020], 19 | 0.014 [−0.162, 0.210], 9 | −0.035 [−0.266, 0.207], 15 | 0.114 [0.006, 0.939], 1 |
| G1s2 | 0.066 [−0.364, 1.328], 4 | −0.018 [−0.353, 0.129], 12 | −0.037 [−0.438, 0.207], 14 | −1.034 [−11.936, −0.014], 20 |
| G2 | 0.250 [−0.069, 1.773], 4 | 0.065 [0.012, 0.362], 3 | 0.076 [−0.149, 0.420], 3 | 0.053 [−0.033, 0.473], 2 |
| G2s1 | 0.028 [−0.216, 0.236], 10 | 0.037 [−0.046, 0.259], 4 | −0.000 [−0.438, 0.220], 11 | 0.007 [−0.315, 0.107], 10 |
| G2s2 | 0.185 [−0.111, 1.516], 2 | 0.066 [−0.041, 0.410], 2 | 0.056 [−0.027, 0.442], 4 | −0.016 [−0.193, 0.173], 12 |
| G3 | −0.114 [−0.714, 0.015], 20 | 0.071 [−0.132, 0.358], 2 | −0.194 [−0.994, −0.048], 21 | −0.002 [−0.152, 0.261], 13 |
| G3s1 | 0.043 [−0.164, 0.385], 9 | −0.055 [−0.506, 0.091], 15 | −0.009 [−0.537, 0.344], 13 | −0.019 [−0.268, 0.127], 10 |
| G3s2 | −0.126 [−0.637, 0.136], 19 | 0.081 [0.001, 0.560], 2 | 0.148 [−0.212, 1.718], 1 | 0.035 [−0.522, 1.208], 4 |

- **h1** is inconclusive in all 9 after the rule. Before it, G2's and G3s2's would confirm (CIs
  just above 0); the rule withdraws both.
- **h2** refutes in G1 and G3 (the trained h2 ranks last of 21 in both), and is inconclusive in
  the other 7.
- Every layer's CI is wide, as h3's is on this stratum (CS-8 (c)). h1 or h2 has a higher
  statistic than h3 in 5 of 9 (h1) and 2 of 9 (h2), mostly where h3's own is negative.

**Step 3: C2 on D.** D's 477 rows allow the inner CV in all 9, so none is "not testable". After
the rule: 7 inconclusive, 2 refute (G1, G1s2). G1s1 would confirm before the rule (0.114
[0.006, 0.939]) and is inconclusive after it.

**Expected directions against the results** (amendment 10 item 5's reading).

| Expected | Result | Holds? |
|---|---|---|
| E15.1: FWL and one-stage gains agree within 0.01 | 2 of 9 (raw gain); FWL lower in 7 of 9, by up to 0.076 | no |
| E15.2: h1's and h2's gains no larger than h3's | h1 4 of 9, h2 7 of 9 | no (h1) |
| E15.3: C2 on D not testable or inconclusive | 7 of 9 inconclusive; G1 and G1s2 refute | yes |

**What CS-15 says (exploratory).**
- **The FWL check does not reproduce the joint fit to 0.01.** The gap sits mostly in s, it
  moves the C2 statistic by at most 0.053, and it changes one unruled band word (G3). It most
  likely reflects the base's ridge penalty (not checked), so it speaks to the probe machinery,
  not to costate information in h3.
- **No actor layer shows costate information beyond z, m and the decoys.** h1, h2 and h3 are all
  inconclusive or refute on these rows; the earlier layers are not systematically below h3,
  because h3's own statistic is near or below 0 in most networks.
- **D alone decides little.** With 477 rows its CIs are wide; two networks refute, one would
  confirm before the single-geometry rule.

## CS-6, CS-12, CS-14 and CS-15: where they leave the refute (exploratory)

- **C1 over training (CS-6).** The critics' gradients never calibrate to λ at any of the 144
  snapshots, and the calibrated agreement is best at 10M–15M timesteps and falls after: training
  makes the gradient's shape worse on ξ, and does not improve its ranking as a rule.
- **C2 and C2-sw beyond the 9 (CS-12).** C2 stays inconclusive everywhere. The C2-sw rank-1
  pattern holds in about half the other generalists and two of three specialists, short of the
  plan's expectation; on the 9 it did not survive residualisation (CS-8 (b)).
- **q_c (CS-14).** Not scorable under decision #8: 134 rows pass the 64-step window rule, short
  of 200. It needs braking and corner rows.
- **Machinery (CS-15).** FWL differs from the joint ridge by up to 0.08 on the raw gain, mostly
  on s; h1 and h2 show no costate information either; D alone is mostly inconclusive.
- None of this changes a decision of `costate-prereg-v1`.

## BC-a0: the run (exploratory, targets only)

- **Plan and code.** BC-a0 ran at `8f2ad68` (the unit cache), under the plan with amendments 11
  to 14: `5491194` (its operational details, committed alone before any of its numbers),
  `f0c8d7c` (its measured cost, after a one-key development run), `f21afaf` (each re-solve's log
  in brief, after a stopped run) and `bf80190` (the unit cache, after a second stopped run). The
  two stopped runs wrote nothing; what was seen from their logs is listed in amendments 13 and 14.
- **Targets only.** No network output of any network was computed or read, no checkpoint was
  opened, and the item does not unlock (decision #8, plan R5).
- **The run.** `scripts/costate_exploratory.py bc-a0`: 359 units (26 keys × 8 continuations, 50
  states × 3, and the wiring key) on 4 workers × 2 XLA threads, niced, beside another workflow on
  the same Mac. It took 10.6 h of wall time and 42 CPU-h of unit time; no unit came from the
  cache. The plan's first estimate was 7 CPU-h; amendment 13's was about 40.
  - Per-key units: median 270 s (W = 400) and 873 s (W = 800).
  - Per-state units: median 175 s (W = 400) and 984 s (W = 800).
- **Stamps.** `summary.json` and `rows.npz` record an empty `changed_vs_head`, no partial flag,
  HEAD `8f2ad68`, the plan's sha256 with 14 amendments, and the sha256 of each input: the bound
  world and its summary, `gates/k2b.json`, the A2 key files' content hash, and the 12 R-dev
  centrelines the grammar is fitted on. Before this note, the stamps' script and plan sha256 were
  checked against their blobs at `8f2ad68`, and the script's `verify` reported 0 problems.
- **Wiring checks (all passed).**
  - K2b's first key, re-solved under both severities, reproduces `gates/k2b.json` with a gap of
    0 (tolerance 10⁻⁶ in SD units).
  - In every unit, the re-polished stored targets equal the bound world's with a gap of 0
    (tolerance 10⁻¹²).
  - SD_T equals `gates.lam_sd`'s.
  - Each strip's first nonzero knot is the one just beyond its run-up, and it turns the drawn
    way.
- **Failures.** 58 of the 358 re-solves failed. None raised an error; each stopped partway up the
  amplitude ramp, after passing amplitudes 0.375–0.875. 54 of them ended with IPOPT at its
  iteration cap.
  - **W = 800 keys:** 54 of 84 failed (32 of 48 per-key, 22 of 36 per-state).
  - **W = 400 keys:** 4 of 274 failed, all on continuation 6.
  - **Dropped under step 4's rule:** 5 of the 6 W = 800 keys (dd6bc9e8, 84ce4d9b, d309952e,
    23fc2c37, e8ea7337, with 5–7 failures each) and 7 of the 50 states. The shares therefore
    rest on 21 keys (20 W = 400 and 2152051c) and 43 states (19 braking, 24 corner).
  - **A possible bias.** Four of the five dropped keys have tight corners (R 35.8–55.9 m), and
    the failed continuations are probably the hardest ones. If so, the shares below understate
    the movement rather than overstate it.
- **Placement facts.**
  - The first corner of the continuation turns the key's own way in 51% of the per-key units.
  - The generator's inside-width cap would have been exceeded on 8 units (largest 0.648, against
    0.6), on 2 keys and continuations 3 and 7. The cap is not applied (amendment 11 item 2).

## BC-a0. Braking and corner targets pilot (exploratory, targets only)

**The question.** How much does realistic road beyond the preview move λ on braking and corner
rows? The answer decides which target route BC-1 should use.

**Two measures.** Δ is the target under a continuation drawn from the v1 grammar minus the stored
target, whose road beyond the preview is a straight.
- **U** is the mean over continuations of Δ², over T's variance of that target within the phase
  (interior rows, each key weighted equally).
- **p90** is K2b's statistic: the 90th percentile of |Δ| / SD_T, with SD_T taken over all of T's
  rows.

**Per-state shares.** The estimand BC-1 would need. Each state's continuations start 20 m beyond
its own preview, so U measures what that state cannot see. CIs are 95% over keys.

| Phase | Component | U [95% CI] | U_bias | U_spread | p90 |
|---|---|---|---|---|---|
| braking (19 states) | λ_s | 0.75 [0.56, 0.96] | 0.60 | 0.15 | 0.25 |
| | λ_n | 0.12 [0.07, 0.19] | 0.10 | 0.02 | 0.29 |
| | λ_ξ | 0.09 [0.04, 0.15] | 0.07 | 0.02 | 0.40 |
| | λ_vx | 0.09 [0.04, 0.15] | 0.07 | 0.02 | 0.20 |
| corner (24 states) | λ_s | 0.20 [0.10, 0.35] | 0.17 | 0.03 | 1.06 |
| | λ_n | 0.22 [0.13, 0.32] | 0.18 | 0.04 | 0.50 |
| | λ_ξ | 0.19 [0.11, 0.27] | 0.15 | 0.04 | 0.87 |
| | λ_vx | 1.03 [0.84, 1.25] | 0.82 | 0.22 | 0.70 |

**Per-key shares.** These place every continuation beyond the furthest preview of all the key's
T states (K2b's point s2, or further). A state earlier on the key then sees less of the road
before it, so these shares are lower bounds.

| Phase | λ_s | λ_n | λ_ξ | λ_vx |
|---|---|---|---|---|
| braking (35 rows, 17 keys): U [CI] | 0.53 [0.41, 0.64] | 0.09 [0.05, 0.13] | 0.06 [0.03, 0.09] | 0.05 [0.03, 0.07] |
| braking: p90 (K2b's slow corner) | 0.19 (0.25) | 0.33 (0.23) | 0.32 (0.24) | 0.15 (0.13) |
| corner (61 rows, 21 keys): U [CI] | 0.16 [0.07, 0.29] | 0.17 [0.10, 0.24] | 0.14 [0.09, 0.20] | 0.97 [0.81, 1.14] |
| corner: p90 (K2b's slow corner) | 0.67 (0.37) | 0.39 (0.23) | 0.62 (0.30) | 0.66 (0.36) |

- **Larger than K2b's slow corner.** On corner rows, realistic continuations move λ about twice
  as much as K2b's slow second corner did, on every K_P component. On braking rows they move it
  about as much.
- **Approach and exit, per key, for comparison.** On approach (56 rows), U is 1.2 for λ_s
  (whose approach variance is tiny), 0.15 for λ_n, 0.11 for λ_ξ and 0.05 for λ_vx. On exit
  (7 rows on 6 keys), U is 2.8, 0.13, 0.14 and 1.19.
- **The other λ components** (per state, U). Braking: λ_vy 0.10, λ_r 0.02, λ_δ 0.24 and λ_pedal
  0.05. Corner: 0.47, 0.11, 0.0003 and 0.14.
- **The bias part dominates.** U_bias is 70–83% of U on every K_P component and phase. So the
  continuations move λ mostly the same way: relative to the road the grammar generates, the
  stored straight is atypical. What remains once a state's preview is fixed is the
  continuations' spread, U_spread. It is 0.02–0.04 of the phase variance, except for braking λ_s
  (0.15) and corner λ_vx (0.22).
- **The state itself moves only a little.** At the per-state rows, the re-solved plan's own state
  moves by a median of 0.006–0.03 m in s and n and 0.01–0.04 m/s in vx. The largest moves are
  1.2 m and 0.9 m/s.

**For CS-5 and CS-14 (descriptive).** Amendment 2 left these p90s to BC-a0. They are per key, on
the interior rows CS-5 and CS-14 score.
- **Π.** |ΔΠ|/SD_T has p90s of 0.79 (Π^in), 1.02 (Π^out), 0.79 (Π^slip) and 1.02 (Π^μ) on
  approach rows. On exit rows they are 1.48, 1.78, 0.96 and 1.67. The Π targets therefore move by
  about one SD_T with the road beyond the preview, on the very rows CS-5 scored. This is the
  caveat CS-5 recorded ("Π's sensitivity to unseen road is unmeasured"), now measured: it is
  large.
- **σ.** |Δσ^ped|/SD_T has p90s of 0.57 on approach and 0.36 on exit. σ^steer does not move (p90
  about 10⁻¹⁰).
- **Where U is not read.** U is not reported where T's phase variance is near zero: Π^in and
  Π^slip on corner and exit rows (variance ≤ 3·10⁻⁵), and σ^steer outside the corner phase
  (≤ 10⁻²²). There U's ratio is meaningless, and the summary's values there are not results.

**Expected directions against the results** (amendment 11 item 5, point estimates, per state).
- **Ea0.1** ("in the corner phase, U is above 0.1 for s and vx"): **holds**. U is 0.20 for λ_s
  and 1.03 for λ_vx. It also holds per key (0.16 and 0.97).
- **Ea0.2** ("in the braking phase, U is smaller than in the corner phase for every K_P
  component"): **does not hold**, 3 of 4. It holds for n, ξ and vx. λ_s is the exception: its
  braking U is 0.75 against 0.20 in the corner. Per key it is also 3 of 4 (0.53 against 0.16).
  λ_s varies little within the braking phase (its variance there is 1/60 of the corner's), so a
  modest shift (p90 0.25 SD_T) is a large share.
- **Ea0.3** ("the per-state shares exceed the per-key shares"): **holds**, 8 of 8.

**The route** (plan BC-a0, "What the outcomes mean"; a recommendation to the lead under decision
#6, not a decision).
- **Option (ii), the preview-matched average target, in all 8 cells.** No per-state U has its CI
  upper end at or below 0.1; the upper ends are 0.15–1.25. Option (i), pre-registering against
  the existing λ with a stated bound of at most 0.1, is open for no K_P component in braking or in
  the corner.
- **The closest cells** are braking λ_ξ and λ_vx (per state 0.09 [0.04, 0.15] for each). Per
  key, their upper ends are 0.09 and 0.07, at or under 0.1, but the per-key shares are only the
  lower bound.
- **Cost.** Option (ii) is costed in the plan at 2,298 states × M draws at 60–75 s each. The
  per-state re-solves here took a median of 175 s at W = 400, on a shared machine. At W = 800
  they took 984 s, and 61% failed (20 of the 22 at IPOPT's iteration cap). At the W = 400 rate
  option (ii) needs about 2.3–2.9 times the plan's 150–380 CPU-h (175 s against 60–75 s), and
  more with W = 800 states. Its W = 800 states also need a solver that
  reaches full amplitude.
- **The pilot's limits.**
  - Its shares rest on 21 keys and 43 states (26 keys before the drops), with 1–2 states on each
    key and 3 continuations per state.
  - It has one W = 800 key.
  - The dropped, mostly tight-corner keys would probably raise the shares, not lower them.

### What BC-a0 says (exploratory, targets only)

- **On braking and corner rows, the existing λ is not a target BC-1 can pre-register against
  with a bound of 0.1.** Realistic road beyond the preview moves it by 9–103% of its phase
  variance (per state). On corner rows it moves λ about twice as much as K2b's slow second corner
  did. The biggest moves are λ_vx in the corner (U ≈ 1) and λ_s in braking (U ≈ 0.75).
- **Most of the movement is a common shift, not noise.** The stored straight continuation is
  atypical of the grammar's road. So a preview-matched average target would differ from the
  existing λ mostly by a systematic offset. Once the preview is matched, the spread that remains
  is small (0.02–0.04 of the phase variance), except for braking λ_s and corner λ_vx (0.15 and
  0.22).
- **The Π targets are also sensitive on approach and exit rows** (p90 about 0.8–1.8 SD_T). That
  qualifies CS-5's and W1's Π comparisons; σ^ped's is smaller (0.4–0.6).
- No network was computed or read. None of this changes a decision of `costate-prereg-v1`.

## Appendix A. Sources of the key numbers

Each `<item>` file is `outputs/costate/exploratory/<item>/summary.json` (local, stamped), and
each has a committed, byte-identical copy under [`costate-followups/`](costate-followups/). In
that folder the CS-9 stage records are named `cs-9-build.json`, `cs-9-k2b.json` and
`cs-9-networks.json`. The local `cs10-iii.json`, `cs11-n.json` and `cs13-n.json` are committed as
`cs-10-iii-summary.json`, `cs-11-n-summary.json` and `cs-13-n-summary.json`. Every file carries
its stamp under `provenance`, and its expected directions under `expected`. `<run>` is a network's
label, G1 to G3s2 for the 9 and the run path for the others, except in `cs-2`, which keys every
network, the 9 included, by its run path (the label is under `.label`).

| Quantity | File | JSON path |
|---|---|---|
| ξ tail: where it sits, flags, enrichment | `cs-1` | `networks.<run>.tail`, `.where` |
| ξ tail: critic against map, channel groups | `cs-1` | `networks.<run>.critic_vs_map`; `groups` |
| Frozen, Huber and rank calibrations; plants | `cs-1` | `networks.<run>.calibrations`; `plants`, `plant_interval` |
| The rule for later items | `cs-1` | `rule_for_later_items` |
| Heads' ρ and C1, winsorisation, floor, folds | `cs-2` | `networks.<run>.heads`, `.reliability`, `.components`, `.folds_mean` |
| C1 on D; accel share | `cs-2` | `networks.<run>.c1_on_d`; `networks.<run>.accel` |
| C3-int pairs (r, p, Holm p, null) | `cs-3` | `pairs.<component>~<measure>`; `descriptive_with_specialists` |
| R²_z, R²_MLP, ΔC1 | `cs-4` | `networks.<run>.components`, `.mean`, `.c1_stat` |
| Π decodability and layer gains; W1 on Π^in | `cs-5` | `networks.<run>.targets`; `networks.<run>.w1_pi_in`; `controls` |
| \|ΔΠ\|/SD_T (left to BC-a0) | `cs-5`; `bc-a0` | `delta_pi_sd_t`; `per_key.<phase>.targets` |
| C1 over the 16 snapshots; trends | `cs-6` | `networks.<run>.snapshots`; `trends.<stat,r2c_s,r2c_xi,spearman_s,spearman_xi>` |
| Critic-layer, head-unit and actor-h3 probes | `cs-7` | `networks.<run>.components`, `.mean_over_s_xi`; `controls` |
| C2-sw random-only null (G1's scaling, own scaling) | `cs-8` | `a_b.null_g1`, `a_b.null_own` |
| C2-sw on approach and exit rows; on σ_res | `cs-8` | `a_b.networks.<run>.sigma`, `.res`, `.share_of_g1_hosts_at_or_above`; `sigma_res_share` |
| Decoy choice and C2 under decoy 2 | `cs-8` | `c.tried`, `c.chosen`, `c.thresholds`; `c.networks.<run>.chosen`, `.frozen`, `.ci_width` |
| N: keys, rows, descriptives | `cs-9-build.json` | `keys`, `n_rows`, `k0_dropped`, `descriptives` |
| K2b on N | `cs-9-k2b.json` | `rule`, `excluded_N`, `excluded_union`, `E9.4` |
| C1, C2 and C2-sw on N | `cs-9` | `agents.<run>.C1`, `.C2`, `["C2-sw"]`, `.residual_shares`, `.confirmatory_T` |
| C1-own on own-run states | `cs-10` | `networks.<run>.c1_own`, `.confirmatory`, `.winsorised`, `.held`, `.runs`, `.xi_rows_vs_T` |
| C1-own on N | `cs-10-iii-summary.json` | `agents.<run>.c1_own`, `.rows`, `.confirmatory` |
| Π^{in,own} on D | `cs-11` | `networks.<run>.common`, `.kept_H`, `.kept_W`, `.B` |
| Π^{in,own} on N | `cs-11-n-summary.json` | `agents.<run>.common` |
| C2 and C2-sw beyond the 9 | `cs-12` | `networks.<run>.C2`, `["C2-sw"]`; `decision_networks_cs8` |
| C2-pairs on N | `cs-13-n-summary.json` | `networks.<run>`; `pair_tolerances` |
| q_c's window rule and stop rule | `cs-14` | `by_phase`, `rows_pass`, `stop_rule`, `decision` |
| FWL, actor h1 and h2, C2 on D | `cs-15` | `networks.<run>.h3`, `.h1`, `.h2`, `.D`, `.confirmatory_c2` |
| BC-a0 per-state and per-key shares, p90 | `bc-a0` | `per_state.<phase>.targets`, `per_key.<phase>.targets`; `k2b_frozen_p90` |
| BC-a0 failures, route, cost | `bc-a0` | `failed`, `route`, `unit_seconds` |
| Confirmatory numbers quoted for comparison | `outputs/costate/analyse/decisions.json` | as in [`costate.md`](costate.md) Appendix A |

## Appendix B. The record

Each amendment was committed on its own, before any number it governs. Amendments 4–6 and 12–14
followed stopped or development runs, and each records what was seen.

| Commit | What |
|---|---|
| `eaf1e80` | The plan (CS-0), before any computation |
| `e1bd601` | The exploratory scaffold |
| `320ef22` → `70979ee` → `0f7f616` | Amendment 1; CS-1 code and tests; CS-1 results |
| `f95456f` → `b62ead3` → `87bb607` | Amendment 2; CS-2 to CS-5 code and tests; results |
| `c9b79db` → `705c5c2` | Amendment 3; CS-10 (ii) and CS-11 on D code and tests |
| `60fe838`, `1c47451`, `d06f28e` (with `6b3ffe2`, `e175355`, `b340a50`) → `fb5f704` | Amendments 4–6 (CS-11's wiring tolerances); CS-10 and CS-11 results |
| `3d41566`, `79c19d6` → `2056a3d` → `d562923` | Amendments 7 and 8; CS-7 and CS-8 code and tests; results |
| `4541068` → `6563b28`, `950918c` → `b737815` | Amendment 9; CS-9 code, tests and the networks-stage fix; results |
| `5cf5d2d` → `937b712` → `df7a806` | Amendment 10; CS-6, CS-12, CS-14 and CS-15 code and tests; results |
| `5491194` → `296bcb3`; `f0c8d7c`, `f21afaf` (with `f478ebd`), `bf80190` (with `8f2ad68`) → `bf9b9a1` | Amendments 11–14; BC-a0 code, tests, logs and unit cache; results |

The canonical runs took about 6.7 h of wall time for the CS items and 10.6 h for BC-a0, on the
Mac, with pools of 4 workers × 2 threads, niced, beside another workflow. That excludes
development runs and the runs that stopped and wrote nothing. The plan's estimate was 15–20 h.
Nothing was pushed.
