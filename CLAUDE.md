# Tesis — Lithography Trajectory Control

Master's thesis (FIME UANL) on yield-aware optimization of 4th-order (snap-bounded) trajectories for step-and-scan photolithography scanners, using a CNN trained on WM-811K as a defect-probability cost function inside a PSO loop.

**Scope is simulation-only**: no scanner access — MuJoCo + CNN form the simulation test bed. Lack of hardware is intended scope, not a risk.

## Directory map

```
presentation/         15-min Seminario I deck (Spanish)
  presentation.tex    main beamer source (sdqbeamer + KIT theme)
  presentation.bib    biblatex authoryear
  pictures/           sim outputs, CNN confusion matrix, WM-811K samples
  logos/              UANL + FIME logos, title-image banner

fullThesis/           full thesis document (LaTeX, ENGLISH — user decision 2026-07)
  TesisPrincipal.tex  master file (chapter order: intro, model, controlObjectives,
                      methodology, experiments, conclusion, apendice1)
  bibliografia.bib    natbib/amsplain bibliography (keep in sync with presentation.bib)
  pictures/           figures copied from research/ outputs

research/
  simulation/         litho_sim.py (full 16-body MuJoCo model — TWO-STAGE,
                      LS+SS per reticle/wafer chain, die/mask end-effector;
                      StepperController now builds the EXACT profiles.Profile4;
                      the i=6 "Stage" body — `w_stage`/`r_stage` — is PASSIVE
                      Kelvin-Voigt (K_ST_F/C_ST_F), only LS+SS are actuated
                      per chain, fixed 2026-08-11, see "m6 fix" below; its
                      mass/inertia and `K8_R_ROT` were themselves wrong until
                      the same-day "stage-mass fix", see below). NO reduced
                      model of any kind is used anywhere in this project —
                      `case1_simulation.py` (a reduced 2-body-per-chain ODE)
                      was deleted 2026-08-11, see "Case 1 correction" below.
  nn/cnn/             CNN trained on WM-811K (wafer_cnn.pth, train/evaluate,
                      test_pipeline*.py sim→map→CNN bridge; needs LSWMD.pkl 2.1 GB)
  nn/lstm/            GRU surrogate (secondary study; run from repo root — uses
                      paths relative to tesis/)
  experiments/        thesis-experiment drivers + archived logs/CSVs (see below)
  drafts/             thesis proposal (hypothesis, O1–O6, scenarios E1–E4)
  reference/          source papers

thesisClass/          previous deck version + coursework deliverables
PlotNeuralNet/        third-party LaTeX library for NN architecture diagrams
```

## research/experiments/ (reproduces every number in the experiments chapter)

- `profiles.py` — **exact** motion-profile generators: Profile2 (accel-bounded
  trapezoid), Profile3 (7-seg jerk-bounded), Profile4 (15-seg snap-bounded,
  Lambrechts feasibility cascade + cruise-velocity bisection). Self-test via
  `python3 profiles.py`. **These are now the ONLY generators used** — the
  quintic `litho_sim.Profile4thOrder` (whose `s_max` was inert) is dead code;
  `StepperController.__init__` builds `profiles.Profile4` directly (litho_sim
  imports `profiles` from ../experiments), so `s_max` is a live bound everywhere.
- `litho_control.py` — the FF / PI / HIGS controller comparison on the full
  MuJoCo model (config names kept as case1..case3b for compatibility):
  case1 (LS position tracking only) → case2 (+ lag feedforward, lag = lv·v +
  la·a fitted by iterative least squares) → case3a (+ PI **on the short-stroke
  actuator directly**, kp=16 ki=480) → case3b (+ **HIGS**, kp=16 ωi=2π·200
  ωh=2π·80 kh=1, anti-windup). Gains retuned 2026-08-11 (stage-mass fix,
  below) — the pre-fix grid (kp up to 56) no longer brackets the stability
  boundary; see `tune_fine.py` / `tune_fine_poststagefix.log`. The thesis
  compares **FF (case2) vs PI (case3a) vs HIGS (case3b)**: at their retuned
  stability-boundary gains PI and HIGS land within a few percent of each
  other, **HIGS now marginally ahead** (reversed from the pre-stage-mass-fix
  ordering — see "stage-mass fix" below for why). Higher gains destabilize
  against the flexible-chain modes.
- Errors are measured on the **die/mask end-effector** (`wafer`/`mask`, App. B
  m8) across K8/C8; `e_syn = e_wafer − α·e_mask`. Die pass/fail is the
  **exposure-window (cruise, middle-third) MSD**, NOT the full-scan max (which
  is transient-dominated) — this matters when judging against an absolute spec.
- **Plant model (matches Al-Rawashdeh 2022 Appendix B)**: LS translational
  joints frictionless — kv acts on tracking-error velocity via `<velocity>`
  actuators commanded with the reference velocity (not joint damping, which
  added a 10 mm/(m/s) drag lag); LS actuator reactions on the base cancelled
  per step (`make_reaction_canceller`, ideal balance mass); error sampling
  pairs body state and reference at the same instant. The generic fine loop's
  authority pole kp/kv = 100 rad/s is a numerical-stability ceiling — raising
  KP_ACT (stiffer mounts) OR lowering KV_ACT_TRANS (higher bandwidth) both
  make tracking WORSE (tested), so nm accuracy comes from the trajectory, not
  the servo.
- `costfn.py` — yield cost J_map = (1−P_none) + P_out-of-domain, plus a
  throughput term. **Read as a pattern-acceptability gate**: (1−P_none) is the
  yield loss to be traded for throughput; P_ood is the wall that stops
  over-aggression once the wafer saturates to an implausible (Center/…)
  morphology. Plausible classes: none/Scratch/Edge-Loc/Random. `exp_pso`
  uses a throughput REWARD (γ·T_wafer/T_ref, γ=1) so the trade is active.
- `exp_controllers.py` (FF/PI/HIGS ladder), `exp_case1_full.py` (Experiment 1,
  the paper's real Case 1 — full topology, zero fine stages — run on the full
  model; regenerates the `case1_*.png` figures), `exp_transition.py`,
  `exp_transition_onset.py`, `exp_order_comparison.py`, `exp6_order_baselines.py`
  (49-die order-3/order-4 baseline classification for Table exp6_pso),
  `exp_pso.py` (E3 servo / E4 CNN), `driver_sweep.py`, `driver_sweep_spec.py`
  (snap-vs-spec sweep, Exp 3), `driver_perturb.py` (disturbance→CNN
  validation), `tune_fine.py` (PI/HIGS gain grid search) — the experiment
  drivers; `run_headless.py` — glfw stub runner.
- Archived `*.log` / `*.csv` files are the runs cited in the thesis. Old
  fine-stage runs are in `archive_finestage_20260810/`. Don't regenerate
  without reason; if regenerated, update the chapter numbers. **All map and time-series experiment scripts natively export `.csv` files alongside their plots to permanently cache raw metrics (like MSD and MA shifts) without requiring MuJoCo reruns.**

## research/experiments_alrawashdeh/ (Parallel Al-Rawashdeh State-Space Suite)

- `rawashdeh_ss_sim.py` — Exact 16-body 96-state linear state-space plant ($\dot{x} = A x + B u$) matching Al-Rawashdeh et al. (2022) Appendix B equations and parameters directly, including the near-rigid $K_{\text{act}} = 10^{12}\ \text{N/m}$ and $C_{\text{act}} = 10^{10}\ \text{N s/m}$ actuated joint stiffnesses. Discretized via exact block matrix exponential scaling-and-squaring ($A_d, B_d$) for unconditional numerical stability.
- `rawashdeh_control.py` — Control ladder implementation (Case 1, Case 2 FF, Case 3a PI, Case 3b HIGS) operating directly on `StateSpaceLithoPlant`.
- Complete parallel experiment suite (`exp_case1_full.py`, `exp_controllers.py`, `exp_order_comparison.py`, `driver_sweep_spec.py`, `driver_sweep.py`, `exp_transition.py`, `exp_transition_onset.py`, `exp_pso.py`, `tune_fine.py`, `exp6_order_baselines.py`, `driver_perturb.py`) configured for the exact state-space plant. Allows direct numerical verification against Al-Rawashdeh et al. (2022) graphs alongside the primary MuJoCo simulation test bed.


## Build commands

```bash
# Presentation (biblatex, pdflatex is fine)
cd presentation && latexmk -pdf -interaction=nonstopmode presentation.tex

# Full thesis — MUST use XeLaTeX (fontspec); plain -pdf fails
cd fullThesis && latexmk -xelatex -interaction=nonstopmode TesisPrincipal.tex
```

If `latexmk` says "up-to-date" but you just edited, add `-g` to force.

## Running Python experiments

- **venv: local environment lives at `./venv`** (ignored by git). Rebuild if needed (`python3 -m venv venv && source venv/bin/activate`) with pyenv 3.10 and `pip install mujoco numpy matplotlib torch opencv-python-headless`; the 2026-08 re-run used **mujoco 3.11, torch 2.13**. Always set `PYTHONPYCACHEPREFIX=<tmp dir>`.
- `import glfw` aborts (libffi trampoline) in shells without window-server
  access, and `litho_sim` imports `mujoco.viewer` at module level: stub
  `glfw`/`mujoco.viewer` in `sys.modules` first (see `run_headless.py`).
- Set `MPLBACKEND=Agg`; LSTM scripts run with cwd = repo root. Full-model
  experiments run headless from the repo root, e.g.
  `python3 research/experiments/exp_pso.py . e4 case3b 0`.

## Conventions

- **Language**: `fullThesis/` is **English** (user decision; overrides older
  Spanish rule). `presentation/` stays Spanish. Code comments English.
- **Audience**: control-theory / EE experts, **not** lithography experts.
  Introduce LS/SS, MA/MSD, overlay, fingerprint terms at first use.
- **Forbidden term**: "digital twin" in user-facing text — say *simulation* /
  *simulación*.
- **Title**: the deck is **Seminario I**, not "Tesis I".
- **Citations**: keys follow `LastName + Year + Tag` (e.g.
  `Butler2011_PositionControl`). Add new entries to **both** bib files.
  Deck uses biblatex authoryear; fullThesis uses natbib/amsplain.
- **Cross-references**: Spell out "Figure", "Table", "Section", "Chapter" completely in the text. Only use "Eq." for equations.
- **Never invent numbers**: every figure/metric in the experiments chapter must
  trace to a script output archived in `research/experiments/`.
- **Figure sizing & Formatting**: TikZ uses `[scale=N]`, never `\resizebox` around
  labeled TikZ. Slide body text floor: `\footnotesize`. Avoid redundant `matplotlib` titles (`fig.suptitle` or `ax.set_title`) if the LaTeX caption already describes the plot. Keep `bbox_inches="tight"` to minimize whitespace, and rely on LaTeX for figure separation spacing (which has been tightened in `TesisPrincipal.tex`).
- **Visuals**: Wafer fingerprints must use green (`#4caf50`) for passing and red (`#f44336`) for failing (supersedes old blue/yellow schema).
- **No emojis** in any file.

## Methodological anchors & experiment status (as of 2026-08-11, stage-mass-fix rebuild)

**Machine (user decision, supersedes the 2026-07-05 fine-stage setup):** the
paper's TWO-STAGE base machine — LS + SS per reticle/wafer chain, die/mask
end-effector, **7 bodies/chain, 16 total, 96 states**. NO fine-stage add-on.
Masses/stiffnesses = Al-Rawashdeh Appendix B (m1..m8). The exact `Profile4` is
used everywhere; old fine-stage runs archived in `archive_finestage_20260810/`.

- **m6 fix (2026-08-11)**: `litho_sim.py` was wiring THREE actuated bodies per
  chain (LS, SS, and the i=6 "Stage") when only LS+SS should be actuated —
  the Stage body had its own `<position>` actuator (kp=1e7) instead of the
  passive `K_ST_F`/`C_ST_F` Kelvin-Voigt coupling those constants were already
  defined for (confirmed against `build_litho.py`, an older fuller-DOF model
  that had this right). Fixed: `w_st_act`/`r_st_act` renamed `w_stage`/
  `r_stage`, made passive; the Case-3a/3b fine loop now commands the
  short-stroke (SS) actuator directly, as `litho_control.py`'s own docstring
  and `FINE_CLIP_W/R` comments already assumed. This changes every archived
  MuJoCo number (Case1/Case2 baselines are numerically unchanged since they
  never drove the fine loop; Case3a/3b and everything downstream shifted).
  Fine-loop gains were retuned by grid search for the new non-collocated
  loop (SS is now 2 springs from the measured error, not ~1): PI kp=48
  ki=7200, HIGS kp=48 ωi=2π·1000 ωh=2π·400. All experiments were rerun and
  `fullThesis/experimentsChapter.tex` + `conclusion.tex` updated to match.
  `research/simulation/build_litho.py` intentionally left alone (unused,
  already had the correct passive-stage design).
- **stage-mass fix (2026-08-11, same day, found auditing the machine against
  a reference structural diagram)**: the m6 fix's passive `w_stage`/`r_stage`
  bodies were wired with `mass={M_ACT}`/`diaginertia={I_ACT_XX,ZZ}` (the
  0.5 kg actuator-placeholder values used for the LS/SS air-bearing bodies)
  instead of the already-defined-but-never-referenced `M_ST_F`=10.5 kg /
  `I_ST_F_XX,ZZ` (paper's own m7) — a leftover from the m6 rename that never
  updated the `<inertial>` tag. This moved the SS→Stage corner from the
  paper's 180 Hz to an artefactual 825 Hz. A second, independent, unrelated
  error found in the same audit: `K8_R_ROT` (mask end-effector rotational
  stiffness) was `1e8`, paper gives `1e9` (matches the already-correct
  `K8_W_ROT`); fixed alongside. Consequence: the Stage body is passive but
  PRESENT in every controller case (1/2/3a/3b), so this invalidated every
  archived full-model number, not just the fine-loop-driven ones — a bigger
  blast radius than the m6 fix. Fine-loop gains retuned (kp≈48→kp≈16 —
  the corrected 180 Hz resonance sits much closer to the fine loop's own
  operating range than the buggy 825 Hz one did, lowering the stability
  boundary); see `tune_fine.py` / `tune_fine_poststagefix.log`. All
  full-model experiments rerun; `fullThesis/methodologyChapter.tex`,
  `modelChapter.tex`, `experimentsChapter.tex`, and `conclusion.tex` updated.
- **Case 1 correction (2026-08-11)**: the thesis previously described
  Experiment 1 as running a "reduced two-body-per-chain model" said to
  reproduce the paper's Case 1 study. Verified against Al-Rawashdeh et al.
  (2022) directly: the paper's Case 1 is the FULL multi-body machine
  (LS→SS→Stage→end-effector per chain) with ZERO fine stages — vs. Case 2,
  which adds two fine stages — nothing to do with reducing body count; no
  reference paper supports a two-body collapse. `case1_simulation.py` (the
  reduced-ODE script) is DELETED along with its output PNGs. Experiment 1 is
  now produced by `exp_case1_full.py`, which runs the real full MuJoCo model
  with the fine-stage loop disabled (the `case1` config used everywhere
  else) — so Experiment 1's numbers are now, by construction, identical to
  the Case-1 row of the controller ladder (`exp_controllers.py`), which they
  were NOT before (the old reduced-model Experiment 1 reported ±985 nm MA /
  428 nm MSD while the same-chapter ladder table reported 817 µm MA /
  19.7 µm MSD for what was nominally the same "Case 1" — two different
  plants silently reported side by side under one label). This project uses
  NO reduced/toy model anywhere, by standing user directive — every number
  in the thesis traces to the real full MuJoCo simulation.
- **litho_control.py segmentation fix (2026-08-19, verified 2026-09-03)**:
  `_run`'s per-die segment close-off was checked *after* `mujoco.mj_step`,
  letting the new die's first SCANNING sample — taken at the
  stepping-to-scanning handoff, where Case 1's uncontrolled short-stroke lag
  is largest — leak into the tail of the PREVIOUS die's array as a spurious
  one-sample spike. Fixed by moving the check before `mj_step`. This landed
  after the 2026-08-11 stage-mass-fix reruns, so every archived `.log` in
  `research/experiments/` predated it until re-verified: reran
  `exp_case1_full.py`, `exp_controllers.py`, `exp_order_comparison.py`, and
  `exp_pso.py` (e3, e4 seed 0, e4 seed 1) on 2026-09-03. Result: **no
  headline number in the thesis or the CNCA paper changed** at reported
  precision — the spike only matters at a pass/fail boundary, and every
  cited number sits either deep in compliance or deep in saturation (e.g.
  Case 1 MA 954,955→955,046 nm, Case 3a MA 7,653→7,676 nm; PSO E3/E4 optima
  reproduced bit-for-bit on the free parameters). Archived logs/CSVs
  overwritten with the fresh, post-fix runs; no chapter or CNCA edits were
  needed. (Housekeeping note: running the two E4 seeds concurrently once
  clobbered the shared `pso_best_map_e4.png`/`pso_history_e4.csv` filenames,
  since the script names them by scenario, not seed — recovered by renaming
  the clobbered pair to `..._seed1` and rerunning seed 0 alone; PSO reruns
  for this project should be run one at a time per scenario going forward.)
- **THE headline unified result — the snap bound walks the exposure error from
  coarse to full spec, at a throughput cost** (snap-vs-spec sweep, Case 3b,
  full wafer, exposure-window MSD): s=10⁵ → 994 nm MSD / 7.52 µm MA / 49-of-49
  fail / CNN=Center / 17.7 s; s=10³ → 9.5 nm / 188 nm / still 49/49 failing
  (no longer compliant post-stage-mass-fix — was the compliant point before);
  s=100 → 1.0 nm / 40.5 nm / **0/49** / none / 48 s (**MSD spec met**); s=50
  → **MSD 0.5 nm, MA 28.7 nm** / 0/49 / none / 56 s (MSD spec met, **MA band
  NOT yet reached**, and now much further away than the pre-stage-mass-fix
  estimate of 3.6 nm). So the MSD spec IS reachable — via the *trajectory*,
  not the servo — and throughput is the price, now roughly 10x steeper
  (tighter snap needed, ~48s vs ~31s wafer cycle for compliance) than the
  earlier estimate. This is the thesis; the MA band needs a finer grid, now
  well below s=50.
- **Servo tuning CANNOT reach nm** (both tested, both worsen): raising KP_ACT
  (toward the paper's rigid 1e12 mounts) destabilizes; lowering KV_ACT_TRANS
  (raising the 100 rad/s authority pole) removes the damping that holds the
  flexible modes and diverges. The residual at a given trajectory is the
  180 Hz stage ring the 16 Hz loop can't *suppress* — but a tight snap never
  *excites* it. Excitation-limited, not bandwidth-limited. (This 180 Hz figure
  was, until the 2026-08-11 stage-mass fix, aspirational — the code was
  actually simulating an 825 Hz artefact; it is now verified true in-code.)
- **Hypothesis status**: H1 (sensitivity) CONFIRMED and now clean — every bound
  incl. snap is monotone, and the old velocity-reversal / snap-inert anomalies
  are gone (they were artefacts of the quintic generator). H2 (expressiveness)
  partial — spatial patterns emerge; classification limited by the CNN
  (Scratch recall 0). H3 (optimality) CONFIRMED in the posed form for E3
  (servo cost); E4 (CNN cost) is **NOT confirmed within the tested seed/
  iteration budget post-stage-mass-fix** — both the default seed and the
  previously-escaping alternate seed now get stuck on the Center-saturation
  plateau for the full 48-eval budget (the compliant snap region moved
  ~100x tighter, shrinking its share of the necessarily-widened search box).
  This is a real property of pattern-based costs as search landscapes, not a
  validation failure — but it is now a stronger, currently-unresolved open
  item than before (see Open items).
- **Thresholds = the real spec, MSD ≤ 7 nm exposure-window** (adopted as the
  die criterion across sweep/transition/PSO, user decision). Die pass/fail =
  cruise (middle-third) MSD, NOT full-scan max. MA reaches the [−1.25,+2] nm
  band at no snap value yet tested (finest tested, s=50, gives MA=28.7 nm);
  MSD ≤ 7 nm at s≈100 (was s≈10³ pre-stage-mass-fix).
- **CNN cost = pattern-acceptability gate** (not a die counter): flatness on
  near-clean maps is the FEATURE — it lets the optimizer trade acceptable-pattern
  yield for throughput; P_ood is the wall at the Center catastrophe. The SAME
  flatness is a BUG on the far side of the spec (deep in Center-saturation,
  J_map pinned at 2.0 regardless of degree of failure) — it gives gradient-free
  search nothing to climb, unlike the continuous servo cost (see PSO E3/E4).
  The die-count regularizer was tried and REJECTED (it fights the near-side idea).
- **Controllers — FF vs PI vs HIGS** (baseline snap, 4-die cruise): Case 1
  955 µm MA / 25.0 µm MSD → FF 99.8 / 6.54 → PI 7.65 / 0.851 → HIGS 7.48 /
  0.834. **HIGS now marginally BEATS PI again (~2%)** — reversed once more
  from the m6-fix-era finding (PI beat HIGS by ~2-3% then): the stage-mass
  fix moved the SS→Stage resonance from an artefactual 825 Hz down to the
  paper's real 180 Hz, close enough to the fine loop's own range that HIGS's
  phase-advantage edge on this cruise metric is restored. Still sustains
  ~42x the PI's equivalent ki/kp before destabilizing (same ratio — both
  loops' gains simply dropped ~3x together). Gap to the paper's *two-stage*
  tens-of-nm = servo fidelity (generic 1e7 servo vs rigid 1e12 mounts), not
  architecture.
- **PSO E3/E4** (Case 3b, t_step=0.15 s, throughput reward γ=1, 7 nm spec;
  search box's snap lower bound widened 2026-08-11 from 5e3 to 50 to reach
  the post-fix compliance region): E3 (servo cost, default seed 0) converges
  reliably — v=0.67, a=44.9, j=5000, s=170, 0/49, none, J_map=0.043, 37.6 s
  (confirms snap is still the dominant lever; compliance now costs ~half the
  velocity and ~70% more wafer cycle time than the pre-stage-mass-fix
  optimum). E4 (CNN cost) got STUCK on the Center plateau for the full
  48-eval budget under **both** seed 0 (best: 49/49 failing, v=0.82,
  J_map=2.000) **and** seed 1 (best: 49/49 failing, v=0.77 a=34.8,
  J_map=2.000) — seed 1 no longer escapes as it did pre-stage-mass-fix (it
  used to reach 4/49 failing by iteration 3). The acceptable region still
  exists and is correctly priced once found (E3 finds it on the identical,
  widened box) — but it is now a much smaller fraction of that box, so 48
  evals from either tested seed aren't enough to land inside it by chance.
  The seed-sensitivity problem is WORSE, not better, post-fix. Fix ideas
  (still not implemented): seed more particles away from the saturated
  baseline, or warm-start E4's swarm from E3's optimum.
- **Order × controller**: under bare PID cruise MSD improves 4.4× with order
  (unchanged by either fix — Case 1 never drives the fine loop, and this
  ratio held constant through both the m6 fix and the stage-mass fix); under
  HIGS 2nd→4th cruise MSD ≈ 29× (24.0→0.834 µm), tight-snap 312 nm (≈77×) —
  the most accurate but 30 % slower → interior optimum via throughput.
- **Transition (reported result): "the controller absorbs the settling, so
  snap is what matters."** Under Case 3b + a spec-compliant trajectory (now
  s=100, was s=10³ pre-stage-mass-fix — that value is no longer compliant
  post-fix, see stage-mass fix) the fine loop suppresses the entry transient
  so well that shrinking t_step to 40 ms leaves 0/241 over the 7 nm spec —
  no onset. The relative-onset version (2× settled-median threshold, S=1e5,
  unaffected by the choice above) reproduces the SAME row-change **15.06×
  enrichment** (identical number — verified genuinely plant-invariant, not a
  caching artefact) but the onset now jumps straight to the 8-die plateau at
  t_step=180 ms (the m6-fix-era intermediate 3-die/180ms step is gone) and
  holds through 160 ms, then a 103-die bulk failure read as *Loc* at 150 ms,
  then full *Center* saturation (232/241) at 120 ms. KEPT for spatially-
  resolved study (`exp_transition_onset.py`).
- **CNN**: 96.88 % hold-out accuracy, macro-F1 0.77, **Scratch recall = 0**
  (imbalance; retrain with class weights is the H2 lever, LSWMD.pkl present).
  Plant-independent — not re-run.
- **Case 1 / full model** (Exp 1, now the real full MuJoCo model — see "Case 1
  correction" above): no fine stages → misses spec by several orders of
  magnitude (max|MA| 955 µm, max MSD 25.0 µm — identical, by construction, to
  the Case-1 row of the controller ladder). MSD is offset-immune; MA is not
  — use offset-corrected errors.
- **Disturbance campaign** (Exp 2/3 CNN validation): vibration→Center (94/241
  failing, 40.3% confidence — was 93/241 at 55.5% pre-stage-mass-fix, same
  class, weaker confidence), drift→Donut (99.9 %), scratch→Loc; clean→none.
  Uses its own **MA-shift + 2× median MSD criterion (not the 7 nm spec)**.
  **CRITICAL**: The MA criterion here is the *relative MA shift* relative to the unperturbed baseline ($|\Delta \bar{e}_{\text{syn}}| > 100\text{ nm}$), NOT an absolute MA > 100 nm. Since Case 1's baseline absolute MA is 955 µm, an absolute 100 nm threshold would cause 100% of dies to fail instantly. The MA-shift metric is offset-immune.
- **Gap framing**: inverse problem (fingerprint → cause) is established
  industry practice (Lam 2015); this thesis fills the **forward direction**.
- **LSTM/GRU surrogate**: secondary; NOT re-run on the 2-stage plant — its
  numbers (31.5 mm MA etc.) are stale; the §-critique argument still holds.
- **Open items**: LSTM surrogate re-run; CNN Scratch rebalancing (H2);
  higher-fidelity fine-stage model to reach the paper's *two-stage* tens-of-nm
  at reasonable throughput; finer snap grid below s=50 to find the MA-band
  crossing point (now well below s=50 post-stage-mass-fix, not yet located);
  PSO E4 seed-sensitivity fix — **more urgent post-stage-mass-fix**, since
  neither tested seed now escapes the Center plateau (seed more particles
  away from the saturated baseline, or warm-start from E3's optimum);
  **presentation/ deck still on old (pre-m6-fix AND pre-stage-mass-fix)
  numbers**.

## Don'ts

- Don't reintroduce the dense rotated multi-body chain schematic on main
  slides — appendix backup frame only (`\detailedMultibodyFigure`).
- Don't add features, refactor figures, or rewrite chapters beyond what's
  asked. This is an iterative, advisor-driven project.
- Don't commit destructive git operations on shared branches.

## Persistent memory

Per-session context (user profile, presentation editing style, deferred
decisions, scope notes) lives in
`~/.claude/projects/-Users-rtrevinnoc-maestria-tesis/memory/MEMORY.md` and the
linked files there. Read those first when picking up work mid-stream.
