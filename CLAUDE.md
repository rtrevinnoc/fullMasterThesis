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
  simulation/         litho_sim.py (full 16-body MuJoCo model, StepperController,
                      quintic Profile4thOrder), case1_simulation.py (reduced
                      2-body-per-chain ODE replication of Al-Rawashdeh 2022)
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
  `python3 profiles.py`. Note: `litho_sim.Profile4thOrder` is a quintic
  smoothstep whose `s_max` is INERT — Exps 3 & 5 used it; Exps 6 & 7 use the
  exact generators through the `profile_factory` hook.
- `litho_control.py` — controller ladder on the full MuJoCo model:
  case1 (LS PID only) → case2 (+ lag feedforward, lag = lv·v + la·a fitted by
  iterative least squares; on the corrected plant lv ≈ 0, la ≈ −30 µm/(m/s²))
  → case3a (+ fine-stage PI, kp=40 ki=15000) → case3b (+ fine-stage **HIGS**,
  kp=40 ωi=2π·2500 ωh=2π·1000 kh=1, anti-windup). Higher gains destabilize
  against the flexible-chain modes (kp≈80 or ki≈25000 oscillates).
- **Plant corrections (2026-07-05, matching Al-Rawashdeh 2022)**: LS y/x
  joints are frictionless — kv acts on tracking-error velocity via
  `<velocity>` actuators (ctrl 18–21) commanded with the reference velocity,
  not as joint damping (which added a 10 mm/(m/s) drag lag); actuator
  reactions on the base are cancelled per step (`make_reaction_canceller`,
  ideal balance mass, paper's external wrenches); error sampling pairs
  body state and reference at the same instant (post-step pairing added a
  160 µm skew at cruise). Fine-stage servo keeps joint damping (authority
  pole kp/kv = 100 rad/s — structural; raising it destabilizes).
- `costfn.py` — corrected yield cost J = (1−P_none) + P_out-of-domain
  (+ β·throughput penalty). Plausible classes: none/Scratch/Edge-Loc/Random.
- `exp_controllers.py`, `exp_transition.py`, `exp_order_comparison.py`,
  `exp_pso.py` — the experiment drivers; `run_headless.py` — glfw stub runner.
- Archived `*.log` / `*.csv` files are the runs cited in the thesis. Don't
  regenerate without reason; if regenerated, update the chapter numbers.

## Build commands

```bash
# Presentation (biblatex, pdflatex is fine)
cd presentation && latexmk -pdf -interaction=nonstopmode presentation.tex

# Full thesis — MUST use XeLaTeX (fontspec); plain -pdf fails
cd fullThesis && latexmk -xelatex -interaction=nonstopmode TesisPrincipal.tex
```

If `latexmk` says "up-to-date" but you just edited, add `-g` to force.

## Running Python experiments

- venv: `~/maestria/venv` (python3 on PATH). Its `.pyc` caches are corrupted —
  always set `PYTHONPYCACHEPREFIX=<tmp dir>`; do not delete the caches unasked.
- `import glfw` aborts (libffi trampoline) in shells without window-server
  access, and `litho_sim` imports `mujoco.viewer` at module level: stub
  `glfw`/`mujoco.viewer` in `sys.modules` first (see `run_headless.py`).
- Set `MPLBACKEND=Agg`; LSTM scripts run with cwd = repo root.

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
- **Never invent numbers**: every figure/metric in the experiments chapter must
  trace to a script output archived in `research/experiments/`.
- **Figure sizing (deck)**: TikZ uses `[scale=N]`, never `\resizebox` around
  labeled TikZ. Slide body text floor: `\footnotesize`.
- **No emojis** in any file.

## Methodological anchors & experiment status (as of 2026-07-05, corrected plant)

- **Hypothesis** (intro §1.2) is split into H1 sensitivity / H2 expressiveness /
  H3 optimality. Status: H1 confirmed (monotone per bound; sign REVERSES for v,
  see sweep); H2 spatial+reproducibility confirmed, classification half limited
  by the classifier; H3 exercised, dominance pending a cost-calibration pass
  (E4's optimum is now a 3-failing-die wafer — the quirk *selects*, not just
  flattens).
- **Validation criterion**: simulated maps should activate only scanner-plausible
  WM-811K classes (*none*, *scratch*, *edge-local*, *random*). "Even
  distribution across 9 classes" is confused softmax, not learning.
- **CNN**: 96.88 % hold-out accuracy, macro-F1 0.77, **Scratch recall = 0**
  (imbalance; fix deferred). Calibration quirk: a map with a few failing dies
  scores more "none" than a perfectly clean map (real WM-811K 'none' wafers
  carry scattered failures); on the corrected plant this inverts the E4
  ranking → planned fix: calibrate P_none + die-count regularizer.
- **Case 1 result** (reduced ODE model, unaffected by the plant corrections):
  no fine stages → misses MA/MSD spec by ~3 orders (MA ±985 nm vs [−1.25,+2]
  nm). Spring lags are **µm**, not nm (docstring in case1_simulation.py has a
  units slip). The full MuJoCo model has a constant ~7.7 mm geometric offset
  in absolute body positions → use offset-corrected errors; MSD is
  offset-immune, MA is not.
- **Controller ladder** (corrected plant): cruise MA 738 → 181 (FF) → 2.50/2.14
  µm, MSD 33.9 → 11.0 → 0.22/0.27 µm (fine PI / HIGS); PI ≈ HIGS within 25 %.
  Residual gap to nm specs = generic fine stage (authority pole 100 rad/s) vs
  the paper's 2 kHz purpose-built stage and K=1e12 actuator mounts —
  experiments use baseline-referenced thresholds, stated in the chapter.
- **Transition experiment** (central premise, corrected plant): first failures
  at row-change dies with 15.1× enrichment (all 8 onset failures are
  row-change dies; Edge-Loc pattern from the trajectory alone), onset plateau
  180–160 ms, collapse 160→150 ms (107/241, scan-order 0.54 band, CNN Loc),
  near-total by 120 ms. Threshold 2× median of 200 ms run = 1.37 µm.
- **Order × controller** (corrected plant): under bare PID cruise MA 0.75–1.0
  mm all orders (transient tail); cruise MSD improves 4.4× with order. Under
  FF+HIGS, 2nd→4th order is **38×** on cruise MSD (6.84→0.18 µm); tight snap
  is now the MOST accurate (46 nm MSD, 0.33 µm MA) but 30 % slower with
  no full-scan gain → interior optimum only once throughput enters the cost.
- **PSO E3/E4** (exact generator, t_step=150 ms, T_budget=15 s, corrected
  plant): E3 drives v to the search bound (1.5 m/s, clean, 14.8 s — the 1/v
  exposure-window effect); E4 best v=1.22 at 13.5 s with **3/49 failing**
  (J=0.026 < clean 0.043). Both dominate the baseline seed (4/49, 15.2 s).
- **Kinematic sweep** (Exp 3, corrected plant): baseline median MSD 152 µm
  (threshold); a and j act monotonically (87/52/45 µm), but **halving v is
  worse** (160 µm, 44/49 fail) — exposure window ∝ 1/v integrates more
  transient. s inert in the quintic generator (unchanged finding).
- **Gap framing**: inverse problem (fingerprint → cause) is established
  industry practice (Lam 2015); this thesis fills the **forward direction**
  (trajectory parameters → predicted defect pattern → cost).
- **LSTM/GRU surrogate**: secondary study only (§5.9 + critique). Retrained on
  the corrected plant (predictions 31.5/2.28/1.09 mm MA — ranking OK, absolute
  scale still carries the geometric offset); grid search still finds no
  feasible sub-nm candidate for Case 1. Offset-corrected retraining remains
  future work.
- **Open items**: CNN rebalancing/calibration → rerun E-scenarios for H3;
  full E1 yield scenario; high-bandwidth fine-stage model (closes the last
  3 orders to nm specs); HIGS study in overshoot-constrained settling;
  **presentation/ deck still cites the old drag-plant numbers** (out of scope
  of the 2026-07-05 correction pass; update before Seminario I).

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
