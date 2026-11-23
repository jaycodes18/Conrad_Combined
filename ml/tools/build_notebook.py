"""Write notebooks/Driven_ML_Pipeline.ipynb: the documented walk through every environment.

The notebook only calls the package's public API; all real code lives in ``driven_ml``. Rebuild it with
``python tools/check.py --notebooks``, which also executes it from a clean kernel.
"""

from __future__ import annotations

from _cells import code, md, save

NOTEBOOK_FILE = "Driven_ML_Pipeline.ipynb"

CELLS = [
    md(
        """
        # Driven ML pipeline

        Driven runs on a dashboard-mounted phone. MediaPipe Face Landmarker, a pre-trained network, turns
        each camera frame into eye, mouth and head-pose signals on the phone. Driven's own model is a small
        head on top: it scores each 60-second window for **drowsiness**, compared with the driver's own alert
        baseline, and drives a spoken in-car alert and texts to parents who opted in. **Eyes off the road**
        for more than 2 seconds is a fixed rule (NHTSA visual-manual guidelines, 78 FR 24818), not a model.

        Driven makes **no intoxication claim**. No licensed, labeled video of impaired drivers exists, nobody
        may drive impaired to create it (Tex. Penal Code 49.04), and advertising the claim without evidence
        violates FTC Act section 5. The model card says so in plain words, and the compliance gate fails any
        run whose user-facing text claims it.

        This notebook documents each environment: dev, train, validate, test, staging and production.
        """
    ),
    md(
        """
        ## 1. Colab setup

        In Colab, put the repository on Google Drive (unzipped, or as `DRIVEN-CONRAD-MLPRODUCT.zip` in
        `MyDrive`), and add `KAGGLE_USERNAME`, `KAGGLE_KEY` and `HF_TOKEN` under **Secrets** if you want the
        dataset search. Outside Colab this cell does nothing: install with
        `.venv/bin/python -m pip install -e "ml[dev]"` instead.
        """
    ),
    code(
        """
        import os
        import subprocess
        import sys
        import zipfile
        from pathlib import Path

        IN_COLAB = "google.colab" in sys.modules
        if IN_COLAB:
            from google.colab import drive, userdata

            drive.mount("/content/drive")
            repo = Path("/content/drive/MyDrive/DRIVEN-CONRAD-MLPRODUCT")
            archive = repo.with_suffix(".zip")
            if not repo.is_dir() and archive.is_file():
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(repo.parent)
            install = [sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo}/ml[discovery]"]
            subprocess.run(install, check=True)
            sys.path.insert(0, str(repo / "ml"))
            for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN"):
                try:
                    os.environ[name] = userdata.get(name)
                except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
                    print(f"Colab secret {name} not set; skipping it")
        print("Colab" if IN_COLAB else "Local run: nothing to set up")
        """
    ),
    md(
        """
        ## 2. Environments

        Each environment runs a fixed list of stages, so none can skip the compliance gate or promote from
        the wrong place. Dev runs everything except promotion, on synthetic people. Train, validate, test and
        staging run on real, licensed video. Production only promotes, and only with a named approver.
        """
    ),
    code(
        """
        import logging

        import pandas as pd

        from driven_ml.settings import STAGES_BY_ENVIRONMENT

        pd.set_option("display.max_colwidth", 160)
        pd.set_option("display.width", 200)
        logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s", force=True)

        pd.DataFrame(
            [{"environment": env.value, "stages": " -> ".join(stages)} for env, stages in STAGES_BY_ENVIRONMENT.items()]
        )
        """
    ),
    md(
        """
        ## 3. Compliance

        The gate runs first in every environment. It checks the legal flags in `product.json`, scans the
        landing page, app and backend (when they sit beside this pipeline) for banned claims and forbidden
        APIs, and fails the run on any finding. The rules are engineering guardrails derived from the team's
        legal review, not legal advice; a lawyer should review them before launch.
        """
    ),
    code(
        """
        from dataclasses import asdict

        from driven_ml.compliance import GATED_SETTINGS, LEGAL_RULES, enforce_compliance
        from driven_ml.product import load_product
        from driven_ml.settings import find_repo_root

        repo_root = find_repo_root()
        product = load_product(repo_root)
        enforce_compliance(product, repo_root)

        rules = pd.DataFrame([asdict(rule) for rule in LEGAL_RULES])
        gated = pd.DataFrame(
            [
                {**asdict(gate), "allowed_without_opinion": ", ".join(sorted(gate.allowed_without_opinion))}
                for gate in GATED_SETTINGS
            ]
        )
        display(rules)
        display(gated)
        """
    ),
    md(
        """
        The dataset registry records each dataset's license and which track may use it. The research track
        may use anything not marked `never_use`; the commercial track, the only one that can ship, needs a
        license that clearly allows commercial use (L9). Scraped and stock footage is never used (L10).
        """
    ),
    code(
        """
        from driven_ml.data import registry_table

        shown = ["key", "name", "license", "commercial_use", "research_track", "commercial_track", "intended_use"]
        registry_table()[shown]
        """
    ),
    md(
        """
        ## 4. Dataset discovery

        Discovery searches Hugging Face and Kaggle for drowsiness, eye-closure, yawn and driver-monitoring
        datasets, and sorts the hits by license: commercially usable first, unknown next, restricted last
        (`-nc`, `-nd` and research-only licenses). It writes `reports/discovered_datasets.csv` and never
        downloads data or edits the registry; a person reads each license and data card before adding a
        reviewed entry. It runs here only when Hugging Face or Kaggle credentials are present.
        """
    ),
    code(
        """
        from driven_ml.data.discovery import run_discovery
        from driven_ml.settings import RunConfig
        from driven_ml.workspace import Workspace

        has_credentials = bool(os.environ.get("HF_TOKEN") or os.environ.get("KAGGLE_KEY")) or (
            Path.home() / ".kaggle" / "kaggle.json"
        ).is_file()
        if has_credentials:
            discovery = run_discovery(Workspace(RunConfig.create("dev", run_discovery=True)).ensure())
            print(discovery.status_by_hub)
            display(pd.read_csv(discovery.report_path).head(20))
        else:
            print("Discovery skipped: no HF_TOKEN, KAGGLE_KEY or ~/.kaggle/kaggle.json")
        """
    ),
    md(
        """
        ## 5. Dev run

        Dev runs every stage except promotion on 30 synthetic people, each with their own eye size, blink
        habit and drowsiness visibility. It proves the whole pipeline works before any real video exists.
        """
    ),
    code(
        """
        from driven_ml import run_pipeline

        state = run_pipeline(RunConfig.create("dev", run_discovery=False))
        """
    ),
    md(
        """
        ## 6. Frames and windows (feature spec v4)

        Each frame holds 13 signals from the landmarks. Each 60-second window, every 15 seconds, summarizes
        them in 12 features, and every one of them is chosen so that **where the phone is mounted cannot move
        it**. A nearer camera makes every eye look more open, and a tilted mount shifts every head angle, so
        absolute levels such as mean eye opening or mean pitch are left out. Instead:

        - eye closure is measured against the eye's own open level in the window (its 90th-percentile EAR),
          and PERCLOS is the share of time the eye is at least 80% closed: the P80 measure of Wierwille et al.,
          NHTSA DOT HS 808 247 (1994), validated in Dinges and Grace, FHWA-MCRT-98-006 (1998);
        - blinks and microsleeps are runs of at least 50% closure;
        - head nods are dips below the window's own median pitch, and gaze-off is measured from the driver's
          own centre, not the camera's axis;
        - the spread of pitch and yaw, yawns (a ratio, so distance cancels) and face coverage complete it.

        Each later window also carries its `delta_` from the driver's baseline (section 7).
        """
    ),
    code(
        """
        frames = pd.read_parquet(state.workspace.frames_path)
        display(frames.head())

        windows = pd.read_parquet(state.workspace.windows_path)
        person = windows["subject"].iloc[0]
        columns = ["session", "t0", "perclos", "delta_perclos", "closure_mean", "delta_closure_mean", "head_nods"]
        windows.loc[windows["subject"] == person, columns].groupby("session").head(4)
        """
    ),
    md(
        """
        ## 7. Two baseline scopes

        The baseline is the driver's own alert face, like Face ID setup, and its scope depends on the data:

        - **person** (public datasets such as UTA-RLDD): the first two minutes of the person's alert video are
          the baseline for all their videos. Every video is a separate recording, so a camera change between
          videos can still pass for drowsiness, and this scope's numbers are reported but never trusted alone.
        - **session** (team recordings, `ml/RECORDING_PROTOCOL.md`): a 2-minute calibration clip and the drive
          are recorded in one mount and one sitting, as the phone does. The calibration windows are the
          baseline for that session's drive and never become training windows. Drive windows are labeled from
          a script written live: drowsy when they overlap an acted drowsiness, yawn or nod; alert when at
          least 30 s from every event; left out otherwise, including looking away, which is a fixed rule.

        Below, one synthetic team session shows the labels and the calibration baseline.
        """
    ),
    code(
        """
        from driven_ml.features.synthetic import simulate_team_session
        from driven_ml.windows import add_baseline_deltas, label_scripted_windows, slide_windows

        team_frames, team_script = simulate_team_session("team_recordings:T00", "s1", seed=3)
        team_session = "team_recordings:T00:s1"
        team_windows = label_scripted_windows(slide_windows(team_frames), team_script.assign(session=team_session))
        display(team_script)
        display(team_windows.groupby(["clip", "label"]).size().rename("windows").reset_index())
        scoped = add_baseline_deltas(team_windows)
        scoped[["clip", "baseline_scope", "t0", "label", "perclos", "delta_perclos"]].head(8)
        """
    ),
    md(
        """
        ## 8. Validation

        Every candidate is scored with 5-fold cross-validation grouped by person, so no number comes from a
        person the model trained on. The claim being tested is the personal baseline: the Driven models use
        the window features plus their deltas from the driver's baseline, B3 is the same logistic regression
        without the deltas, and B1 is a fixed PERCLOS threshold for everyone. If B3 matches Driven, the
        baseline is not earning its keep on this data, and that is a finding to report, not hide. Thresholds
        are chosen by the false-alert budget (0.2 per hour), and probabilities are temperature-calibrated on
        out-of-fold logits.
        """
    ),
    code(
        """
        validation = state.validation.to_frame()
        display(validation.round(3))
        display(state.validation.scope_frame().round(3))
        best = state.validation.best()
        print(f"Exported: {best.name} ({len(best.inputs)} inputs), threshold {best.threshold:.2f}")
        """
    ),
    md(
        """
        ## 9. Locked test

        Test people are chosen by a hash of their id, so nobody can reshuffle them, and only the test stage
        touches them. Every test run is logged, but the first result for a model hash is the one kept:
        rerunning cannot swap a bad number for a lucky one. The report breaks every metric down by baseline
        scope and gives the within-recording AUC over held-out team sessions (none in dev).
        """
    ),
    code(
        """
        report = state.test_report
        print(f"{report.people} test people, {report.windows} windows, prior runs of this model: {report.prior_runs}")
        overall = {"Driven": asdict(report.driven), "B1 fixed PERCLOS": asdict(report.fixed_threshold)}
        display(pd.DataFrame(overall).T.round(3))
        display(pd.DataFrame([asdict(score) for score in report.by_scope]).round(3))
        print(
            f"within-recording AUC {report.within_recording.auc:.3f} "
            f"over {report.within_recording.sessions} held-out team sessions"
        )
        """
    ),
    md(
        """
        ## 10. Staging

        Staging packages the model, its model card, the feature spec and the parity fixtures, replays scripted
        sessions exactly as the phone would score them, checks the unit economics at the measured false-alert
        rate, and writes a manifest. Every check must pass before anyone may promote. Three checks exist
        because of the camera confound (section 11):

        - **null control**: the pipeline must score at chance when the labels carry no drowsiness;
        - **within-recording AUC ≥ 0.80** on held-out team sessions, gated on real data and only reported in dev;
        - **replay** on people the model never trained on: `replay_only` people and locked test people. Real
          data needs at least 2 replay people and 6 acted drowsiness events. The dev scripted person is
          synthetic and outside training.

        The parity fixtures keep missing values as JSON `null`: 15 rows have 1 to 5 inputs removed and one row
        has none at all, so a phone port that skips `fill_values` fails parity.
        """
    ),
    code(
        """
        manifest = state.manifest
        display(pd.DataFrame([asdict(check) for check in manifest.checks]))
        print("staging", "PASSED" if manifest.passed else "FAILED")
        display(pd.DataFrame([asdict(result) for result in state.replay]))
        """
    ),
    code(
        """
        import json

        import numpy as np

        from driven_ml.modeling import ModelArtifact, predict_probability

        stage_dir = state.workspace.staging_dir(manifest.model_sha256[:12])
        staged = ModelArtifact.from_json(json.loads((stage_dir / "model.json").read_text()))
        parity = json.loads((stage_dir / "parity_fixtures.json").read_text())
        parity_inputs = np.array(parity["inputs"], dtype=float)
        expected = np.array(parity["expected"])
        error = np.max(np.abs(predict_probability(staged, parity_inputs) - expected))
        print(f"parity: {len(parity_inputs)} rows, {parity['rows_with_missing']} with missing values")
        print(f"max error {error:.1e} (tolerance {parity['tolerance']})")
        skipped = np.max(np.abs(predict_probability(staged, np.nan_to_num(parity_inputs)) - expected))
        print(f"a port that writes 0 instead of the fill values misses by {skipped:.3f}")
        print((stage_dir / "MODEL_CARD.md").read_text())
        """
    ),
    md(
        """
        ## 11. What would fool this model, and how we check

        **The camera, not the driver.** On real data every video has its own mount angle and distance. A model
        can learn "this is a different recording" instead of "this driver is drowsy". That is shortcut
        learning (Geirhos et al., "Shortcut learning in deep neural networks," *Nature Machine Intelligence*
        2, 665-673, 2020). In UTA-RLDD every video is a separate recording and the drowsy videos are drowsy
        from the start (Ghoddoosian, Alexander and Ross, CVPRW 2019), so its baseline always comes from a
        different recording than its drowsy data.

        **Defence 1: dev data contains the confound.** Every synthetic recording draws its own camera
        placement, which scales eye and mouth opening and offsets head pose and gaze.

        **Defence 2: camera-invariant features (spec v4).** Section 6.

        **Defence 3: the null-signal control.** Forty synthetic people whose drowsy sessions show no
        drowsiness at all go through the whole pipeline, three seeds. A sound pipeline scores at chance; the
        gate needs a mean AUC of at most 0.60 and every seed at most 0.65. This is a permutation-style check
        (Ojala and Garriga, "Permutation tests for studying classifier performance," *JMLR* 11, 1833-1863,
        2010). On feature spec v3 it scored AUC 0.516, 0.779 and 0.972 (mean 0.756): the confound, measured.
        On v4 it scores:
        """
    ),
    code(
        """
        null_control = pd.DataFrame(manifest.null_control["results"])
        display(null_control)
        print(f"mean AUC {manifest.null_control['mean_auc']:.3f}, passed: {manifest.null_control['passed']}")
        """
    ),
    md(
        """
        **Defence 4: accuracy inside one recording.** On real data the number that counts is the AUC between
        drowsy and alert windows of the *same* drive clip, pooled over held-out team sessions. No camera
        difference can inflate it, and a model cannot ship without it (at least 0.80).
        """
    ),
    md(
        """
        ## 12. Production

        Promotion is refused for this model, and the reasons say why: it was trained on synthetic data on the
        research track. The real path is:

        1. Record consented team sessions per `ml/RECORDING_PROTOCOL.md`: a calibration clip and a scripted
           drive in one mount, in a parked car with acted cues only (L11), with `consent.csv` filled in for
           every person, plus a guardian's consent for minors (L5). Mark at least two people `replay_only`.
        2. Train on the commercial track: `--env train --track commercial`.
        3. Run `test`, then `staging`, until every staging check passes.
        4. Promote with a named approver: `--env production --track commercial --approved-by "<name>"`.
        """
    ),
    code(
        """
        from driven_ml.errors import ComplianceError
        from driven_ml.release import promote

        attempt = RunConfig.create("dev", run_discovery=False, approved_by="Notebook reviewer")
        try:
            promote(attempt, state.workspace, state.product)
        except ComplianceError as refused:
            print(refused)
        """
    ),
    md(
        """
        ## 13. Running each environment

        From the repository, or from Colab after the setup cell (`!cd <repo>/ml && python -m driven_ml ...`):

        | Environment | Command |
        |---|---|
        | dev | `python -m driven_ml --env dev --no-discovery` |
        | train | `python -m driven_ml --env train --track commercial` (add `--reuse-features` to skip re-extraction) |
        | validate | `python -m driven_ml --env validate --track commercial` |
        | test | `python -m driven_ml --env test --track commercial` |
        | staging | `python -m driven_ml --env staging --track commercial` |
        | production | `python -m driven_ml --env production --track commercial --approved-by "<name>"` |

        Public video goes in `driven_work/raw/<dataset>/<person>/{0,5,10}.mp4` (UTA-RLDD naming: 0 alert,
        5 low vigilance, 10 drowsy); team sessions go in `driven_work/recordings/<person>/<session>/` with
        `calibration.mp4`, `drive.mp4` and `script.csv`. Add
        `--write-app --app-dir <app>` to copy fixtures or a released model into the app.
        """
    ),
]


if __name__ == "__main__":
    print(save(CELLS, NOTEBOOK_FILE))
