from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk


class ChessControlPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PyChess Control Panel")
        self.geometry("980x760")
        self.minsize(920, 680)

        self.procs: dict[str, subprocess.Popen] = {}
        self.log_queue: "queue.Queue[tuple[str, object, object]]" = queue.Queue()

        self._build_ui()
        self.after(120, self._drain_logs)

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=10, pady=10)

        top.grid_columnconfigure(0, weight=1)
        top.grid_rowconfigure(0, weight=0, minsize=390)
        top.grid_rowconfigure(1, weight=0)
        top.grid_rowconfigure(2, weight=1)

        self.notebook = ttk.Notebook(top, height=390)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tabs = {}
        for name in [
            "Self-Play Train",
            "Puzzle Train",
            "Build Puzzle Cache",
            "Generate Puzzles",
            "Play vs Model",
            "Model vs Model",
            "PGN Viewer",
            "Feedback Candidates",
        ]:
            frame = self._create_scrollable_tab(self.notebook, name)
            self.tabs[name] = frame

        self._build_selfplay_tab()
        self._build_puzzle_train_tab()
        self._build_cache_tab()
        self._build_generate_tab()
        self._build_play_tab()
        self._build_model_vs_model_tab()
        self._build_viewer_tab()
        self._build_feedback_candidates_tab()

        run_bar = ttk.Frame(top)
        run_bar.grid(row=1, column=0, sticky="ew", pady=(10, 6))
        ttk.Button(run_bar, text="Run Selected Tab", command=self.run_selected_tab).pack(
            side="left"
        )
        ttk.Button(run_bar, text="Stop Selected Tab", command=self.stop_process).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(run_bar, text="Stop All", command=self.stop_all_processes).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(run_bar, text="Clear Logs", command=self.clear_logs).pack(
            side="left", padx=(8, 0)
        )

        self.status_var = tk.StringVar(value="Idle (0 running)")
        ttk.Label(run_bar, textvariable=self.status_var).pack(side="right")

        log_frame = ttk.LabelFrame(top, text="Live Output")
        log_frame.grid(row=2, column=0, sticky="nsew")
        self.log = tk.Text(log_frame, wrap="word", height=20)
        self.log.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scroll.set)

    def _create_scrollable_tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        outer = ttk.Frame(notebook)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")

        inner = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(window_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)
        inner.bind("<MouseWheel>", _on_mousewheel)

        self.notebook.add(outer, text=title)
        return inner

    def _add_entry(
        self,
        parent: ttk.Frame,
        label: str,
        default: str,
        hint: str = "",
    ) -> tk.StringVar:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=(4, 2))
        ttk.Label(row, text=label, width=24).pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
        if hint:
            hint_lbl = ttk.Label(
                parent,
                text=f"Hint: {hint}",
                justify="left",
                foreground="#5f6368",
                wraplength=860,
            )
            hint_lbl.pack(fill="x", padx=10, pady=(0, 4), anchor="w")
        return var

    def _add_check(
        self,
        parent: ttk.Frame,
        label: str,
        default: bool = False,
        hint: str = "",
    ) -> tk.BooleanVar:
        var = tk.BooleanVar(value=default)
        ttk.Checkbutton(parent, text=label, variable=var).pack(anchor="w", padx=8, pady=(4, 2))
        if hint:
            hint_lbl = ttk.Label(
                parent,
                text=f"Hint: {hint}",
                justify="left",
                foreground="#5f6368",
                wraplength=860,
            )
            hint_lbl.pack(fill="x", padx=26, pady=(0, 4), anchor="w")
        return var

    def _build_selfplay_tab(self) -> None:
        f = self.tabs["Self-Play Train"]
        ttk.Label(
            f,
            text="Main RL self-play training. Recommended default is to prefer puzzle checkpoint at startup.",
            foreground="#37474f",
            wraplength=860,
            justify="left",
        ).pack(fill="x", padx=8, pady=(8, 6))
        self.sp_init_ckpt = self._add_entry(
            f,
            "Init checkpoint",
            "checkpoint_latest.pt",
            "Normal: checkpoint_latest.pt. Any .pt path. If file is older/weaker, training starts weaker.",
        )
        self.sp_puzzle_ckpt = self._add_entry(
            f,
            "Puzzle checkpoint",
            "checkpoint_puzzle_best.pt",
            "Normal: checkpoint_puzzle_best.pt. Best puzzle model for warm start. Better file usually improves early self-play.",
        )
        self.sp_prefer_puzzle = self._add_check(
            f,
            "Prefer puzzle init",
            True,
            "Normal: ON. ON = load puzzle checkpoint first if present. OFF = prioritize init checkpoint.",
        )
        self.sp_iters = self._add_entry(
            f,
            "Iterations",
            "5",
            "Number of train/eval loops.",
        )
        self.sp_games_per_iter = self._add_entry(
            f,
            "Games per iter",
            "40",
            "Self-play games generated each iteration.",
        )
        self.sp_batch_size = self._add_entry(
            f,
            "Batch size",
            "64",
            "Optimization batch size.",
        )
        self.sp_train_batches = self._add_entry(
            f,
            "Train batches",
            "32",
            "Number of gradient updates per iteration.",
        )
        self.sp_lr = self._add_entry(
            f,
            "Learning rate",
            "1e-4",
            "Safer self-play default. Lower helps avoid forgetting puzzle pretraining.",
        )
        self.sp_replay_dir = self._add_entry(
            f,
            "Replay dir",
            "replay",
            "Directory for replay shards. Use a fresh folder to avoid stale data mixing.",
        )
        self.sp_num_sims = self._add_entry(
            f,
            "Self-play sims",
            "400",
            "MCTS simulations per self-play move.",
        )
        self.sp_eval_num_sims = self._add_entry(
            f,
            "Eval sims",
            "50",
            "MCTS simulations used for evaluations and gating.",
        )
        self.sp_draw_penalty = self._add_entry(
            f,
            "Draw penalty",
            "0.0",
            "Target for draw-like outcomes. Keep 0.0 for unbiased baseline training.",
        )
        self.sp_stop_threefold = self._add_check(
            f,
            "Stop on threefold",
            False,
            "Stop game when threefold repetition can be claimed.",
        )
        self.sp_no_prog_limit = self._add_entry(
            f,
            "No-progress limit",
            "30",
            "Halfmove cutoff for no pawn/capture progress.",
        )
        self.sp_no_prog_penalty = self._add_entry(
            f,
            "No-progress penalty",
            "0.0",
            "Target used when no-progress stop triggers.",
        )
        self.sp_repeat2_penalty = self._add_entry(
            f,
            "Repeat2 penalty",
            "0.0",
            "Target used when repeat2 stop triggers.",
        )
        self.sp_stop_repeat2 = self._add_check(
            f,
            "Stop on repeat2",
            False,
            "End game on second repetition of a position (optional).",
        )
        self.sp_temp_floor = self._add_entry(
            f,
            "Temp floor",
            "0.1",
            "Minimum move temperature after opening exploration window.",
        )
        self.sp_use_mat_shape = self._add_check(
            f,
            "Use material shaping",
            False,
            "Usually OFF for stable baseline. ON adds heuristic reward shaping.",
        )
        self.sp_mat_scale = self._add_entry(
            f,
            "Material scale",
            "0.0",
            "Material shaping coefficient.",
        )
        self.sp_exch_scale = self._add_entry(
            f,
            "Exchange scale",
            "0.0",
            "Exchange shaping coefficient.",
        )
        self.sp_early_sims = self._add_entry(
            f,
            "Early sims",
            "0",
            "Opening sims (0 means use Self-play sims).",
        )
        self.sp_early_plies = self._add_entry(
            f,
            "Early plies",
            "16",
            "Opening plies that use Early sims.",
        )
        self.sp_late_sims = self._add_entry(
            f,
            "Late sims",
            "0",
            "Late-game sims (0 means use Self-play sims).",
        )
        self.sp_gate_games = self._add_entry(
            f,
            "Gate games",
            "30",
            "Candidate-vs-previous games per iteration (0 disables gating).",
        )
        self.sp_gate_min_score = self._add_entry(
            f,
            "Gate min score",
            "0.52",
            "Minimum gate score to accept a newly trained model.",
        )
        self.sp_feedback_jsonl = self._add_entry(
            f,
            "Feedback JSONL (optional)",
            "",
            "Path to labeled feedback rows: fen, bad_move, good_move. Leave empty to disable feedback loss.",
        )
        self.sp_feedback_weight = self._add_entry(
            f,
            "Feedback weight",
            "0.2",
            "0 disables feedback; typical mix is 0.1-0.3 with replay loss.",
        )
        self.sp_feedback_batch = self._add_entry(
            f,
            "Feedback batch size",
            "32",
            "Feedback samples per train batch when feedback is enabled.",
        )
        self.sp_feedback_margin = self._add_entry(
            f,
            "Feedback margin",
            "0.2",
            "Required policy-logit margin between good and bad move.",
        )
        self.sp_feedback_max_samples = self._add_entry(
            f,
            "Feedback max samples",
            "0",
            "0 = load all rows; set cap for quick experiments.",
        )
        self.sp_augment = self._add_check(
            f,
            "Color-flip augmentation",
            True,
            "50% chance to color-flip each sampled position during training.",
        )
        preset_row = ttk.Frame(f)
        preset_row.pack(fill="x", padx=8, pady=(8, 6))
        ttk.Button(
            preset_row,
            text="Apply Safe Preset",
            command=self._apply_safe_selfplay_preset,
        ).pack(side="left")
        ttk.Label(
            preset_row,
            text="Sets stable defaults to reduce self-play regression.",
            foreground="#5f6368",
        ).pack(side="left", padx=(8, 0))

    def _apply_safe_selfplay_preset(self) -> None:
        """Populate conservative self-play defaults for more stable fine-tuning."""
        self.sp_prefer_puzzle.set(True)
        self.sp_puzzle_ckpt.set("checkpoint_puzzle_best.pt")
        self.sp_lr.set("1e-4")
        self.sp_replay_dir.set("replay_fresh")
        self.sp_num_sims.set("400")
        self.sp_eval_num_sims.set("50")
        self.sp_draw_penalty.set("0.0")
        self.sp_stop_threefold.set(False)
        self.sp_no_prog_limit.set("30")
        self.sp_no_prog_penalty.set("0.0")
        self.sp_repeat2_penalty.set("0.0")
        self.sp_stop_repeat2.set(False)
        self.sp_temp_floor.set("0.1")
        self.sp_use_mat_shape.set(False)
        self.sp_mat_scale.set("0.0")
        self.sp_exch_scale.set("0.0")
        self.sp_early_sims.set("0")
        self.sp_early_plies.set("16")
        self.sp_late_sims.set("0")
        self.sp_gate_games.set("30")
        self.sp_gate_min_score.set("0.52")
        self.sp_feedback_jsonl.set("")
        self.sp_feedback_weight.set("0.2")
        self.sp_feedback_batch.set("32")
        self.sp_feedback_margin.set("0.2")
        self.sp_feedback_max_samples.set("0")
        self._append_log("[Preset] Applied Safe self-play preset.\n")

    def _build_puzzle_train_tab(self) -> None:
        f = self.tabs["Puzzle Train"]
        ttk.Label(
            f,
            text="Puzzle supervision (policy pretraining). Use cache_dir for large datasets; use puzzles_csv for small quick runs.",
            foreground="#37474f",
            wraplength=860,
            justify="left",
        ).pack(fill="x", padx=8, pady=(8, 6))
        self.pt_cache_dir = self._add_entry(
            f,
            "Cache dir",
            "puzzle_cache_lichess",
            "Normal: puzzle_cache_lichess. Path with NPZ shards. Faster for big datasets; empty means CSV mode.",
        )
        self.pt_puzzles_csv = self._add_entry(
            f,
            "Puzzles CSV (optional)",
            "",
            "Used only when cache dir is empty. Large CSV is slower than cache.",
        )
        self.pt_limit = self._add_entry(
            f,
            "Limit",
            "0",
            "Range: 0 or positive int. 0 = full dataset. Lower = faster experiments, weaker generalization.",
        )
        self.pt_batch = self._add_entry(
            f,
            "Batch size",
            "128",
            "Normal: 64-256. Higher = faster/steadier gradients but more RAM. Lower = noisier/slower.",
        )
        self.pt_epochs = self._add_entry(
            f,
            "Epochs",
            "10",
            "Normal: 5-30. Higher = more fitting, but too high can overfit.",
        )
        self.pt_lr = self._add_entry(
            f,
            "Learning rate",
            "3e-4",
            "Normal: 1e-4 to 1e-3. Higher learns faster but can diverge; lower is safer but slower.",
        )
        self.pt_workers = self._add_entry(
            f,
            "Data workers",
            "0",
            "CSV mode only. 0 = main process. Try 2-8 if CSV loading is slow.",
        )
        self.pt_torch_threads = self._add_entry(
            f,
            "Torch CPU threads",
            "0",
            "0 = backend default. On CPU-only training, try cpu_count-1.",
        )
        self.pt_auto_tune = self._add_check(
            f,
            "Auto tune CPU",
            False,
            "Benchmark several batch/thread settings, pick fastest, then train with it.",
        )
        self.pt_tune_batches = self._add_entry(
            f,
            "Tune batch sizes",
            "128,256,512",
            "Comma-separated list used when Auto tune CPU is ON.",
        )
        self.pt_tune_threads = self._add_entry(
            f,
            "Tune threads",
            "4,6,8",
            "Comma-separated thread counts used when Auto tune CPU is ON.",
        )
        self.pt_tune_steps = self._add_entry(
            f,
            "Tune max batches",
            "120",
            "Benchmark batches per config (higher = more accurate, slower startup).",
        )
        self.pt_tune_only = self._add_check(
            f,
            "Tune only",
            False,
            "ON runs benchmark and exits without full training.",
        )
        self.pt_val_ratio = self._add_entry(
            f,
            "Val ratio",
            "0.1",
            "Range: 0.05-0.2 typically. Higher = better evaluation confidence, fewer train samples.",
        )
        self.pt_seed = self._add_entry(
            f,
            "Seed",
            "42",
            "Any integer. Same seed => reproducible split/order.",
        )
        self.pt_label_smoothing = self._add_entry(
            f,
            "Label smoothing",
            "0.0",
            "Range: 0.0-0.1. Higher = less overconfidence, but too high can hurt top1.",
        )
        self.pt_overfit = self._add_entry(
            f,
            "Overfit debug n",
            "0",
            "0 disables. Use 64-512 to test pipeline sanity quickly (should overfit strongly).",
        )
        self.pt_pgn_dir = self._add_entry(
            f,
            "PGN dir",
            "puzzle_games",
            "Folder where best validation puzzle examples are exported as PGN.",
        )
        self.pt_pgn_max = self._add_entry(
            f,
            "PGN max games",
            "25",
            "Range: 0+. 0 disables PGN export. Higher writes more files/IO.",
        )
        self.pt_progress = self._add_entry(
            f,
            "Progress every batches",
            "100",
            "Range: 0+ int. 0 = quieter logs. Smaller = more frequent status output.",
        )
        self.pt_resume = self._add_entry(
            f,
            "Resume checkpoint",
            "checkpoint_puzzle_latest.pt",
            "Checkpoint path to continue puzzle training. Invalid path starts from fresh weights.",
        )

    def _build_cache_tab(self) -> None:
        f = self.tabs["Build Puzzle Cache"]
        ttk.Label(
            f,
            text="Convert puzzle CSV into fast NPZ shards. Recommended for large Lichess CSV.",
            foreground="#37474f",
            wraplength=860,
            justify="left",
        ).pack(fill="x", padx=8, pady=(8, 6))
        self.bc_csv = self._add_entry(
            f,
            "Puzzles CSV",
            "lichess_db_puzzle.csv",
            "Input CSV with columns like FEN/Moves. Large files are supported.",
        )
        self.bc_out = self._add_entry(
            f,
            "Out dir",
            "puzzle_cache_lichess",
            "Output cache folder. Reusing same folder may overwrite shard names.",
        )
        self.bc_limit = self._add_entry(
            f,
            "Limit",
            "0",
            "0 = no limit. Lower for quick smoke builds.",
        )
        self.bc_val_ratio = self._add_entry(
            f,
            "Val ratio",
            "0.1",
            "Typical: 0.1. Higher gives larger validation set and smaller train set.",
        )
        self.bc_seed = self._add_entry(
            f,
            "Seed",
            "42",
            "Any integer. Controls deterministic split hashing.",
        )
        self.bc_shard_size = self._add_entry(
            f,
            "Shard size",
            "2048",
            "Typical: 1024-4096. Larger = fewer files, slightly higher memory during write/read.",
        )
        self.bc_max_train = self._add_entry(
            f,
            "Max train shards",
            "0",
            "0 = unlimited. Positive cap limits dataset size and build time.",
        )
        self.bc_max_val = self._add_entry(
            f,
            "Max val shards",
            "0",
            "0 = unlimited. Positive cap keeps validation set bounded.",
        )
        self.bc_workers = self._add_entry(
            f,
            "Workers",
            "0",
            "0 = auto (cpu_count-1). Use 1 for old single-process behavior.",
        )
        self.bc_compress = self._add_check(
            f,
            "Use compressed NPZ",
            True,
            "ON = smaller files, slower build. OFF = larger files, much faster build.",
        )
        self.bc_clean = self._add_check(
            f,
            "Clean out dir before build",
            True,
            "ON removes old train/val shard files first to prevent stale data mixing with new cache builds.",
        )

    def _build_generate_tab(self) -> None:
        f = self.tabs["Generate Puzzles"]
        ttk.Label(
            f,
            text="Generate synthetic capture-focused puzzles. Useful for warm-up experiments, not a full substitute for real puzzles.",
            foreground="#37474f",
            wraplength=860,
            justify="left",
        ).pack(fill="x", padx=8, pady=(8, 6))
        self.gp_out = self._add_entry(
            f,
            "Out",
            "puzzles_synthetic.csv",
            "Output CSV filename/path.",
        )
        self.gp_count = self._add_entry(
            f,
            "Count",
            "256",
            "Range: 100-10000+ depending on time. Higher gives more diversity, takes longer to generate.",
        )
        self.gp_seed = self._add_entry(
            f,
            "Seed",
            "42",
            "Any integer for reproducible generation.",
        )

    def _build_play_tab(self) -> None:
        f = self.tabs["Play vs Model"]
        ttk.Label(
            f,
            text="Play against current model. Increase sims for stronger but slower opponent.",
            foreground="#37474f",
            wraplength=860,
            justify="left",
        ).pack(fill="x", padx=8, pady=(8, 6))
        self.pm_device = self._add_entry(
            f,
            "Device",
            "cpu",
            "Use cpu (safe) or cuda (faster if GPU is configured).",
        )
        self.pm_ckpt = self._add_entry(
            f,
            "Checkpoint",
            "checkpoint_latest.pt",
            "Model to play against. Stronger checkpoint usually gives tougher play.",
        )
        self.pm_sims = self._add_entry(
            f,
            "Num sims",
            "60",
            "Typical: 30-200. Higher = stronger move quality but slower response.",
        )
        self.pm_human_replay = self._add_entry(
            f,
            "Human replay dir",
            "replay_human",
            "Saved human-vs-model training shards destination.",
        )
        self.pm_human_pgn = self._add_entry(
            f,
            "Human PGN dir",
            "human_games",
            "Saved PGNs destination.",
        )
        self.pm_black = self._add_check(
            f,
            "Play as black",
            False,
            "ON makes model play first as White.",
        )
        self.pm_save_samples = self._add_check(
            f,
            "Save training samples",
            True,
            "ON stores gameplay as replay shards for potential later training.",
        )

    def _build_model_vs_model_tab(self) -> None:
        f = self.tabs["Model vs Model"]
        ttk.Label(
            f,
            text="Run model-vs-model games in real time with selectable checkpoints, board review, and bad-move marking.",
            foreground="#37474f",
            wraplength=860,
            justify="left",
        ).pack(fill="x", padx=8, pady=(8, 6))
        self.mm_device = self._add_entry(
            f,
            "Device",
            "cpu",
            "Use cpu (safe) or cuda (faster if GPU is configured).",
        )
        self.mm_sims = self._add_entry(
            f,
            "Num sims",
            "50",
            "MCTS simulations per model move.",
        )
        self.mm_delay = self._add_entry(
            f,
            "Move delay ms",
            "400",
            "Delay between model moves for real-time playback speed.",
        )
        self.mm_pgn_dir = self._add_entry(
            f,
            "PGN dir",
            "model_games",
            "Saved model-vs-model PGN destination.",
        )

    def _build_viewer_tab(self) -> None:
        f = self.tabs["PGN Viewer"]
        ttk.Label(
            f,
            text="Open PGN viewer directly, optionally loading a specific PGN or latest file from a folder.",
            foreground="#37474f",
            wraplength=860,
            justify="left",
        ).pack(fill="x", padx=8, pady=(8, 6))
        self.pv_pgn_path = self._add_entry(
            f,
            "PGN path (optional)",
            "",
            "If set, this file opens immediately at startup.",
        )
        self.pv_games_dir = self._add_entry(
            f,
            "Games dir",
            "model_games",
            "Folder used when Load latest is ON.",
        )
        self.pv_load_latest = self._add_check(
            f,
            "Load latest on start",
            True,
            "ON loads newest PGN in games dir when no specific path is provided.",
        )

    def _build_feedback_candidates_tab(self) -> None:
        f = self.tabs["Feedback Candidates"]
        ttk.Label(
            f,
            text="Generate candidate bad-move rows from many PGNs for faster manual labeling.",
            foreground="#37474f",
            wraplength=860,
            justify="left",
        ).pack(fill="x", padx=8, pady=(8, 6))
        self.fc_pgn_glob = self._add_entry(
            f,
            "PGN glob",
            "model_games/*.pgn",
            "Glob pattern for source PGNs. Use recursive toggle for ** patterns.",
        )
        self.fc_recursive = self._add_check(
            f,
            "Recursive glob",
            False,
            "Enable recursive glob expansion for patterns like model_games/**/*.pgn.",
        )
        self.fc_out = self._add_entry(
            f,
            "Out JSONL",
            "feedback_candidates.jsonl",
            "Output file for candidate rows.",
        )
        self.fc_max_games = self._add_entry(
            f,
            "Max games",
            "0",
            "0 = all games matched by glob.",
        )
        self.fc_max_plies = self._add_entry(
            f,
            "Max plies/game",
            "0",
            "0 = include all plies per game.",
        )
        self.fc_min_ply = self._add_entry(
            f,
            "Min ply",
            "1",
            "Only include plies at or beyond this 1-based index.",
        )
        self.fc_side = self._add_entry(
            f,
            "Side (both|white|black)",
            "both",
            "Filter candidate rows by moving side.",
        )
        self.fc_max_legal = self._add_entry(
            f,
            "Max legal move hints",
            "20",
            "Number of legal UCI hints added per candidate row (0 disables).",
        )

    def run_selected_tab(self) -> None:
        base = [sys.executable, "-u", "chess_interface.py"]
        tab_name = self.notebook.tab(self.notebook.select(), "text")
        if tab_name == "Self-Play Train":
            cmd = [
                *base, "train-selfplay",
                "--init_checkpoint", self.sp_init_ckpt.get(),
                "--puzzle_checkpoint", self.sp_puzzle_ckpt.get(),
                "--iters", self.sp_iters.get(),
                "--games_per_iter", self.sp_games_per_iter.get(),
                "--batch_size", self.sp_batch_size.get(),
                "--train_batches", self.sp_train_batches.get(),
                "--lr", self.sp_lr.get(),
                "--replay_dir", self.sp_replay_dir.get(),
                "--num_sims", self.sp_num_sims.get(),
                "--eval_num_sims", self.sp_eval_num_sims.get(),
                "--draw_penalty", self.sp_draw_penalty.get(),
                "--no_progress_limit", self.sp_no_prog_limit.get(),
                "--no_progress_penalty", self.sp_no_prog_penalty.get(),
                "--repeat2_penalty", self.sp_repeat2_penalty.get(),
                "--temp_floor", self.sp_temp_floor.get(),
                "--material_scale", self.sp_mat_scale.get(),
                "--exchange_scale", self.sp_exch_scale.get(),
                "--early_sims", self.sp_early_sims.get(),
                "--early_plies", self.sp_early_plies.get(),
                "--late_sims", self.sp_late_sims.get(),
                "--gate_games", self.sp_gate_games.get(),
                "--gate_min_score", self.sp_gate_min_score.get(),
                "--feedback_weight", self.sp_feedback_weight.get(),
                "--feedback_batch_size", self.sp_feedback_batch.get(),
                "--feedback_margin", self.sp_feedback_margin.get(),
                "--feedback_max_samples", self.sp_feedback_max_samples.get(),
            ]
            if self.sp_feedback_jsonl.get().strip():
                cmd += ["--feedback_jsonl", self.sp_feedback_jsonl.get().strip()]
            if self.sp_prefer_puzzle.get():
                cmd.append("--prefer_puzzle_init")
            if self.sp_stop_threefold.get():
                cmd.append("--stop_on_threefold")
            if self.sp_stop_repeat2.get():
                cmd.append("--stop_on_repeat2")
            if self.sp_use_mat_shape.get():
                cmd.append("--use_material_shaping")
            if self.sp_augment.get():
                cmd.append("--augment")
        elif tab_name == "Puzzle Train":
            cmd = [
                *base, "train-puzzles",
                "--limit", self.pt_limit.get(),
                "--batch_size", self.pt_batch.get(),
                "--epochs", self.pt_epochs.get(),
                "--lr", self.pt_lr.get(),
                "--num_workers", self.pt_workers.get(),
                "--torch_threads", self.pt_torch_threads.get(),
                "--tune_batch_sizes", self.pt_tune_batches.get(),
                "--tune_torch_threads", self.pt_tune_threads.get(),
                "--tune_max_batches", self.pt_tune_steps.get(),
                "--val_ratio", self.pt_val_ratio.get(),
                "--seed", self.pt_seed.get(),
                "--label_smoothing", self.pt_label_smoothing.get(),
                "--overfit_debug_n", self.pt_overfit.get(),
                "--pgn_dir", self.pt_pgn_dir.get(),
                "--pgn_max_games", self.pt_pgn_max.get(),
                "--progress_every_batches", self.pt_progress.get(),
                "--resume_checkpoint", self.pt_resume.get(),
            ]
            if self.pt_auto_tune.get():
                cmd.append("--auto_tune_cpu")
            if self.pt_tune_only.get():
                cmd.append("--tune_only")
            if self.pt_cache_dir.get().strip():
                cmd += ["--cache_dir", self.pt_cache_dir.get().strip()]
            elif self.pt_puzzles_csv.get().strip():
                cmd += ["--puzzles_csv", self.pt_puzzles_csv.get().strip()]
            else:
                self._append_log("Puzzle Train needs cache_dir or puzzles_csv.\n")
                return
        elif tab_name == "Build Puzzle Cache":
            cmd = [
                *base, "build-puzzle-cache",
                "--puzzles_csv", self.bc_csv.get(),
                "--out_dir", self.bc_out.get(),
                "--limit", self.bc_limit.get(),
                "--val_ratio", self.bc_val_ratio.get(),
                "--seed", self.bc_seed.get(),
                "--shard_size", self.bc_shard_size.get(),
                "--max_train_shards", self.bc_max_train.get(),
                "--max_val_shards", self.bc_max_val.get(),
                "--workers", self.bc_workers.get(),
                "--compression", ("compressed" if self.bc_compress.get() else "none"),
            ]
            if self.bc_clean.get():
                cmd += ["--clean_out_dir"]
        elif tab_name == "Generate Puzzles":
            cmd = [
                *base, "generate-puzzles",
                "--out", self.gp_out.get(),
                "--count", self.gp_count.get(),
                "--seed", self.gp_seed.get(),
            ]
        elif tab_name == "Play vs Model":
            cmd = [
                *base, "play-vs-model",
                "--device", self.pm_device.get(),
                "--checkpoint", self.pm_ckpt.get(),
                "--num_sims", self.pm_sims.get(),
                "--human_replay_dir", self.pm_human_replay.get(),
                "--human_pgn_dir", self.pm_human_pgn.get(),
            ]
            if self.pm_black.get():
                cmd.append("--play_as_black")
            if self.pm_save_samples.get():
                cmd.append("--save_training_samples")
        elif tab_name == "Model vs Model":
            cmd = [
                *base, "model-vs-model",
                "--device", self.mm_device.get(),
                "--num_sims", self.mm_sims.get(),
                "--move_delay_ms", self.mm_delay.get(),
                "--pgn_dir", self.mm_pgn_dir.get(),
            ]
        elif tab_name == "Feedback Candidates":
            side = self.fc_side.get().strip().lower()
            if side not in ("both", "white", "black"):
                self._append_log("Feedback Candidates: side must be one of both|white|black.\n")
                return
            cmd = [
                *base, "generate-feedback-candidates",
                "--pgn_glob", self.fc_pgn_glob.get(),
                "--out", self.fc_out.get(),
                "--max_games", self.fc_max_games.get(),
                "--max_plies_per_game", self.fc_max_plies.get(),
                "--min_ply", self.fc_min_ply.get(),
                "--side", side,
                "--max_legal_moves", self.fc_max_legal.get(),
            ]
            if self.fc_recursive.get():
                cmd.append("--recursive")
        else:  # PGN Viewer
            cmd = [
                *base, "pgn-viewer",
                "--games_dir", self.pv_games_dir.get(),
            ]
            if self.pv_pgn_path.get().strip():
                cmd += ["--pgn_path", self.pv_pgn_path.get().strip()]
            if self.pv_load_latest.get():
                cmd.append("--load_latest")

        self.start_process(tab_name, cmd)

    def _update_status(self) -> None:
        n = len(self.procs)
        if n == 0:
            self.status_var.set("Idle (0 running)")
            return
        self.status_var.set(f"Running ({n} process{'es' if n != 1 else ''})")

    def start_process(self, tab_name: str, cmd: list[str]) -> None:
        existing = self.procs.get(tab_name)
        if existing is not None and existing.poll() is None:
            self._append_log(f"[{tab_name}] A process is already running for this tab. Stop it first.\n")
            return
        self._append_log(f"\n[{tab_name}] [Run] " + " ".join(cmd) + "\n")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        creationflags = 0
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            creationflags=creationflags,
        )
        self.procs[tab_name] = proc
        self._update_status()

        t = threading.Thread(target=self._reader_thread, args=(tab_name, proc), daemon=True)
        t.start()

    def _reader_thread(self, tab_name: str, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log_queue.put(("out", tab_name, line))
        code = proc.wait()
        self.log_queue.put(("done", tab_name, int(code)))

    def _drain_logs(self) -> None:
        while True:
            try:
                kind, payload, extra = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "out":
                tab_name = str(payload)
                line = str(extra)
                if line.endswith("\n"):
                    self._append_log(f"[{tab_name}] {line}")
                else:
                    self._append_log(f"[{tab_name}] {line}\n")
                continue
            if kind == "done":
                tab_name = str(payload)
                code = int(extra)
                proc = self.procs.get(tab_name)
                if proc is not None and proc.poll() is not None:
                    self.procs.pop(tab_name, None)
                self._append_log(f"\n[{tab_name}] [Process exited with code {code}]\n")
                self._update_status()
        self.after(120, self._drain_logs)

    def stop_process(self) -> None:
        tab_name = self.notebook.tab(self.notebook.select(), "text")
        proc = self.procs.get(tab_name)
        if proc and proc.poll() is None:
            pid = proc.pid
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                proc.terminate()
            self._append_log(f"[{tab_name}] [Process termination requested for PID {pid}]\n")
        else:
            self._append_log(f"[{tab_name}] [No running process]\n")

    def stop_all_processes(self) -> None:
        if not self.procs:
            self._append_log("[No running process]\n")
            return
        for tab_name, proc in list(self.procs.items()):
            if proc.poll() is not None:
                continue
            pid = proc.pid
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                proc.terminate()
            self._append_log(f"[{tab_name}] [Process termination requested for PID {pid}]\n")

    def clear_logs(self) -> None:
        self.log.delete("1.0", "end")

    def _append_log(self, text: str) -> None:
        # Auto-follow only if user is already near bottom; keep manual history browsing stable.
        y0, y1 = self.log.yview()
        at_bottom = y1 >= 0.995
        self.log.insert("end", text)
        if at_bottom:
            self.log.see("end")


if __name__ == "__main__":
    app = ChessControlPanel()
    app.mainloop()
