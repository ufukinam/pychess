from __future__ import annotations

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

        self.proc: subprocess.Popen | None = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()

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
            "PGN Viewer",
        ]:
            frame = self._create_scrollable_tab(self.notebook, name)
            self.tabs[name] = frame

        self._build_selfplay_tab()
        self._build_puzzle_train_tab()
        self._build_cache_tab()
        self._build_generate_tab()
        self._build_play_tab()
        self._build_viewer_tab()

        run_bar = ttk.Frame(top)
        run_bar.grid(row=1, column=0, sticky="ew", pady=(10, 6))
        ttk.Button(run_bar, text="Run Selected Tab", command=self.run_selected_tab).pack(
            side="left"
        )
        ttk.Button(run_bar, text="Stop Running", command=self.stop_process).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(run_bar, text="Clear Logs", command=self.clear_logs).pack(
            side="left", padx=(8, 0)
        )

        self.status_var = tk.StringVar(value="Idle")
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
            "games",
            "Folder used when Load latest is ON.",
        )
        self.pv_load_latest = self._add_check(
            f,
            "Load latest on start",
            True,
            "ON loads newest PGN in games dir when no specific path is provided.",
        )

    def run_selected_tab(self) -> None:
        tab_name = self.notebook.tab(self.notebook.select(), "text")
        if tab_name == "Self-Play Train":
            cmd = [
                sys.executable, "chess_interface.py", "train-selfplay",
                "--init_checkpoint", self.sp_init_ckpt.get(),
                "--puzzle_checkpoint", self.sp_puzzle_ckpt.get(),
            ]
            if self.sp_prefer_puzzle.get():
                cmd.append("--prefer_puzzle_init")
        elif tab_name == "Puzzle Train":
            cmd = [
                sys.executable, "chess_interface.py", "train-puzzles",
                "--limit", self.pt_limit.get(),
                "--batch_size", self.pt_batch.get(),
                "--epochs", self.pt_epochs.get(),
                "--lr", self.pt_lr.get(),
                "--val_ratio", self.pt_val_ratio.get(),
                "--seed", self.pt_seed.get(),
                "--label_smoothing", self.pt_label_smoothing.get(),
                "--overfit_debug_n", self.pt_overfit.get(),
                "--pgn_dir", self.pt_pgn_dir.get(),
                "--pgn_max_games", self.pt_pgn_max.get(),
                "--progress_every_batches", self.pt_progress.get(),
                "--resume_checkpoint", self.pt_resume.get(),
            ]
            if self.pt_cache_dir.get().strip():
                cmd += ["--cache_dir", self.pt_cache_dir.get().strip()]
            elif self.pt_puzzles_csv.get().strip():
                cmd += ["--puzzles_csv", self.pt_puzzles_csv.get().strip()]
            else:
                self._append_log("Puzzle Train needs cache_dir or puzzles_csv.\n")
                return
        elif tab_name == "Build Puzzle Cache":
            cmd = [
                sys.executable, "chess_interface.py", "build-puzzle-cache",
                "--puzzles_csv", self.bc_csv.get(),
                "--out_dir", self.bc_out.get(),
                "--limit", self.bc_limit.get(),
                "--val_ratio", self.bc_val_ratio.get(),
                "--seed", self.bc_seed.get(),
                "--shard_size", self.bc_shard_size.get(),
                "--max_train_shards", self.bc_max_train.get(),
                "--max_val_shards", self.bc_max_val.get(),
            ]
        elif tab_name == "Generate Puzzles":
            cmd = [
                sys.executable, "chess_interface.py", "generate-puzzles",
                "--out", self.gp_out.get(),
                "--count", self.gp_count.get(),
                "--seed", self.gp_seed.get(),
            ]
        elif tab_name == "Play vs Model":
            cmd = [
                sys.executable, "chess_interface.py", "play-vs-model",
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
        else:  # PGN Viewer
            cmd = [
                sys.executable, "chess_interface.py", "pgn-viewer",
                "--games_dir", self.pv_games_dir.get(),
            ]
            if self.pv_pgn_path.get().strip():
                cmd += ["--pgn_path", self.pv_pgn_path.get().strip()]
            if self.pv_load_latest.get():
                cmd.append("--load_latest")

        self.start_process(cmd)

    def start_process(self, cmd: list[str]) -> None:
        if self.proc and self.proc.poll() is None:
            self._append_log("A process is already running. Stop it first.\n")
            return
        self._append_log("\n[Run] " + " ".join(cmd) + "\n")
        self.status_var.set("Running...")

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        t = threading.Thread(target=self._reader_thread, daemon=True)
        t.start()

    def _reader_thread(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self.log_queue.put(line)
        code = self.proc.wait()
        self.log_queue.put(f"\n[Process exited with code {code}]\n")
        self.log_queue.put("__PROCESS_DONE__")

    def _drain_logs(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if line == "__PROCESS_DONE__":
                self.status_var.set("Idle")
                continue
            self._append_log(line)
        self.after(120, self._drain_logs)

    def stop_process(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._append_log("[Process termination requested]\n")
        else:
            self._append_log("[No running process]\n")

    def clear_logs(self) -> None:
        self.log.delete("1.0", "end")

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")


if __name__ == "__main__":
    app = ChessControlPanel()
    app.mainloop()
