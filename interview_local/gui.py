"""Optional native interface with separate views for all persistent stages."""

import json
import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .artifacts import save_readable, save_speakers_with_aliases
from .llm import OllamaClient, enhance, review
from .pipeline import process

STAGE_VIEWS = {
    "Raw": "01_raw.txt",
    "Speakers": "02_speakers.txt",
    "Readable": "03_readable.txt",
    "Review": "04_review.txt",
}


def launch():
    root = tk.Tk()
    root.title("Interview Local · German · local processing")
    root.geometry("1000x820")
    events = queue.Queue()
    # pragma: no mutate start -- only the truthiness of running is read
    running = False
    # pragma: no mutate end
    output = None
    path = tk.StringVar()
    name_a = tk.StringVar(value="Speaker A")
    name_b = tk.StringVar(value="Speaker B")
    token = tk.StringVar()
    offline = tk.BooleanVar()
    alignment = tk.BooleanVar()
    use_llm = tk.BooleanVar(value=True)
    status = tk.StringVar(
        value="Ready. Completed stages are saved separately; originals are retained."
    )
    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Interview Local", font=("Helvetica", 22, "bold")).pack(
        anchor="w"
    )
    ttk.Entry(frame, textvariable=path).pack(fill="x", pady=8)

    def choose():
        value = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a"), ("All files", "*")]
        )
        if value:
            path.set(value)

    ttk.Button(frame, text="Choose audio file …", command=choose).pack(anchor="w")
    ttk.Label(
        frame,
        text="Optional HF token: only needed without a complete local community-1 folder",
    ).pack(anchor="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=token, show="•").pack(fill="x")
    ttk.Checkbutton(
        frame, text="Offline speech models (download caches first)", variable=offline
    ).pack(anchor="w")
    ttk.Checkbutton(
        frame, text="Additional German word alignment (optional)", variable=alignment
    ).pack(anchor="w")
    ttk.Checkbutton(
        frame, text="Create a readable draft with local Ministral", variable=use_llm
    ).pack(anchor="w")
    names = ttk.Frame(frame)
    names.pack(fill="x", pady=8)
    ttk.Label(names, text="Names for the new run:").pack(side="left")
    ttk.Entry(names, textvariable=name_a, width=24).pack(side="left", padx=8)
    ttk.Entry(names, textvariable=name_b, width=24).pack(side="left")
    notebook = ttk.Notebook(frame)
    notebook.pack(fill="both", expand=True, pady=8)
    views = {}
    for label in STAGE_VIEWS:
        view = tk.Text(notebook, wrap="word", font=("Helvetica", 13))
        notebook.add(view, text=label)
        views[label] = view
    ttk.Label(frame, textvariable=status, wraplength=950).pack(anchor="w")
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=8)
    controls = []

    def progress(message):
        events.put(("status", message))

    def set_controls(state):
        for button in controls:
            button.config(state=state)

    def submit(function):
        nonlocal running
        if running:
            return
        running = True
        set_controls("disabled")

        def job():
            try:
                function()
                # pragma: no mutate start -- every kind but status and error finishes the job as done
                events.put(("done", None))
                # pragma: no mutate end
            # A failure must reach the window; an uncaught one would end the thread silently.
            except Exception as exc:  # noqa: BLE001
                events.put(("error", str(exc)))

        threading.Thread(target=job, daemon=True).start()

    def refresh_views():
        if output is None:
            return
        for label, filename in STAGE_VIEWS.items():
            file = output / filename
            if file.exists():
                views[label].delete("1.0", "end")
                views[label].insert("1.0", file.read_text(encoding="utf-8"))

    def start():
        nonlocal output
        source = Path(path.get()).expanduser()
        if not source.is_file():
            messagebox.showerror("File not found", "Choose an existing audio file.")
            return
        base = filedialog.askdirectory(
            title="Choose a parent folder for this run's saved stages"
        )
        if not base:
            return
        run_output = Path(base) / datetime.now().strftime(
            "interview_%Y-%m-%d_%H-%M-%S-%f"
        )
        output = run_output
        entered_token = token.get().strip() or None
        offline_only = offline.get()
        aligned = alignment.get()
        names_snapshot = entered_names()
        edit_enabled = use_llm.get()
        clear_views()

        def job():
            if edit_enabled:
                OllamaClient().ready()
            result = process(
                source, progress, entered_token, offline_only, aligned, run_output
            )
            result["speakers"].update(names_snapshot)
            save_speakers_with_aliases(result, run_output)
            if edit_enabled:
                save_readable(enhance(result, progress), run_output)

        submit(job)

    def entered_names():
        return {
            "A": name_a.get().strip() or "Speaker A",
            "B": name_b.get().strip() or "Speaker B",
        }

    def clear_views():
        for view in views.values():
            view.delete("1.0", "end")

    def check():
        nonlocal output
        selected = filedialog.askopenfilename(
            title="Select 03_readable.json for a separate comparison",
            filetypes=[("JSON transcript", "*.json")],
        )
        if not selected:
            return
        review_output = Path(selected).parent / datetime.now().strftime(
            "review_%Y-%m-%d_%H-%M-%S-%f"
        )
        output = review_output

        def job():
            edited = json.loads(Path(selected).read_text(encoding="utf-8"))
            save_readable(review(edited, progress), review_output)

        submit(job)

    for label, action in (
        ("Process and save stages …", start),
        ("Compare readable draft …", check),
        (
            "Download / prepare Ministral",
            lambda: submit(lambda: OllamaClient().prepare(progress)),
        ),
    ):
        # pragma: no mutate start -- the parent only places the button within the layout
        button = ttk.Button(buttons, text=label, command=action)
        # pragma: no mutate end
        button.pack(side="left", padx=(0, 8))
        controls.append(button)

    def poll():
        for _ in range(100):
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                status.set(payload[-500:])
            else:
                finish_job(kind, payload)
        root.after(150, poll)

    def finish_job(kind, payload):
        nonlocal running
        # pragma: no mutate start -- only the truthiness of running is read
        running = False
        # pragma: no mutate end
        token.set("")
        set_controls("normal")
        refresh_views()
        if kind == "error":
            status.set("Processing failed. Completed stages remain saved.")
            messagebox.showerror("Processing failed", payload)
        else:
            status.set(
                f"Done. Saved stages: {output}" if output else "Local model prepared."
            )

    def close():
        if running:
            messagebox.showinfo(
                "Processing in progress",
                "Wait for completion or press Ctrl+C in the terminal to cancel.",
            )
            return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    poll()
    root.mainloop()
