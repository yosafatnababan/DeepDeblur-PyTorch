"""Tkinter GUI for DeepDeblur-PyTorch demo inference.

This GUI is intentionally lightweight and only wraps the existing demo pipeline
from the repository. It lets the user select:
- pretrained experiment directory (save_dir)
- input folder with blurred images
- output folder for deblurred results
- precision (single / half)
- device (CPU / CUDA)

The GUI runs `src/main.py --demo true ...` in a subprocess so the original
training code remains unchanged.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT_DIR, ".."))
MAIN_SCRIPT = os.path.join(ROOT_DIR, "main.py")


class DeblurGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeepDeblur-PyTorch GUI")
        self.geometry("820x620")
        self.minsize(780, 580)

        self.proc = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self._build_vars()
        self._build_ui()
        self.after(100, self._poll_log_queue)

    def _build_vars(self):
        self.save_dir_var = tk.StringVar(value="GOPRO_L1")
        self.input_dir_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value="")
        self.device_var = tk.StringVar(value="cuda")
        self.precision_var = tk.StringVar(value="single")
        self.n_gpus_var = tk.StringVar(value="1")
        self.extra_args_var = tk.StringVar(value="")

    def _build_ui(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="DeepDeblur-PyTorch GUI", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w", pady=(0, 10))

        subtitle = ttk.Label(
            main,
            text="GUI sederhana untuk menjalankan demo/inference deblurring dari folder input ke folder output.",
            wraplength=760,
        )
        subtitle.pack(anchor="w", pady=(0, 14))

        form = ttk.LabelFrame(main, text="Konfigurasi Demo", padding=12)
        form.pack(fill="x")

        self._row_entry(form, 0, "Save dir / pretrained experiment", self.save_dir_var, self._choose_save_dir)
        self._row_entry(form, 1, "Input folder blur images", self.input_dir_var, self._choose_input_dir)
        self._row_entry(form, 2, "Output folder hasil", self.output_dir_var, self._choose_output_dir)

        row3 = ttk.Frame(form)
        row3.grid(row=3, column=0, sticky="ew", pady=6)
        row3.columnconfigure(1, weight=1)
        ttk.Label(row3, text="Device", width=28).grid(row=0, column=0, sticky="w")
        device_box = ttk.Combobox(row3, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly", width=18)
        device_box.grid(row=0, column=1, sticky="w")
        ttk.Label(row3, text="Precision").grid(row=0, column=2, sticky="w", padx=(18, 8))
        precision_box = ttk.Combobox(row3, textvariable=self.precision_var, values=["single", "half"], state="readonly", width=18)
        precision_box.grid(row=0, column=3, sticky="w")

        row4 = ttk.Frame(form)
        row4.grid(row=4, column=0, sticky="ew", pady=6)
        row4.columnconfigure(1, weight=1)
        ttk.Label(row4, text="n_GPUs", width=28).grid(row=0, column=0, sticky="w")
        ttk.Entry(row4, textvariable=self.n_gpus_var, width=20).grid(row=0, column=1, sticky="w")
        ttk.Label(row4, text="Extra args (optional)").grid(row=0, column=2, sticky="w", padx=(18, 8))
        ttk.Entry(row4, textvariable=self.extra_args_var, width=36).grid(row=0, column=3, sticky="w")

        btns = ttk.Frame(main)
        btns.pack(fill="x", pady=12)
        self.run_btn = ttk.Button(btns, text="Run Deblur", command=self._run)
        self.run_btn.pack(side="left")
        self.clear_btn = ttk.Button(btns, text="Clear Log", command=self._clear_log)
        self.clear_btn.pack(side="left", padx=8)
        self.open_out_btn = ttk.Button(btns, text="Open Output Folder", command=self._open_output_folder)
        self.open_out_btn.pack(side="left", padx=8)

        log_frame = ttk.LabelFrame(main, text="Log", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", height=18)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

        self._log("Siap. Pilih folder input dan output, lalu klik Run Deblur.")

    def _row_entry(self, parent, row, label, variable, browse_command):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=6)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label, width=28).grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=variable).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Browse", command=browse_command).grid(row=0, column=2, sticky="e")

    def _choose_save_dir(self):
        path = filedialog.askdirectory(title="Pilih folder experiment pretrained")
        if path:
            self.save_dir_var.set(os.path.basename(path) if os.path.basename(path) else path)

    def _choose_input_dir(self):
        path = filedialog.askdirectory(title="Pilih folder input blur images")
        if path:
            self.input_dir_var.set(path)

    def _choose_output_dir(self):
        path = filedialog.askdirectory(title="Pilih folder output")
        if path:
            self.output_dir_var.set(path)

    def _open_output_folder(self):
        path = self.output_dir_var.get().strip()
        if not path:
            messagebox.showinfo("Info", "Pilih output folder dulu.")
            return
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def _log(self, message: str):
        self.log_text.insert(tk.END, message.rstrip() + "\n")
        self.log_text.see(tk.END)

    def _poll_log_queue(self):
        try:
            while True:
                self._log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _run(self):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Masih berjalan", "Proses deblur sedang berjalan.")
            return

        save_dir = self.save_dir_var.get().strip()
        input_dir = self.input_dir_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        device = self.device_var.get().strip()
        precision = self.precision_var.get().strip()
        n_gpus = self.n_gpus_var.get().strip()
        extra_args = self.extra_args_var.get().strip()

        if not save_dir:
            messagebox.showerror("Error", "Save dir / pretrained experiment belum diisi.")
            return
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Input folder tidak valid.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Output folder belum diisi.")
            return

        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable,
            MAIN_SCRIPT,
            "--save_dir", save_dir,
            "--demo", "true",
            "--demo_input_dir", input_dir,
            "--demo_output_dir", output_dir,
            "--device_type", device,
            "--precision", precision,
            "--n_GPUs", n_gpus,
        ]

        if extra_args:
            cmd.extend(extra_args.split())

        self._log("Menjalankan:")
        self._log(" ".join(cmd))
        self.run_btn.configure(state="disabled")

        def runner():
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    self.log_queue.put(line.rstrip())
                code = self.proc.wait()
                self.log_queue.put(f"Proses selesai dengan exit code {code}.")
            except Exception as exc:  # pragma: no cover - GUI runtime safety
                self.log_queue.put(f"Gagal menjalankan proses: {exc}")
            finally:
                self.proc = None
                self.after(0, lambda: self.run_btn.configure(state="normal"))

        threading.Thread(target=runner, daemon=True).start()


def main():
    app = DeblurGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
