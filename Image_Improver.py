import shutil
import tempfile
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from upscale_ncnn import (
    DEFAULT_MODEL,
    download_ncnn_bundle,
    find_ncnn_executable,
    install_windows_prerequisites,
    lite_upscale,
    ncnn_available,
    prepare_ncnn_input,
    run_ncnn_upscale,
)

PREVIEW_MAX = 520

class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Image Improver")
        self.root.minsize(420, 480)

        self._source_path: Path | None = None
        self._pending_output: Path | None = None
        self._pending_workdir: Path | None = None
        self._photo_preview: ImageTk.PhotoImage | None = None
        self._busy = False

        self.status_var = tk.StringVar(value="Open an image.")

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="Open…", command=self._on_open).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Enhance (AI, NCNN)", command=self._on_ai).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Fast (without AI)", command=self._on_lite).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Save as…", command=self._on_save).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Download NCNN…", command=self._on_download_ncnn).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(
            toolbar,
            text="Install deps (Win)…",
            command=self._on_install_win_deps,
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.preview_label = ttk.Label(self.root)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(self.root, textvariable=self.status_var, wraplength=560).pack(
            fill=tk.X, padx=8, pady=(0, 8)
        )

        self._refresh_ncnn_hint()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.root.config(cursor="watch" if busy else "")

    def _refresh_ncnn_hint(self) -> None:
        if ncnn_available():
            exe = find_ncnn_executable()
            self.status_var.set(f"NCNN found: {exe}")
        else:
            self.status_var.set(
                "NCNN not found — download in vendor/ or click «Download NCNN…»."
            )

    def _clear_pending(self) -> None:
        if self._pending_workdir and self._pending_workdir.is_dir():
            shutil.rmtree(self._pending_workdir, ignore_errors=True)
        self._pending_output = None
        self._pending_workdir = None

    def _show_preview(self, image_path: Path) -> None:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.Resampling.LANCZOS)
        self._photo_preview = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self._photo_preview)

    def _on_open(self) -> None:
        if self._busy:
            return
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._clear_pending()
        self._source_path = Path(path)
        try:
            self._show_preview(self._source_path)
            self.status_var.set(f"Opened: {self._source_path.name}")
        except OSError as e:
            messagebox.showerror("Error", f"Failed to open file:\n{e}")
            self._source_path = None

    def _on_save(self) -> None:
        if self._busy or not self._pending_output or not self._pending_output.is_file():
            messagebox.showinfo(
                "Saving",
                "No result to save. First, perform enhancement.",
            )
            return
        dest = filedialog.asksaveasfilename(
            title="Save as",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("WebP", "*.webp"),
            ],
        )
        if not dest:
            return
        try:
            dest_path = Path(dest)
            ext = dest_path.suffix.lower()
            img = Image.open(self._pending_output).convert("RGB")
            if ext in {".jpg", ".jpeg"}:
                img.save(dest_path, quality=92, optimize=True)
            elif ext == ".webp":
                img.save(dest_path, quality=90, method=6)
            else:
                img.save(dest_path)
            self.status_var.set(f"Saved: {dest_path.name}")
        except OSError as e:
            messagebox.showerror("Error", f"Failed to save:\n{e}")

    def _run_in_thread(self, target: Callable[[], None]) -> None:
        if self._busy:
            return
        self._set_busy(True)

        def wrapper() -> None:
            try:
                target()
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=wrapper, daemon=True).start()

    def _on_ai(self) -> None:
        if not self._source_path or not self._source_path.is_file():
            messagebox.showwarning("AI", "First, open an image.")
            return
        if not ncnn_available():
            messagebox.showwarning(
                "NCNN not found",
                "Install realesrgan-ncnn-vulkan in the vendor/ folder "
                "or click «Download NCNN…».",
            )
            return

        def job() -> None:
            td: tempfile.TemporaryDirectory | None = None
            out_path: Path | None = None
            workdir: Path | None = None
            try:
                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        "AI: preparation and launch NCNN (may take time)…"
                    ),
                )
                inp, td = prepare_ncnn_input(self._source_path)
                out_dir = tempfile.mkdtemp(prefix="improver-ncnn-")
                workdir = Path(out_dir)
                out_path = workdir / "upscaled.png"
                run_ncnn_upscale(inp, out_path, model_name=DEFAULT_MODEL)

                def done() -> None:
                    self._clear_pending()
                    self._pending_workdir = workdir
                    self._pending_output = out_path
                    self._show_preview(out_path)
                    self.status_var.set(
                        f"Done (AI, {DEFAULT_MODEL}). Save the result if necessary."
                    )

                self.root.after(0, done)
            except Exception as e:
                if workdir and workdir.is_dir():
                    shutil.rmtree(workdir, ignore_errors=True)
                err_text = str(e)

                def err() -> None:
                    messagebox.showerror("AI", err_text)
                    self._refresh_ncnn_hint()

                self.root.after(0, err)
            finally:
                if td is not None:
                    td.cleanup()

        self._run_in_thread(job)

    def _on_lite(self) -> None:
        if not self._source_path or not self._source_path.is_file():
            messagebox.showwarning("Fast", "First, open an image.")
            return

        def job() -> None:
            out_path: Path | None = None
            workdir: Path | None = None
            try:
                self.root.after(
                    0,
                    lambda: self.status_var.set("Fast mode: increase ×4…"),
                )
                out_dir = tempfile.mkdtemp(prefix="improver-lite-")
                workdir = Path(out_dir)
                out_path = workdir / "lite.png"
                lite_upscale(self._source_path, out_path, scale=4)

                def done() -> None:
                    self._clear_pending()
                    self._pending_workdir = workdir
                    self._pending_output = out_path
                    self._show_preview(out_path)
                    self.status_var.set("Done (without AI). Save the result if necessary.")

                self.root.after(0, done)
            except Exception as e:
                if workdir and workdir.is_dir():
                    shutil.rmtree(workdir, ignore_errors=True)
                err_text = str(e)

                def err() -> None:
                    messagebox.showerror("Fast", err_text)

                self.root.after(0, err)

        self._run_in_thread(job)

    def _on_download_ncnn(self) -> None:
        if self._busy:
            return

        def hook(block: int, block_size: int, total: int) -> None:
            if total > 0:
                pct = min(100, int(100 * (block * block_size) / total))

                def upd() -> None:
                    self.status_var.set(f"Downloading NCNN… {pct}%")

                self.root.after(0, upd)

        def job() -> None:
            try:
                self.root.after(
                    0, lambda: self.status_var.set("Downloading and unpacking NCNN…")
                )
                download_ncnn_bundle(progress_hook=hook)

                def ok() -> None:
                    exe = find_ncnn_executable()
                    messagebox.showinfo(
                        "Done",
                        f"NCNN installed.\n{exe}",
                    )
                    self._refresh_ncnn_hint()

                self.root.after(0, ok)
            except Exception as e:
                err_text = str(e)

                def err() -> None:
                    messagebox.showerror("Downloading NCNN", err_text)

                self.root.after(0, err)

        self._run_in_thread(job)

    def _on_install_win_deps(self) -> None:
        if self._busy:
            return

        def job() -> None:
            try:
                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        "Installing/checking Windows prerequisites…"
                    ),
                )
                result = install_windows_prerequisites()

                def ok() -> None:
                    messagebox.showinfo("Windows prerequisites", result)
                    self.status_var.set("Windows prerequisites step finished.")

                self.root.after(0, ok)
            except Exception as e:
                err_text = str(e)

                def err() -> None:
                    messagebox.showerror("Windows prerequisites", err_text)

                self.root.after(0, err)

        self._run_in_thread(job)

    def run(self) -> None:
        self.root.mainloop()

if __name__ == "__main__":
    App().run()