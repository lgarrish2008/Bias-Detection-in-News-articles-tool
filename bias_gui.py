# Requires predict_url.py, Scraper.py and train_regression_model.py to be in the same directory
import tkinter as tk
import tkinter.filedialog as fd
import threading
import os

from predict_url import build_vector, scale_to_0_100
from Scraper import import_csv_to_db, scraper
import train_regression_model
import joblib
import numpy as np

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH    = os.path.join(BASE_DIR, "tf_idf.db")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab_wordid_to_col.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "bias_regression_model.joblib")

# colours used throughout GUI
BG      = "#0d0d0d"
SURFACE = "#1a1a1a"
BORDER  = "#2e2e2e"
TEXT    = "#ffffff"
MUTED   = "#888888"
BLUE    = "#5ab4f5"
RED     = "#c0334d"
NEUTRAL = "#aaaaaa"


def predict(url, model, wordid_to_col):
    # scraper imported from Scraper.py 
    tokens, total_words = scraper(url)

    # reject pages with too little content
    # image-only pages or paywalled articles still return some text
    if total_words < 250:
        raise ValueError(
            f"Page has too little content ({total_words} words), minimum is 250."
        )

    X = build_vector(tokens, total_words, DB_PATH, wordid_to_col)
    raw = model.predict(X)[0]
    return float(scale_to_0_100(raw))


# Donut canvas widget
class DonutChart(tk.Canvas):
    SIZE  = 300
    THICK = 40

    def __init__(self, parent):
        super().__init__(
            parent,
            width=self.SIZE,
            height=self.SIZE,
            bg=BG,
            highlightthickness=0
        )
        self._score = None
        self._draw_idle()

    def set_score(self, score):
        self._score = score
        self._redraw()

    def reset(self):
        self._score = None
        self._draw_idle()

    def _arc_colour(self, score):
        # red for left of centre, blue for right, grey for centre or uncertain
        if score < 40:
            return RED
        if score > 60:
            return BLUE
        return NEUTRAL

    def _label(self, score):
        if score < 40:
            return "LEFT OF\nCENTRE"
        if score > 60:
            return "RIGHT OF\nCENTRE"
        return "CENTRE /\nUNCERTAIN"

    def _draw_idle(self):
        # clears canvas
        # shows placeholder before any URL has been tested
        self.delete("all")
        pad = 20
        self._draw_ring(pad, self.SIZE - pad, BORDER)
        cx = cy = self.SIZE // 2
        self.create_text(
            cx, cy,
            text="BIAS\nRATING:\n\n-\n\nENTER A URL",
            fill=MUTED,
            font=("Helvetica", 11, "bold"),
            anchor="center",
            justify="center"
        )

    def _redraw(self):
        # called when a new score is set
        # clears canvas then redraws with updated score and colour
        self.delete("all")
        score  = self._score
        pad    = 20
        s      = self.SIZE
        colour = self._arc_colour(score)
        label  = self._label(score)

        # draw background ring in dark grey
        self._draw_ring(pad, s - pad, SURFACE)

        # convert score 0-100 to degrees 0-360 for the arc
        # start is 90deg puts the arc start at top of circle
        extent = (score / 100) * 360
        self.create_arc(
            pad, pad, s - pad, s - pad,
            start=90,
            extent=-extent,
            style="arc",
            outline=colour,
            width=self.THICK
        )

        # place text labels in the centre 
        cx = cy = s // 2
        self.create_text(
            cx, cy - 24,
            text="BIAS\nRATING:",
            fill=MUTED,
            font=("Helvetica", 11, "bold"),
            anchor="center",
            justify="center"
        )
        self.create_text(
            cx, cy + 16,
            text=str(round(score)),
            fill=TEXT,
            font=("Helvetica", 38, "bold"),
            anchor="center"
        )
        self.create_text(
            cx, cy + 52,
            text=label,
            fill=colour,
            font=("Helvetica", 11, "bold"),
            anchor="center",
            justify="center"
        )

    def _draw_ring(self, x0, x1, colour):
        # extent=359.99 not 360 because exactly 360 disappears in tkinter
        self.create_arc(
            x0, x0, x1, x1,
            start=0,
            extent=359.99,
            style="arc",
            outline=colour,
            width=self.THICK
        )


# Main application window
class BiasApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Artificial Ignorance - Bias Rater")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(700, 560)

        # model and vocab are None until the user loads or trains 
        self.model = None
        self.wordid_to_col = None

        # StringVars allow labels to update automatically when their value changes
        self.model_path_var = tk.StringVar(value="No model loaded")
        self.mae_var  = tk.StringVar(value="--")
        self.rmse_var = tk.StringVar(value="--")
        self.r2_var   = tk.StringVar(value="--")
        self.status_var = tk.StringVar(
            value="Load a model to begin, or upload a CSV to train one."
        )

        # tracks whether the pipeline is running
        # buttons are disabled while busy
        self._busy = False

        self._build_ui()

    def _build_ui(self):
        root = self.root

        # top bar shows which model is currently loaded
        info = tk.Frame(root, bg=SURFACE, pady=6)
        info.pack(fill="x", side="top")

        tk.Label(
            info, text="Current model loaded:", bg=SURFACE, fg=MUTED,
            font=("Helvetica", 9)
        ).grid(row=0, column=0, padx=(12, 6), sticky="w")

        tk.Label(
            info, textvariable=self.model_path_var, bg=SURFACE, fg=TEXT,
            font=("Helvetica", 9)
        ).grid(row=0, column=1, padx=6, sticky="w")

        # columnconfigure weight=1 pushes the folder button to the far right
        info.columnconfigure(2, weight=1)
        tk.Button(
            info, text="📂", bg=SURFACE, fg=TEXT, relief="flat",
            font=("Helvetica", 11), cursor="hand2",
            command=self._load_model
        ).grid(row=0, column=2, padx=10, sticky="e")

        # stats bar showing MAE, RMSE and R2 of the loaded model
        # uses a list of (label_text, stringvar) pairs to build the row
        stats = tk.Frame(
            root, bg=SURFACE, pady=5,
            highlightbackground=BORDER, highlightthickness=1
        )
        stats.pack(fill="x", side="top")

        for col_idx, (label, var) in enumerate([
            ("Model Statistics:", None),
            ("MAE",  None), (None, self.mae_var),
            ("RMSE", None), (None, self.rmse_var),
            ("R^2",  None), (None, self.r2_var),
        ]):
            if label is not None:
                tk.Label(
                    stats, text=label, bg=SURFACE, fg=MUTED,
                    font=("Helvetica", 9, "bold"), padx=10
                ).grid(row=0, column=col_idx, sticky="w")
            else:
                tk.Label(
                    stats, textvariable=var, bg=SURFACE, fg=TEXT,
                    font=("Helvetica", 9), padx=10
                ).grid(row=0, column=col_idx, sticky="w")

        # CSV upload and retrain buttons
        csv_bar = tk.Frame(root, bg=BG, pady=8)
        csv_bar.pack(fill="x")

        self._csv_btn = tk.Button(
            csv_bar, text="UPLOAD CSV & TRAIN MODEL", bg=SURFACE, fg=TEXT,
            font=("Helvetica", 9, "bold"), relief="flat",
            padx=20, pady=4, cursor="hand2",
            command=self._upload_and_train
        )
        self._csv_btn.pack(side="left", padx=(0, 6))

        # retrain button re-runs model training on existing database
        self._train_btn = tk.Button(
            csv_bar, text="RETRAIN FROM DB", bg=SURFACE, fg=MUTED,
            font=("Helvetica", 9, "bold"), relief="flat",
            padx=20, pady=4, cursor="hand2",
            command=self._train_only
        )
        self._train_btn.pack(side="left")

        csv_bar.pack(fill="x")

        # URL input row
        url_row = tk.Frame(root, bg=BG, pady=6)
        url_row.pack(fill="x", padx=60)

        tk.Label(
            url_row, text="Enter URL:", bg=BG, fg=TEXT,
            font=("Helvetica", 10, "bold")
        ).pack(side="left", padx=(0, 10))

        self.url_entry = tk.Entry(
            url_row, bg=SURFACE, fg=TEXT,
            insertbackground=TEXT, relief="flat",
            font=("Helvetica", 10),
            highlightbackground=BORDER, highlightthickness=1
        )
        self.url_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=4)
        # bind Enter key so user doesn't have to click the button
        self.url_entry.bind("<Return>", lambda e: self._run_prediction())

        tk.Button(
            url_row, text="▶", bg=BLUE, fg=BG,
            font=("Helvetica", 11, "bold"), relief="flat",
            cursor="hand2", padx=10,
            command=self._run_prediction
        ).pack(side="left", padx=(6, 0))

        # donut chart in the centre of the window
        chart_frame = tk.Frame(root, bg=BG)
        chart_frame.pack(expand=True)
        self.donut = DonutChart(chart_frame)
        self.donut.pack(pady=16)

        # status bar at the bottom for feedback messages
        status_bar = tk.Frame(root, bg=SURFACE, pady=4)
        status_bar.pack(fill="x", side="bottom")
        tk.Label(
            status_bar, textvariable=self.status_var, bg=SURFACE, fg=MUTED,
            font=("Helvetica", 9)
        ).pack(padx=12, anchor="w")

    def _set_busy(self, busy):
        # disables buttons while pipeline is running 
        self._busy = busy
        state = "disabled" if busy else "normal"
        self._csv_btn.config(state=state)
        self._train_btn.config(state=state)

    def _load_model(self):
        # open file dialog 
        # filtered to .joblib files
        path = fd.askopenfilename(
            title="Select model (.joblib)",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self.model = joblib.load(path)

            # look for vocab file in same directory 
            model_dir  = os.path.dirname(path)
            vocab_path = os.path.join(model_dir, VOCAB_PATH)

            # if vocab not found automatically, ask user to locate it
            if not os.path.exists(vocab_path):
                vocab_path = fd.askopenfilename(
                    title="Select vocab file (vocab_wordid_to_col.joblib)",
                    filetypes=[("Joblib files", "*.joblib")]
                )

            if not vocab_path:
                self.status_var.set("No vocab file selected -- model not loaded.")
                return

            self.wordid_to_col = joblib.load(vocab_path)
            self.model_path_var.set(path)

            # load metrics joblib if it exists to populate stats bar
            metrics_path = os.path.join(model_dir, "model_metrics.joblib")
            if os.path.exists(metrics_path):
                m = joblib.load(metrics_path)
                self.mae_var.set(round(m["mae"], 2))
                self.rmse_var.set(round(m["rmse"], 2))
                self.r2_var.set(round(m["r2"], 2))
            else:
                self.mae_var.set("--")
                self.rmse_var.set("--")
                self.r2_var.set("--")

            self.status_var.set("Model loaded. Enter a URL and press Enter or ▶")

        except Exception as e:
            self.status_var.set(f"Error loading model: {e}")

    def _upload_and_train(self):
        if self._busy:
            return

        path = fd.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        self._set_busy(True)
        self.status_var.set("Step 1/2 - Scraping articles from CSV...")

        def worker():
            try:
                # run scraping and database population from Scraper.py
                import_csv_to_db(path, DB_PATH)

                self.root.after(0, lambda: self.status_var.set(
                    "Step 2/2 - Training regression model..."
                ))

                # run the training pipeline from train_regression_model.py
                train_regression_model.main()

                # once training is done 
                # load the new model automatically
                self.root.after(0, self._auto_load_trained_model)

            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Pipeline error: {e}"))
                self.root.after(0, lambda: self._set_busy(False))

        # run on a background thread so the window stays responsivewas there 
        threading.Thread(target=worker, daemon=True).start()

    def _train_only(self):
        if self._busy:
            return

        if not os.path.exists(DB_PATH):
            self.status_var.set("No database found. Upload a CSV first.")
            return

        self._set_busy(True)
        self.status_var.set("Training regression model from existing database...")

        def worker():
            try:
                train_regression_model.main()
                self.root.after(0, self._auto_load_trained_model)
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Training error: {e}"))
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _auto_load_trained_model(self):
        # called on the main thread after training completes
        # loads the three joblib files that train_regression_model.main() saved
        try:
            self.model = joblib.load(MODEL_PATH)
            self.wordid_to_col = joblib.load(VOCAB_PATH)
            self.model_path_var.set(MODEL_PATH)

            if os.path.exists(os.path.join(BASE_DIR, "model_metrics.joblib")):
                m = joblib.load(os.path.join(BASE_DIR, "model_metrics.joblib"))
                self.mae_var.set(round(m["mae"], 2))
                self.rmse_var.set(round(m["rmse"], 2))
                self.r2_var.set(round(m["r2"], 2))

            self.status_var.set(
                "Pipeline complete - model trained and loaded. Enter a URL to predict."
            )

        except Exception as e:
            self.status_var.set(f"Training finished but failed to load model: {e}")

        finally:
            # finally block ensures buttons are always re-enabled
            # even if loading the model fails after training
            self._set_busy(False)

    def _run_prediction(self):
        url = self.url_entry.get().strip()

        if not url:
            self.status_var.set("Please enter a URL.")
            return

        if not (url.startswith("http://") or url.startswith("https://")):
            self.status_var.set("Invalid URL - must start with http:// or https://")
            return

        if self.model is None or self.wordid_to_col is None:
            self.status_var.set("Load a model first.")
            return

        self.status_var.set("Scraping and predicting...")
        # reset chart to idle state while prediction runs
        self.donut.reset()

        def worker():
            try:
                score = predict(url, self.model, self.wordid_to_col)
                # use root.after to update the UI from the main thread
                # tkinter widgets cannot be updated from background threads
                self.root.after(0, lambda: self._show_result(score))
            except Exception as e:
                self.root.after(0, lambda e=e: self.status_var.set(f"Error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _show_result(self, score):
        self.donut.set_score(score)
        if score < 40:
            label = "Left of Centre"
        elif score > 60:
            label = "Right of Centre"
        else:
            label = "Centre / Uncertain"
        self.status_var.set(f"Done - Score: {round(score, 1)} / 100 ({label})")

    def run(self):
        # mainloop hands control to tkinter event loop
        # waits for user input until the window is closed
        self.root.mainloop()


if __name__ == "__main__":
    BiasApp().run()