<table>
    <tr>
        <td colspan=2><strong>International University of La Rioja (UNIR)</strong></td>
    </tr>
    <tr>
        <td colspan=2><strong>Master's Degree in Mathematical Engineering and Computing</strong></td>
    </tr>
    <tr>
        <td>Master's thesis submitted by:</td>
        <td><strong>Andrés Benavides, Diego Salamanca, Yeferson Salamanca.</strong></td>
    </tr>
    <tr>
        <td>Supervisor:</td>
        <td>PhD. Juan Francisco Martínez Cerda</td>
    </tr>
</table>


---


# Alert system based on Long Short-Term Memory (LSTM) neural networks for classifying stock market trend signals (buy, sell, or hold)

This project explores whether a type of artificial intelligence model called
**LSTM** (a neural network designed for time-ordered data) can learn to
interpret stock price behavior and suggest a **BUY / HOLD / SELL** signal
for the coming days.
It was developed as a **Master's thesis project (TFM)**.

Everything is organized as a series of **Jupyter notebooks** (interactive
Python documents that run step by step, one cell at a time).

---

## ⚠️ Important notice — read this first

This is an **academic project**, created solely for learning and research.

- It is **NOT financial advice** and must **NOT** be used to make real investment
  or trading decisions.
- The signals produced are the output of an experimental model and may be wrong.
- A stock's past performance is never a guarantee of future results.

If you are considering investing, consult a qualified, licensed financial professional.

---

## What the project does

1. **Downloads** ~12 years of daily historical prices for 25 well-known companies
   (in technology, healthcare, agriculture, energy, and financial services),
   plus 3 "market sentiment" indicators (stock market volatility, the U.S. dollar
   index, and 10-year interest rates).
2. **Cleans and merges** everything into a single organized table.
3. **Transforms** prices into features the model can learn from
   (recent trends, momentum, etc.).
4. **Trains** the LSTM model to classify each stock's likely short-term movement
   into one of five categories (strong/moderate buy, hold, moderate/strong sell).
5. **Analyzes** how well the model performed on data it had never seen, and
   **plots** the suggested signals over the actual price chart.

---

## The 25 companies and market indicators

| Sector | Companies |
| --- | --- |
| Technology | Apple, Microsoft, NVIDIA, Alphabet (Google), Amazon |
| Healthcare | Eli Lilly, UnitedHealth, Johnson & Johnson, Merck, Intuitive Surgical |
| Agriculture | Deere, Archer-Daniels-Midland, Bunge, Mosaic, CF Industries |
| Energy | Exxon Mobil, Chevron, Shell, EOG Resources, Schlumberger |
| Financial services | JPMorgan, Blackstone, Goldman Sachs, Morgan Stanley, American Express |

**Market context indicators:** `^VIX` (volatility index), `DX-Y.NYB`
(U.S. dollar index), `^TNX` (10-year Treasury bond yield).

---

## Software requirements and installation

You only need to do this once. The following steps assume you are starting from scratch.

### 1. Install Python

Install **Python 3.11 or newer** from [python.org](https://www.python.org/downloads/).
On Windows, **check the "Add Python to PATH" box** during installation.

To confirm it worked, open a terminal (on Windows: **PowerShell**; on macOS:
**Terminal**) and type:

```bash
python --version
```

You should see something like `Python 3.11.x` or newer.

### 2. Locate the project on your computer

Place this project folder somewhere easy to find (for example, your Documents
folder). Then, in the terminal, navigate to the project folder:

```bash
cd path/to/TFM_LSTM
```

### 3. Create an isolated environment (recommended)

This keeps the project's software separate from the rest of your computer.

```bash
python -m venv .venv
```

Then **activate it**:

- **Windows (PowerShell):**

```bash
.venv\Scripts\Activate.ps1
```

- **macOS / Linux:**

```bash
source .venv/bin/activate
```

When it is active, you will see `(.venv)` at the start of the terminal prompt.

### 4. Install the required software

All required libraries are listed in `requirements.txt`. Install them with
one command:

```bash
pip install -r requirements.txt
```

This installs, among others:

- `jupyter` / `ipykernel` — to run the notebooks
- `pandas`, `numpy` — to work with data tables
- `yfinance` — to download stock prices
- `tensorflow` — the engine that runs the LSTM model
- `scikit-learn` — to measure prediction quality
- `matplotlib` — to draw charts

### 5. Open the notebooks

Start Jupyter from the project folder:

```bash
jupyter notebook
```

Your web browser will open. Enter the `notebooks/` folder and open any notebook.
To run it, use **Run → Run All Cells**, or run the cells one at a time
with **Shift + Enter**.

> Tip: You can also open the notebooks directly in editors such as **VS Code** or
> **Cursor**, which have a built-in "Run" button in each cell.

---

## About the data (read before running)

We have **intentionally kept a copy of the raw and processed data** inside the project:

- `data/raw/` — one file per company/indicator, exactly as originally downloaded.
- `data/processed/` — the clean merged table used by the model.

This dataset covers the **last 12 years, from `2014-05-30` to `2026-05-29`**.

We kept it deliberately so **you can reproduce exactly the same results shown
in the notebooks**. Financial data changes every day, so if everyone downloaded
fresh data, each person would get slightly different numbers and charts.

### Which notebooks should I run?

You have two options, depending on your goal:

**Option A — Reproduce our results (recommended for most users)**

Run the notebooks **starting from `03`** onward (`03 → 04 → 05`). These use the
data already included in the project, so your charts and metrics should closely
match the notebook outputs.

**Note**: running notebook 04 (training the model) could take up to 30 minutes.

**Option B — Try it with the latest market data**

If you want up-to-date prices, run the notebooks **starting from `01`** (`01 → 02_1 →
02_2 → 03 → 04 → 05`). Notebook `01` downloads the latest data from Yahoo Finance,
which **overwrites** the default dataset. Your results will then differ from ours
because the market will have changed.

---

## The notebooks, in order

| Notebook | What it does |
| --- | --- |
| `01_download_data.ipynb` | Downloads ~12 years of daily prices for all tickers from Yahoo Finance. |
| `02_merge_data.ipynb` | Merges each company into a single table and cleans it (numeric, weekdays only, missing values imputed). |
| `03_visualize_processed_data.ipynb` | Draws price and indicator charts so you can check data consistency. |
| `04_preprocess_data.ipynb` | Converts prices into model-ready features and labels, and splits the data for training/testing. |
| `05_lstm_model.ipynb` | Builds and trains the LSTM model, then saves it. |
| `06_test_analysis.ipynb` | Evaluates the trained model on unseen data and plots the suggested signals over actual prices. |

The **first cell** of each notebook explains its inputs, outputs, and steps in more detail.

---

## How the signals are organized

The model classifies each stock's likely short-term movement into **five categories**
(from most bullish to most bearish):

`BUY_1000USD` → `BUY_500USD` → `HOLD` → `SELL_50%` → `SELL_100%`

These labels are illustrative position-sizing ideas used for the study —
**again, they are not investment recommendations.**

---

## Project structure

```
TFM_LSTM/
├── notebooks/        # Step-by-step Jupyter notebooks (start here)
├── metadata/         # Configuration files (tickers, model and preprocessing settings)
├── data/
│   ├── raw/          # Financial data downloads (one file per ticker)
│   ├── processed/    # Clean merged financial dataset
│   ├── datasets/     # Model-ready data produced by notebook 03
│   └── models/       # The trained model and results reports
├── src/              # Shared helper code reused by the notebooks
├── requirements.txt  # The list of software to install
└── README.md         # This file
```

You normally do **not** need to modify `src/` or `metadata/` — the notebooks read
them automatically. Advanced users can edit the settings in `metadata/` to experiment.

---

## A note for nontechnical readers

If a notebook displays a red error message, the most common causes are:

1. The software was not installed (repeat the installation steps above).
2. You ran a cell out of order — use **Run → Run All Cells** to run them from top to bottom.
3. You started with notebook `01` without an internet connection (downloads require one).

When in doubt, restart and run everything from the beginning: **Kernel → Restart & Run All**.

---

*This repository is an academic Master's thesis project. It is shared for educational
purposes and comes with no guarantees. It must not be used as the basis for any
financial or trading decision.*
