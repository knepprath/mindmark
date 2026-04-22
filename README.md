<div align="center">

# 🔖 mindmark

**Your bookmarks, finally searchable.**\
Ask in natural language — mindmark remembers what you saved.

[![PyPI](https://img.shields.io/pypi/v/mindmark?color=blue&label=PyPI)](https://pypi.org/project/mindmark/)
[![Python](https://img.shields.io/pypi/pyversions/mindmark)](https://pypi.org/project/mindmark/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/sukanth/mindmark/actions/workflows/ci.yml/badge.svg)](https://github.com/sukanth/mindmark/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)]()

100% local · No cloud · No API keys · Nothing leaves your machine

<img src="assets/mindmark-hero.gif" alt="mindmark demo" width="800" />

</div>

---

## Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Install](#-install)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Storage Layout](#-storage-layout)
- [Uninstall](#-uninstall)
- [Development](#-development)
- [License](#-license)

---

## ✨ Features

| Command | What it does |
|---|---|
| `mindmark sync` | **Auto-detect** installed browsers and sync bookmarks directly — no export needed |
| `mindmark find "query"` | Semantic search over titles, folders, domains, and URL slugs — returns top-K with similarity scores |
| `mindmark open "query"` | Search and open the best match in your default browser |
| `mindmark stats` | Show index size, model info, top domains, and top folders |
| `mindmark index <file>` | Import bookmarks from an exported HTML file (legacy workflow) |
| `mindmark validate` | Check indexed bookmark URLs for stale links (HTTP 4xx/5xx or unreachable) and report them |
| `mindmark drop-index` | Delete the local SQLite index database (with confirmation unless `--yes`) |

> 🔌 **Works offline** after the first run. Embeddings run on-device via [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime, ~130 MB one-time model download).

### Supported Browsers

| Browser | macOS | Linux | Windows |
|---|---|---|---|
| **Chrome** | ✅ | ✅ | ✅ |
| **Edge** | ✅ | ✅ | ✅ |
| **Brave** | ✅ | ✅ | ✅ |
| **Firefox** | ✅ | ✅ | ✅ |

mindmark reads bookmark files directly from browser data directories — no export step, no browser extension.

---

## 📋 Prerequisites

| Requirement | Details |
|---|---|
| **Python 3.9+** | [python.org/downloads](https://www.python.org/downloads/) — on Windows, check **"Add Python to PATH"** during setup |
| **pip** | Bundled with Python — verify with `pip --version` or `pip3 --version` |
| **Internet** | Needed only once to download the embedding model (~130 MB). Everything after that is offline |

<details>
<summary>💡 <strong>Windows tip — Python PATH</strong></summary>

If you installed Python from the **Microsoft Store**, `python` and `pip` are already on your PATH.\
If you installed from **python.org**, make sure you checked **"Add Python to PATH"** during setup.
</details>

---

## 📦 Install

### Recommended — pipx (isolated + globally on PATH)

```bash
pipx install mindmark
```

<details>
<summary>Don't have pipx?</summary>

```bash
pip install --user pipx && pipx ensurepath    # then restart your terminal
```

Or on macOS with Homebrew: `brew install pipx`
</details>

<details>
<summary>Alternative — pip with a virtual environment</summary>

**macOS / Linux:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mindmark
```

**Windows (PowerShell):**

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install mindmark
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv && .venv\Scripts\activate.bat
pip install mindmark
```
</details>

<details>
<summary>Editable install for development</summary>

```bash
git clone https://github.com/sukanth/mindmark.git
cd mindmark
pip install -e .[dev]
```
</details>

---

## ⚡ Quick Start

### 1️⃣ Sync your bookmarks (no export needed!)

```bash
mindmark sync
```

That's it — mindmark auto-detects your installed browsers, reads their bookmark files directly, and builds a searchable index. **No manual export required.**

> First run downloads the embedding model (~130 MB) and caches it locally. Every run after that is instant and fully offline.

<details>
<summary>💡 <strong>See which browsers were detected</strong></summary>

```bash
mindmark sync --list-browsers
```

Example output:

```
Browser      Profile                  Path
-------      -------                  ----
Chrome       Default                  ~/Library/.../Google/Chrome/Default/Bookmarks
Chrome       Profile 3                ~/Library/.../Google/Chrome/Profile 3/Bookmarks
Edge         Default                  ~/Library/.../Microsoft Edge/Default/Bookmarks
```

</details>

<details>
<summary>💡 <strong>Sync a specific browser only</strong></summary>

```bash
mindmark sync --browser chrome
mindmark sync --browser firefox
mindmark sync --browser edge
mindmark sync --browser brave
```

</details>

<details>
<summary>💡 <strong>Alternative — import from an exported HTML file</strong></summary>

If you prefer the manual export workflow, or need to import bookmarks from an unsupported browser:

| Browser | How to export |
|---|---|
| **Edge** | `edge://favorites` → `⋯` → **Export favorites** → save as HTML |
| **Chrome** | `chrome://bookmarks` → `⋮` → **Export bookmarks** → save as HTML |
| **Firefox** | `Ctrl+Shift+O` (`Cmd+Shift+O` on macOS) → **Import and Backup** → **Export Bookmarks to HTML** |

```bash
# macOS / Linux
mindmark index ~/Downloads/bookmarks.html

# Windows (PowerShell)
mindmark index "$env:USERPROFILE\Downloads\bookmarks.html"
```

</details>

### 2️⃣ Search in natural language

<p align="center">
  <img src="assets/mindmark-find.gif" alt="mindmark find demo" width="800" />
</p>

```bash
mindmark find "python async tutorial"
mindmark find "react hooks best practices" -k 5
mindmark find "helm chart examples" --domain github.com
mindmark find "docker compose setup" --folder devops
```

### 3️⃣ Open a result directly

```bash
mindmark open "k8s cheat sheet"           # opens the best match
mindmark find "docker setup" --open 2     # opens result #2 from the list
```

<details>
<summary>💡 <strong>Tip — create a short alias</strong></summary>

**macOS / Linux** — add to `~/.bashrc` or `~/.zshrc`:

```bash
alias mm='mindmark open'
mm "docker setup"
```

**Windows** — add to your PowerShell `$PROFILE`:

```powershell
Set-Alias mm mindmark
mm open "docker setup"
```
</details>

### 4️⃣ JSON output for scripting

Pipe results into **fzf**, **jq**, **Alfred**, **Raycast**, **PowerToys Run**, or any tool that accepts JSON:

```bash
# macOS / Linux
mindmark find "istio service mesh" --json | jq '.[].url'

# Windows (PowerShell)
mindmark find "istio service mesh" --json | ConvertFrom-Json | ForEach-Object { $_.url }
```

---

## 📖 Usage

### Syncing

`mindmark sync` reads bookmarks directly from your browser data directories. It's **incremental** — only new or changed bookmarks are re-embedded, making re-syncs near-instant.

```bash
mindmark sync                         # sync all detected browsers
mindmark sync --browser chrome        # sync only Chrome
mindmark sync --browser firefox       # sync only Firefox
mindmark sync --list-browsers         # list detected browsers and profiles
```

When you add new bookmarks in your browser, just run `mindmark sync` again — it will pick up only the changes.

> 💡 **Note:** If you change the embedding model with `--model`, all bookmarks will be re-embedded on the next sync. Browser names are case-insensitive (e.g., `--browser Chrome` and `--browser chrome` both work).

### Filters

Narrow down results without changing your query:

```bash
mindmark find "useful tools" --domain github.com     # only github.com results
mindmark find "useful tools" --folder work/kusto      # only bookmarks in matching folders
mindmark find "useful tools" -k 20                    # return top 20 instead of 10
```

### Re-indexing

For the `sync` workflow, just rerun `mindmark sync`. It's incremental — only changed bookmarks are re-embedded.

For the `index` workflow, rerun `mindmark index <file>`. It clears and rebuilds the index. The model is cached, so re-indexing 800+ bookmarks takes only seconds.

### Drop the local index

Use `drop-index` to remove the local SQLite index database when you want a clean slate.

```bash
mindmark drop-index               # asks for confirmation
mindmark drop-index --yes         # skip confirmation
mindmark drop-index --db /path/to/index.db
```

### Validate stale links

Use `validate` to probe all indexed HTTP(S) bookmark URLs and identify stale ones (HTTP 4xx/5xx or unreachable hosts). Mindmark will report which bookmarks may be stale and where they are located, but does not modify them. You can then manually remove stale bookmarks from your browser or re-index after cleaning them up.

```bash
mindmark validate                     # identify all stale bookmarks
mindmark validate --timeout 5         # per-request timeout in seconds (default 8)
mindmark validate --workers 32        # parallel URL checks (default 16)
```

Non-HTTP URLs (for example `file:` or browser-internal URLs) are skipped and not checked.

### Swap the embedding model

```bash
mindmark sync --model BAAI/bge-small-en-v1.5                # default, 384-dim
mindmark sync --model sentence-transformers/all-MiniLM-L6-v2
mindmark sync --model BAAI/bge-base-en-v1.5                 # 768-dim, higher quality
```

The `--model` flag also works with `mindmark index`. Switching models triggers a full re-embed automatically. See the [fastembed supported models list](https://qdrant.github.io/fastembed/examples/Supported_Models/).

---

## 🧠 How It Works

```
Browser data files                              "python async tutorial"
(Chrome JSON / Firefox SQLite)                            │
       │                                                  │
       ▼                                                  ▼
  ┌────────────┐  ┌──────────┐  ┌──────────┐     ┌──────────┐
  │  Detect &  │─▶│  Embed   │─▶│  Store   │     │  Embed   │
  │   Parse    │  │ (ONNX)   │  │ (SQLite) │◀────│  query   │
  └────────────┘  └──────────┘  └──────────┘     └──────────┘
                      ▲               │                │
                      │               ▼                ▼
                 only new/      ┌──────────────────────────┐
                 changed        │  Dot-product similarity  │
                 bookmarks      │   → top-K results        │
                                └──────────────────────────┘
```

1. **Detect** — Auto-discover installed browsers (Chrome, Edge, Brave, Firefox) and their profiles across macOS, Linux, and Windows.
2. **Parse** — Read bookmark files natively: Chromium JSON format or Firefox `places.sqlite`. No export step needed.
3. **Diff** — Hash each bookmark's content and compare against the existing index. Only new or changed bookmarks proceed to embedding.
4. **Embed** — Each bookmark becomes a rich text string (`title | folder | domain | path`) and is passed through a BGE/MiniLM ONNX model. Vectors are L2-normalized.
5. **Store** — Vectors live as `float32` blobs in a single SQLite file. A `bookmark_sources` table tracks which browser contributed each bookmark, so multi-browser syncs don't conflict.
6. **Search** — Encode the query, compute dot products against all vectors, return the top-K.

---

## 🗂️ Storage Layout

| What | macOS / Linux | Windows | Override |
|---|---|---|---|
| Index database | `~/.mindmark/index.db` | `%LOCALAPPDATA%\mindmark\index.db` | `--db` flag or `MINDMARK_DB` env var |
| Home directory | `~/.mindmark/` | `%LOCALAPPDATA%\mindmark\` | `MINDMARK_HOME` env var |
| Embedding model | `~/.cache/fastembed/` | `%LOCALAPPDATA%\fastembed\` | Managed by fastembed |

---

## 🗑️ Uninstall

```bash
pipx uninstall mindmark    # if installed with pipx
pip uninstall mindmark      # if installed with pip
```

<details>
<summary>Remove stored data (optional)</summary>

The index and cached model are stored outside the package:

**macOS / Linux:**

```bash
rm -rf ~/.mindmark              # index database
rm -rf ~/.cache/fastembed        # cached embedding model (~130 MB)
```

**Windows (PowerShell):**

```powershell
Remove-Item -Recurse "$env:LOCALAPPDATA\mindmark"     # index database
Remove-Item -Recurse "$env:LOCALAPPDATA\fastembed"     # cached embedding model
```

> If you set a custom `MINDMARK_HOME`, remove that directory instead.
</details>

---

## 🛠️ Development

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

```bash
git clone https://github.com/sukanth/mindmark.git
cd mindmark
pip install -e .[dev]
pytest -q
```

<details>
<summary>Publishing to PyPI</summary>

### First-time setup

1. Create an account at [pypi.org](https://pypi.org/account/register/)
2. Generate an API token at [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)
3. Install build tools: `pip install build twine`

### Test on TestPyPI first (recommended)

```bash
python -m build
python -m twine upload --repository testpypi dist/*
pipx install --index-url https://test.pypi.org/simple/ mindmark
```

### Publish to PyPI

```bash
python -m build
python -m twine upload dist/*
```

Use `__token__` as the username when prompted.
</details>

<details>
<summary>Alternative distribution methods</summary>

### GitHub release

```bash
python -m build
gh release create v0.1.0 dist/*
# Users install:
pipx install https://github.com/sukanth/mindmark/releases/download/v0.1.0/mindmark-0.1.0-py3-none-any.whl
```

### Standalone executable (no Python required)

```bash
pip install pyinstaller
pyinstaller --onefile -n mindmark -p src src/mindmark/__main__.py
# Creates: dist/mindmark (macOS/Linux) or dist/mindmark.exe (Windows)
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
ENTRYPOINT ["mindmark"]
```

```bash
docker build -t mindmark .

# Sync from browser bookmarks (mount browser data directories)
# Note: browser data paths vary — this example is for macOS Chrome
docker run --rm \
    -v $HOME/.mindmark:/root/.mindmark \
    -v "$HOME/Library/Application Support/Google/Chrome":/chrome:ro \
    mindmark sync

# Or import from an exported HTML file
docker run --rm -v $HOME/.mindmark:/root/.mindmark \
    -v $HOME/Downloads:/downloads mindmark \
    index /downloads/bookmarks.html
```
</details>

---

## 📄 License

MIT — see [LICENSE](LICENSE).
