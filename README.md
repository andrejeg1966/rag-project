# RAG Projects

## 1. Projektstruktur

```text
RAG-projects/
├── .env                            # lokale Umgebungsvariablen (Zugangsdaten)
├── .gitignore                      # Git-Ignore: .env, documents/, scratch/
├── README.md                       # Projektbeschreibung
├── pyproject.toml                  # Abhaengigkeiten und CLI-Einstiege
├── uv.lock                         # Lock-Datei fuer die Abhaengigkeiten
├── .python-version                 # optionales Python-Tooling-Setup
├── docs/
│   ├── handbuch.pdf / handbuch.txt # Wissensbasis Deutsch
│   └── handbook.pdf / handbook.txt # Wissensbasis Englisch
├── documents/                      # Extrakt der PDFs (nicht versioniert)
├── scratch/                        # temporaere oder explorative Dateien
├── htmlcov/                        # Coverage-HTML-Berichte (erzeugt)
├── src/
│   └── rag_project/
│       ├── __init__.py             # Paket-API: main, Settings, Provider, RAG-Pipeline
│       ├── app/
│       │   ├── main.py             # Projekt-CLI (--check, --models, --show-config)
│       │   ├── main_chatbot.py     # Terminal-Chatbot
│       │   ├── main_chunking.py    # CLI: chunken
│       │   ├── main_cleaning.py    # CLI: bereinigen
│       │   ├── main_loading.py     # CLI: laden
│       │   ├── rag_app.py          # CLI der RAG-Anwendung
│       │   ├── rag_eval.py         # RAG-Evaluation und LLM-Judges
│       │   ├── rag_lib.py          # Index, Retriever, Kette, Sprachwahl
│       │   ├── ui_chatbot.py       # Gradio-Web-Chatbot mit Streaming
│       │   └── ui_rag.py           # Gradio-Web-RAG-Chatbot
│       ├── core/
│       │   ├── config.py           # Settings, API-Keys, Pfade
│       │   ├── models.py           # Modellkatalog je Anbieter
│       │   └── paths.py            # Quelldokumente aufloesen
│       ├── llm/
│       │   └── providers.py        # LLM-Client bauen, Key entschluesseln
│       ├── pipeline/
│       │   ├── chunking.py         # Dokumente aufteilen
│       │   ├── cleaning.py         # Dokumente bereinigen
│       │   ├── csv_load.py         # CSV laden und bereinigen
│       │   ├── loading.py          # generisches Dokument laden
│       │   ├── markdown_load.py    # Markdown laden
│       │   ├── web_load.py         # Webseiten per URL laden
│       │   └── wikipedia_load.py   # Wikipedia-Artikel laden
│       └── utils/
│           └── math_utils.py       # add, mult
└── tests/
    ├── test_config_and_models.py   # Settings, Katalog, Provider-Aufloesung
    ├── test_csv_markdown_load.py   # CSV- und Markdown-Loader
    ├── test_main.py                # Einstieg ohne Argumente
    ├── test_main_chatbot.py        # Client-Aufbau und System-Reset
    ├── test_main_cleaning.py       # Parser der Cleaning-CLI
    ├── test_main_loading.py        # Schreiben geladener Dokumente
    ├── test_paths.py               # Pfadaufloesung
    ├── test_remote_pdf_loader.py   # PDF von einer Adresse
    ├── test_ui_chatbot.py          # Gradio-History- und Modelltests
    ├── test_ui_rag.py              # RAG-Weboberflaeche und Abfragen
    └── test_wikipedia_loader.py    # Wikipedia-Loader
```

## 2. Allgemeines

Ein einfaches Python-Projekt für Retrieval-Augmented Generation (RAG)-Experimente und Beispielimplementierungen.

Typische Nutzung:

- Implementierung von RAG-Komponenten
- Datenvorverarbeitung und Suchlogik
- Verbindung zu Vektordatenbanken oder LLM-APIs
- Hilfsfunktionen für Dokumentenverarbeitung

### Projekt-Setup

```bash
git clone <repository-url>
uv sync                 # Abhängigkeiten installieren
uv sync --group dev     # inkl. Testwerkzeuge (pytest, coverage, pytest-cov)
```

Umgebung aktivieren (optional):

```bash
source .venv/bin/activate          # Linux/macOS
```

```powershell
.\.venv\Scripts\Activate.ps1       # Windows PowerShell
```

Zugangsdaten der LLM-Anbieter gehören in die `.env`-Datei bzw. die aktive Umgebung. Prüfen:

```bash
uv run rag-project --check
```

### Einstiegspunkte (`pyproject.toml`)

| Befehl | Ziel | Zweck |
| --- | --- | --- |
| `uv run rag-project` | `app/main.py` | Konfiguration prüfen, Modellkatalog zeigen |
| `uv run chatbot` | `app/main_chatbot.py` | Terminal-Chatbot mit Verlauf |
| `uv run loading` | `app/main_loading.py` | Dokumente laden |
| `uv run cleaning` | `app/main_cleaning.py` | Text bereinigen |
| `uv run chunking` | `app/main_chunking.py` | Dokumente chunken |
| `uv run csv-load` | `pipeline/csv_load.py` | CSV laden, bereinigen, chunken, speichern |
| `uv run markdown-load` | `pipeline/markdown_load.py` | Markdown laden, bereinigen, chunken, speichern |
| `uv run wikipedia-load` | `pipeline/wikipedia_load.py` | Wikipedia laden, bereinigen, chunken, speichern |
| `uv run web-load` | `pipeline/web_load.py` | Webseite laden, bereinigen, chunken, speichern |
| `uv run rag` | `app/rag_app.py` | vollständige RAG-Anwendung |
| `uv run ui-chatbot` | `app/ui_chatbot.py` | Web-Chatbot mit Gradio |
| `uv run ui-rag` | `app/ui_rag.py` | Web-RAG-Chatbot mit Gradio |

### Entwicklungshinweise

- Keine großen Änderungen im Basis-Setup ohne die Abhängigkeiten zu überprüfen.
- Neue Funktionen mit passenden Tests absichern.
- Pytest regelmäßig lokal ausführen.
- Neue Module gehören in ihren Fachordner unter `src/rag_project/`, nicht in die Paketwurzel.

### Dokumentation

- `00-rag-project-architektur.pdf` — Aufbau, Schichten und Datenfluss
- `00-rag-project-betrieb.pdf` — Installation, Kommandos und Ablaeufe
- `01-projekt-validierung.pdf` — Konfigurationspruefung und Testsuite
- `02-cli-main-chatbot.pdf` — Terminal-Chatbot
- `03-cli-main-loading.pdf` — Dokumente laden
- `04-cli-main-cleaning.pdf` — Textbereinigung
- `05-cli-main-chunking.pdf` — Chunking und Kennzahlen
- `06-modellaufloesung.pdf` — Modellkatalog und Kennungen
- `07-provideraufloesung.pdf` — Provider-Schicht und Client
- `08-generierungsparameter.pdf` — Einstellungen und ihre Wirkung
- `09-rag-dokumentation.pdf` — Index, Suche und Kette

### Lizenz

Das Projekt ist derzeit ohne spezielle Lizenzangabe konfiguriert. Bitte bei Bedarf eine passende Open-Source-Lizenz ergänzen.

## 3. Beschreibung: Projekt, `src` und `tests`

### Projekt

Aus Dokumenten wird eine Wissensbasis aufgebaut (laden → bereinigen → chunken → indexieren). Darauf beantworten LLM-Anbieter Fragen über Terminal oder Web-Oberfläche.

### Source-Ordner (`src/`)

Dort liegt das Paket `rag_project`, gegliedert in fünf Schichten:

- `core/` — Konfiguration (`config.py`), Modellkatalog (`models.py`) und Pfadaufloesung (`paths.py`)
- `llm/` — Provider-Schicht: aus Settings und Katalog einen fertigen Client bauen
- `pipeline/` — Vorverarbeitung: laden, bereinigen, chunken
- `app/` — CLIs und Anwendungslogik, einschliesslich der RAG-Anwendung
- `utils/` — kleine Hilfsfunktionen

Alle Module holen ihre Einstellungen ausschliesslich über `get_settings()` aus `core/config.py` — sie lesen niemals selbst aus `os.environ`.

### Test-Ordner (`tests/`)

Dateien mit Namen `test_*.py` werden automatisch erkannt (siehe nächster Abschnitt).

## 4. Pytest und Coverage

```bash
uv run pytest
pytest                                   # mit aktivierter Umgebung
uv run pytest --cov=src/rag_project --cov-report=term-missing --cov-report=html -q
```

Der Coverage-Pfad muss `src/rag_project` sein. Ergebnis: kurze Ausgabe im Terminal und ein HTML-Bericht unter `htmlcov/`.

## 5. Terminal-Chatbot (`uv run chatbot`)

```bash
uv run chatbot
uv run rag-project --check      # Konfiguration prüfen
uv run python -m rag_project    # direkt als Modul
```

Der Chatbot nutzt Provider und Modell aus dem Katalog und führt einen Gesprächsverlauf.

## 6. RAG-Implementierung

Die RAG-Schicht liegt in `app/rag_lib.py` (Bibliothek) und `app/rag_app.py` (Bedienung):

```text
laden           pipeline/loading.py   load_documents
bereinigen      pipeline/cleaning.py  clean_documents
chunken         pipeline/chunking.py  split_documents
indexieren      app/rag_lib.py        build_vectorstore
antworten       app/rag_lib.py        build_rag_chain
```

Die ersten drei Stufen brauchen weder Netz noch Zugangsdaten. Der Index liegt im Arbeitsspeicher und wird bei jedem Lauf neu aufgebaut.

### Befehle

```bash
uv run rag                               # Uebersicht der Pipeline
uv run rag --config                      # aktive Einstellungen
uv run rag --load                        # Stufe 1: Ladebericht
uv run rag --clean                       # Stufe 2: Reinigungsbilanz
uv run rag --chunks                      # Stufe 3: Chunking-Kennzahlen
uv run rag --preview                     # die ersten Chunks mit Herkunft
uv run rag --write docs/chunks.md        # Chunks als Markdown ablegen
uv run rag --retrieve "FRAGE"            # Treffer ohne Modellaufruf
uv run rag --ask "FRAGE"                 # eine Frage beantworten
uv run rag --ask "FRAGE" --show-hits     # Antwort mit Belegstellen
uv run rag --compare                     # beide Suchmodi vergleichen
uv run rag --ask "Wie lange ist der Link zum Zuruecksetzen gueltig?"
```

### Optionen

| Option | Wirkung | Standard |
| --- | --- | --- |
| `--language de\|en` | Sprache der Wissensbasis | `de` |
| `--format pdf\|txt` | Quellformat des Handbuchs | `pdf` |
| `--mode dense\|hybrid` | Suchmodus | `dense` |
| `--k N` | Zahl der Treffer | `3` |
| `--chunk-size N` | Zielgroesse eines Chunks | `800` |
| `--chunk-overlap N` | Ueberlappung zweier Chunks | `100` |
| `--strategy NAME` | recursive, character oder token | `recursive` |
| `--embedding-model NAME` | Modell der Indexierung | text-embedding-3-small |
| `--temperature N` | Temperatur des antwortenden Modells | `0.0` |

### Sprachfassungen

Deutsch und Englisch liegen als getrennte Wissensbasen vor, jede mit eigenem Quelldokument und eigenem System-Prompt. Deutsch ist die Werkseinstellung. Die englischen Kapitelueberschriften tragen keine Nummern, daher bleiben `chapter` und `section` in den Chunk-Metadaten dort leer.

## 7. Gradio-Oberflächen

```bash
uv sync
uv run ui-chatbot                          # Web-Chatbot
uv run ui-rag                              # Web-RAG-Chatbot
uv run python -m rag_project.app.ui_chatbot
uv run python -m rag_project.app.ui_rag
```

Bedienung:

1. Modellanbieter und Modell in den Einstellungen auswählen.
2. Frage im Chatfenster eingeben.
3. Mit **Reset Chat** den Verlauf löschen.
4. Mit `Ctrl+C` im Terminal beenden.

## 8. Loader

Die allgemeinen Befehle `loading`, `cleaning` und `chunking` verarbeiten nur `.txt`, `.rst`, `.pdf` und Webseiten (`--url`). Wikipedia, CSV und Markdown laden ausschließlich `wikipedia-load`, `csv-load` und `markdown-load`.

### PDF (lokal und per URL)

PDF-URLs werden mit `--url` geladen und danach in `docs/` gespeichert. Bei Namenskonflikt wird nummeriert; eine identische PDF wird wiederverwendet.

```powershell
$pdfUrl = "https://archive.org/download/alicesadventures1901carr/alicesadventures1901carr.pdf"
uv run python -m rag_project.app.main_loading --url $pdfUrl --allow-remote
uv run python -m rag_project.app.main_cleaning --url $pdfUrl --allow-remote
uv run python -m rag_project.app.main_chunking --url $pdfUrl --allow-remote
```

Lokale PDF bereinigen und chunken:

```bash
uv run python -m rag_project.app.main_cleaning docs/alicesadventures1901carr.pdf --write docs/alice_cleaned.txt
uv run python -m rag_project.app.main_chunking docs/alice_cleaned.txt --write docs/alice_chunked.txt
```

### TXT

```bash
uv run loading docs/handbuch.txt
uv run cleaning docs/handbuch.txt
uv run chunking docs/handbuch.txt
```

### Webseiten (`WebBaseLoader`)

Die Adresse muss mit `http://` oder `https://` beginnen. Benötigt `beautifulsoup4`; User-Agent über `USER_AGENT`.

```bash
uv run web-load https://example.com
uv run web-load https://example.com --clean
uv run web-load https://example.com --clean --chunk
uv run web-load https://example.com --clean --chunk --write docs/web_chunks.md
```

### Wikipedia (`WikipediaLoader`)

Erster Parameter ist der Suchbegriff; `--lang` (Standard `de`), `--max-docs` (Standard 1). Cache unter `.cache/wikipedia/`, User-Agent über `WIKIPEDIA_USER_AGENT`.

```bash
uv run wikipedia-load "Künstliche Intelligenz"
uv run wikipedia-load "Künstliche Intelligenz" --lang de --max-docs 2 --clean --chunk
uv run wikipedia-load "Artificial intelligence" --lang en --max-docs 2 --write docs/update/artificial_intelligence.txt
uv run wikipedia-load "Artificial intelligence" --lang en --clean --chunk --write docs/ai_chunks.md
```

### CSV (`csv_load`)

Erster Parameter ist die Datei (oder eine `http(s)://`-Adresse), optional folgt die Encoding-Kodierung.

```bash
uv run csv-load docs/sales.csv
uv run csv-load docs/sales.csv utf-8
uv run csv-load docs/sales.csv --clean
uv run csv-load docs/sales.csv utf-8 --clean --chunk
uv run csv-load docs/sales.csv --clean --chunk --write docs/sales_chunks.md
```

### Markdown (`markdown_load`)

```bash
uv run markdown-load docs/ML_plan.md
uv run markdown-load docs/ML_plan.md utf-8 --clean --chunk
uv run markdown-load docs/ML_plan.md --clean --chunk --write docs/ML_plan_chunks.md
```

### Optionen `--clean`, `--chunk`, `--write`

- `--clean`: Bereinigung der geladenen Datensätze.
- `--chunk`: Zerlegung in überlappende Chunks; Statistik und Chunks werden ausgegeben.
- `--write DATEI` (auch `-write`): speichert das Ergebnis zusätzlich als UTF-8-Datei. Fehlende Ordner werden angelegt, vorhandene Dateien überschrieben. Ohne Pfad bricht der Befehl mit Usage-Meldung ab (Exit-Code 2).

Das Layout richtet sich nach der Dateiendung:

| Endung | Format bei `--chunk` | Trenner |
| --- | --- | --- |
| `.md`, `.markdown` | Markdown, jeder Chunk mit Überschrift `### Chunk N` | `---` |
| alle anderen (`.txt`, `.csv`) | Text, jeder Chunk mit Präfix `Chunk N:` | Leerzeile |

Ohne `--chunk` wird der Inhalt unverändert geschrieben. Auch `example.csv` enthält Text im Format `Spalte: Wert`.

## Beispiel-Workflow

```bash
uv sync
uv run pytest
uv run pytest --cov=src/rag_project --cov-report=term-missing --cov-report=html -q
uv run rag-project
uv run rag
uv run rag --ask "Wie lange ist der Link zum Zuruecksetzen gueltig?"
```
