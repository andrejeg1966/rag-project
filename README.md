# RAG Projects

Ein einfaches Python-Projekt für Retrieval-Augmented Generation (RAG)-Experimente und Beispielimplementierungen.

## Projektstruktur

```text
RAG-projects/
├── .env                            # lokale Umgebungsvariablen (Zugangsdaten)
├── .gitignore                      # Git-Ignore: .env, documents/, scratch/
├── README.md                       # Projektbeschreibung
├── pyproject.toml                  # Abhaengigkeiten und CLI-Einstiege
├── uv.lock                         # Lock-Datei fuer die Abhaengigkeiten
├── docs/
│   ├── handbuch.pdf                # Wissensbasis Deutsch (PDF)
│   ├── handbuch.txt                # Wissensbasis Deutsch (Text)
│   ├── handbook.pdf                # Wissensbasis Englisch (PDF)
│   └── handbook.txt                # Wissensbasis Englisch (Text)
├── documents/                      # Extrakt des PDFs (nicht versioniert)
├── scratch/                        # temporaere oder explorative Dateien
├── src/
│   └── rag_project/
│       ├── __init__.py             # Paket-API: main, RagSystem, create_rag_system
│       ├── app/
│       │   ├── __init__.py
│       │   ├── rag_app.py          # CLI der RAG-Anwendung
│       │   ├── rag_lib.py          # Index, Retriever, Kette, Sprachwahl
│       │   ├── rag_utils.py        # format_text, print_wrapped
│       │   ├── main.py             # Projekt-CLI (--check, --models, --show-config)
│       │   ├── main_chatbot.py     # Terminal-Chatbot
│       │   ├── main_chunking.py    # CLI: chunken
│       │   ├── main_cleaning.py    # CLI: bereinigen
│       │   └── main_loading.py     # CLI: laden
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py           # Settings, API-Keys, Pfade
│       │   ├── models.py           # Modellkatalog je Anbieter
│       │   └── paths.py            # Quelldokumente aufloesen
│       ├── llm/
│       │   ├── __init__.py
│       │   └── providers.py        # LLM-Client bauen, Key entschluesseln
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── chunking.py         # split_documents
│       │   ├── cleaning.py         # clean_documents
│       │   └── loading.py          # load_documents
│       ├── utils/
│       │   ├── __init__.py
│       │   └── math_utils.py
│       └── ...
├── tests/
│   ├── __init__.py
│   ├── test_config_and_models.py
│   ├── test_main.py
│   ├── test_main_chatbot.py
│   └── test_paths.py
└── .python-version                 # optionales Python-Tooling-Setup
```

## Source-Ordner

Der Quellcode befindet sich im Verzeichnis `src/`. Dort liegt das Paket `rag_project`, das als Python-Paket importiert werden kann.

Typische Nutzung:

- Implementierung von RAG-Komponenten
- Datenvorverarbeitung und Suchlogik
- Verbindung zu Vektordatenbanken oder LLM-APIs
- Hilfsfunktionen für Dokumentenverarbeitung

## Test-Ordner

Der Ordner `tests/` enthält automatisierte Tests. Standardmäßig werden dort Dateien mit Namen wie `test_*.py` erkannt.

Mit Pytest können die Tests so ausgeführt werden:

```bash
uv run pytest
```

oder mit aktivierter Umgebung:

```bash
pytest
```

## Projekt-Setup

1. Repository klonen
2. Abhängigkeiten installieren:

```bash
uv sync
```

Für das Development-Setup mit Testwerkzeugen, z. B. `pytest`, `coverage` und `pytest-cov`, nutzt man:

```bash
uv sync --group dev
```

3. Projekt-Umgebung aktivieren (optional):

```bash
source .venv/bin/activate
```

Auf Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Ausführen des Projekts

Das Projekt definiert einen Einstiegspunkt in `pyproject.toml`:

```bash
uv run rag-project
```

Zur RAG-Anwendung mit Konfigurationsausgabe:

```bash
uv run rag --config
```

Oder direkt als Python-Modul, falls erforderlich:

```bash
uv run python -m rag_project
```

## RAG-Anwendung

Die RAG-Schicht liegt in `app/rag_lib.py` (Bibliothek) und `app/rag_app.py` (Bedienung).
Sie baut auf den drei Vorverarbeitungsstufen des Projekts auf:

```text
laden                pipeline/loading.py   load_documents
bereinigen           pipeline/cleaning.py  clean_documents
chunken              pipeline/chunking.py  split_documents
indexieren           app/rag_lib.py        build_vectorstore
antworten            app/rag_lib.py        build_rag_chain
```

Die ersten drei Stufen brauchen weder Netz noch Zugangsdaten. Der Index liegt
im Arbeitsspeicher und wird bei jedem Lauf neu aufgebaut.

### Einstieg

```bash
uv run rag                              # Uebersicht der Pipeline
uv run rag --config                     # aktive Einstellungen
uv run rag --load                       # Stufe 1: Ladebericht
uv run rag --clean                      # Stufe 2: Reinigungsbilanz
uv run rag --chunks                     # Stufe 3: Chunking-Kennzahlen
uv run rag --preview                    # die ersten Chunks mit Herkunft
uv run rag --write docs/chunks.md        # Chunks als Markdown ablegen
uv run rag --retrieve "FRAGE"            # Treffer ohne Modellaufruf
uv run rag --ask "FRAGE"                 # eine Frage beantworten
uv run rag --ask "FRAGE" --show-hits     # Antwort mit Belegstellen
uv run rag --compare                     # beide Suchmodi vergleichen
```

### Wichtige Optionen

| Option | Wirkung | Standard |
| --- | --- | --- |
| `--language de\|en` | Sprache der Wissensbasis | `de` |
| `--mode dense\|hybrid` | Suchmodus | `dense` |
| `--k N` | Zahl der Treffer | `3` |
| `--chunk-size N` | Zielgroesse eines Chunks | `800` |
| `--chunk-overlap N` | Ueberlappung zweier Chunks | `100` |
| `--strategy NAME` | recursive, character oder token | `recursive` |
| `--format pdf\|txt` | Quellformat | `pdf` |
| `--embedding-model NAME` | Modell der Indexierung | text-embedding-3-small |

### Sprachfassungen

Deutsch und Englisch liegen als getrennte Wissensbasen vor, jede mit eigenem
Quelldokument und eigenem System-Prompt. Deutsch ist die Werkseinstellung.

### Hinweis zur englischen Fassung

Die englischen Kapitelueberschriften tragen keine Nummern. Deshalb bleiben
`chapter` und `section` in den Chunk-Metadaten dort leer, waehrend sie in der
deutschen Fassung gefuellt sind.

## Entwicklungshinweise

- Keine großen Änderungen im Basis-Setup ohne die Abhängigkeiten zu überprüfen.
- Neue Funktionen sollten mit passenden Tests abgesichert werden.
- Pytest sollte regelmäßig lokal ausgeführt werden, bevor neue Änderungen übernommen werden.

## Pytest Coverage

Für die Testabdeckung kann `pytest-cov` verwendet werden. Das Paket im Projekt liegt unter `src/rag_project`, deshalb sollte der Coverage-Pfad genau so gesetzt werden:

```bash
uv run pytest --cov=src/rag_project --cov-report=term-missing --cov-report=html -q
```

Damit werden:

- eine kurze Coverage-Ausgabe im Terminal erzeugt
- ein HTML-Bericht unter `htmlcov/` erstellt

## Beispiel-Workflow

```bash
uv sync
uv run pytest
uv run pytest --cov=src/rag_project --cov-report=term-missing --cov-report=html -q
uv run rag-project
uv run rag
uv run rag --ask "Wie lange ist der Link zum Zuruecksetzen gueltig?"
```

## PDF per URL laden

PDF-URLs werden mit `--url` geladen. Nach erfolgreichem Parsen wird die PDF im Projektordner `docs/` gespeichert. Gibt es dort bereits eine andere Datei mit demselben Namen, wird ein nummerierter Dateiname verwendet; eine identische PDF wird wiederverwendet.

```powershell
$pdfUrl = "https://archive.org/download/alicesadventures1901carr/alicesadventures1901carr.pdf"
uv run python -m rag_project.app.main_loading --url $pdfUrl --allow-remote
uv run python -m rag_project.app.main_cleaning --url $pdfUrl --allow-remote
uv run python -m rag_project.app.main_chunking --url $pdfUrl --allow-remote
```

Jeder Befehl verarbeitet die URL erneut; die heruntergeladene PDF wird unter demselben Dateinamen in `docs/` wiederverwendet.

## Lokale PDF bereinigen und chunken

Eine lokale PDF-Datei kann zuerst bereinigt und anschließend gechunked werden:

```bash
uv run python -m rag_project.app.main_cleaning docs/alicesadventures1901carr.pdf --write docs/alice_cleaned.txt
uv run python -m rag_project.app.main_chunking docs/alice_cleaned.txt --write docs/alice_chunked.txt
```

## Wikipedia-CLI-Beispiele

Die folgenden Befehle laden, bereinigen und chunken Wikipedia-Artikel sowohl auf Deutsch als auch auf Englisch.

### Deutsch

```bash
uv run python -m rag_project.app.main_loading --wikipedia "Künstliche Intelligenz" --wiki-lang de --wiki-max-docs 2 --allow-remote
uv run python -m rag_project.app.main_cleaning --wikipedia "Künstliche Intelligenz" --wiki-lang de --wiki-max-docs 2 --allow-remote
uv run python -m rag_project.app.main_chunking --wikipedia "Künstliche Intelligenz" --wiki-lang de --wiki-max-docs 2 --allow-remote
```

### Englisch

```bash
uv run python -m rag_project.app.main_loading --wikipedia "Artificial intelligence" --wiki-lang en --wiki-max-docs 2 --allow-remote --write docs/artificial_intelligence.txt
uv run python -m rag_project.app.main_cleaning --wikipedia "Artificial intelligence" --wiki-lang en --wiki-max-docs 2 --allow-remote
uv run python -m rag_project.app.main_chunking --wikipedia "Artificial intelligence" --wiki-lang en --wiki-max-docs 2 --allow-remote
```

## Lizenz

Das Projekt ist derzeit ohne spezielle Lizenzangabe konfiguriert. Bitte bei Bedarf eine passende Open-Source-Lizenz ergänzen.
