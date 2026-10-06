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
├── .python-version                 # optionales Python-Tooling-Setup
├── docs/
│   ├── handbuch.pdf                # Wissensbasis Deutsch (PDF)
│   ├── handbuch.txt                # Wissensbasis Deutsch (Text)
│   ├── handbook.pdf                # Wissensbasis Englisch (PDF)
│   └── handbook.txt                # Wissensbasis Englisch (Text)
├── documents/                      # Extrakt der PDFs (nicht versioniert)
├── scratch/                        # temporaere oder explorative Dateien
├── htmlcov/                        # Coverage-HTML-Berichte (erzeugt)
└── src/
    └── rag_project/
        ├── __init__.py             # Paket-API: main, Settings, Provider, RAG-Pipeline
        ├── app/
        │   ├── __init__.py
        │   ├── main.py             # Projekt-CLI (--check, --models, --show-config)
        │   ├── main_chatbot.py     # Terminal-Chatbot
        │   ├── main_chunking.py    # CLI: chunken
        │   ├── main_cleaning.py    # CLI: bereinigen
        │   ├── main_loading.py     # CLI: laden
        │   ├── rag_app.py          # CLI der RAG-Anwendung
        │   ├── rag_lib.py          # Index, Retriever, Kette, Sprachwahl
        │   └── rag_utils.py        # format_text, print_wrapped
        ├── core/
        │   ├── __init__.py
        │   ├── config.py           # Settings, API-Keys, Pfade
        │   ├── models.py           # Modellkatalog je Anbieter
        │   └── paths.py            # Quelldokumente aufloesen
        ├── llm/
        │   ├── __init__.py
        │   └── providers.py        # LLM-Client bauen, Key entschluesseln
        ├── pipeline/
        │   ├── __init__.py
        │   ├── chunking.py         # split_documents
        │   ├── cleaning.py         # clean_documents
        │   └── loading.py          # load_documents
        └── utils/
            ├── __init__.py
            └── math_utils.py       # add, mult

tests/
├── __init__.py
├── test_config_and_models.py       # Settings, Katalog, Provider-Aufloesung
├── test_main.py                    # Einstieg ohne Argumente
├── test_main_chatbot.py            # Client-Aufbau und System-Reset
├── test_main_cleaning.py           # Parser der Cleaning-CLI
├── test_main_loading.py            # Schreiben geladener Dokumente
├── test_paths.py                   # Pfadaufloesung
├── test_remote_pdf_loader.py       # PDF von einer Adresse
└── test_wikipedia_loader.py        # Wikipedia-Loader
```

## Source-Ordner

Der Quellcode befindet sich im Verzeichnis `src/`. Dort liegt das Paket `rag_project`, das als Python-Paket importiert werden kann. Es ist in fünf Schichten gegliedert:

- `core/` — Konfiguration (`config.py`), Modellkatalog (`models.py`) und Pfadaufloesung (`paths.py`)
- `llm/` — Provider-Schicht: aus Settings und Katalog einen fertigen Client bauen
- `pipeline/` — die drei Vorverarbeitungsstufen: laden, bereinigen, chunken
- `app/` — CLIs und Anwendungslogik, einschliesslich der RAG-Anwendung
- `utils/` — kleine Hilfsfunktionen

Alle anderen Module holen ihre Einstellungen ausschliesslich ueber `get_settings()` aus `core/config.py` — sie lesen niemals selbst aus `os.environ`.

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

Das Projekt definiert mehrere Einstiegspunkte in `pyproject.toml`:

| Befehl | Ziel | Zweck |
| --- | --- | --- |
| `uv run rag-project` | `app/main.py` | Konfiguration prüfen, Modellkatalog zeigen |
| `uv run chatbot` | `app/main_chatbot.py` | Terminal-Chatbot mit Verlauf |
| `uv run loading` | `app/main_loading.py` | Dokumente laden |
| `uv run cleaning` | `app/main_cleaning.py` | Text bereinigen |
| `uv run chunking` | `app/main_chunking.py` | Dokumente chunken |
| `uv run rag` | `app/rag_app.py` | vollständige RAG-Anwendung |

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
| `--format pdf\|txt` | Quellformat des Handbuchs | `pdf` |
| `--mode dense\|hybrid` | Suchmodus | `dense` |
| `--k N` | Zahl der Treffer | `3` |
| `--chunk-size N` | Zielgroesse eines Chunks | `800` |
| `--chunk-overlap N` | Ueberlappung zweier Chunks | `100` |
| `--strategy NAME` | recursive, character oder token | `recursive` |
| `--embedding-model NAME` | Modell der Indexierung | text-embedding-3-small |
| `--temperature N` | Temperatur des antwortenden Modells | `0.0` |

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
- Neue Module gehören in ihren Fachordner unter `src/rag_project/`, nicht in die Paketwurzel.

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
uv run python -m rag_project.app.main_loading --wikipedia "Artificial intelligence" --wiki-lang en --wiki-max-docs 2 --allow-remote --write docs/update/artificial_intelligence.txt
uv run python -m rag_project.app.main_cleaning --wikipedia "Artificial intelligence" --wiki-lang en --wiki-max-docs 2 --allow-remote
uv run python -m rag_project.app.main_chunking --wikipedia "Artificial intelligence" --wiki-lang en --wiki-max-docs 2 --allow-remote
```

## Dokumentation

Die Projektdokumentation liegt als PDF-Sammlung vor:

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

## Lizenz

Das Projekt ist derzeit ohne spezielle Lizenzangabe konfiguriert. Bitte bei Bedarf eine passende Open-Source-Lizenz ergänzen.
