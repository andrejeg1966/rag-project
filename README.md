# RAG Projects

Ein einfaches Python-Projekt für Retrieval-Augmented Generation (RAG)-Experimente und Beispielimplementierungen.

## Projektstruktur

```text
RAG-projects/
├── .venv/                          # virtuelle Python-Umgebung
├── docs/
│   └── handbuch.txt               # Projekt-/Nutzungsdokumentation
├── documents/                     # Eingabedokumente oder Beispiel-Daten
├── htmlcov/                       # Coverage-HTML-Berichte
│   ├── index.html
│   ├── status.json
│   └── ...
├── scratch/
│   └── clean.txt                  # temporäre oder explorative Dateien
├── src/
│   └── rag_project/
│       ├── __init__.py            # Paketinitialisierung
│       ├── chunking.py            # Chunking-Logik
│       ├── cleaning.py            # Bereinigung und Vorverarbeitung
│       ├── config.py              # Konfiguration und Einstellungen
│       ├── loading.py             # Laden von Daten/Modellen
│       ├── main.py                # Einstiegspunkt des Projekts
│       ├── main_chatbot.py        # Chatbot-Skripte
│       ├── main_chunking.py      # Chunking-Utilities
│       ├── main_cleaning.py      # Bereinigungs-Utilities
│       ├── main_loading.py        # Lade-Utilities
│       ├── math_utils.py          # mathematische Hilfsfunktionen
│       ├── models.py              # Datenmodelle
│       ├── paths.py               # Pfad- und Dateistrukturlogik
│       ├── providers.py           # Provider-/API-Integrationen
│       └── ...
├── tests/
│   ├── __init__.py
│   ├── test_config_and_models.py
│   ├── test_main_chatbot.py
│   ├── test_main.py
│   ├── test_paths.py
│   └── ...
├── coverage.json                  # Coverage-Metadaten
├── pyproject.toml                 # Projektkonfiguration und Abhängigkeiten
├── README.md                      # Projektbeschreibung
├── uv.lock                        # Lock-Datei für die Abhängigkeitsverwaltung
├── .gitignore                     # Git-Ignore-Dateien
└── .python-version                # optionales Python-Tooling-Setup
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

Oder direkt als Python-Modul, falls erforderlich:

```bash
uv run python -m rag_project
```

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
```

## Lizenz

Das Projekt ist derzeit ohne spezielle Lizenzangabe konfiguriert. Bitte bei Bedarf eine passende Open-Source-Lizenz ergänzen.
