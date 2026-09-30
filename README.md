# RAG Projects

Ein einfaches Python-Projekt für Retrieval-Augmented Generation (RAG)-Experimente und Beispielimplementierungen.

## Projektstruktur

```text
RAG-projects/
├── .venv/                  # virtuelle Python-Umgebung
├── src/
│   └── rag_project/
│       ├── __init__.py    # Projekt-Initialisierung und Einstiegspunkt
│       └── ...            # weitere Module und RAG-Komponenten
├── tests/
│   ├── __init__.py
│   └── test_main.py       # Basis-Tests für das Projekt
├── pyproject.toml         # Projektkonfiguration, Abhängigkeiten und Pytest-Setup
├── README.md              # Projektbeschreibung
├── uv.lock                # Lock-Datei für die Abhängigkeitsverwaltung
└── .gitignore             # Git-Ignore-Dateien (falls vorhanden)
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

## Framwork

Welche Modelle zur Wahl stehen

```bash
uv run rag-project --models
```

Modelwechsel fuer einen einzelnen Lauf

```bash
uv run python -m rag_project.main_chatbot --model openrouter:z-ai/glm-5.3
```

Im Code — wenn der Bot fest gebunden sein soll

```bash
bot = ChatBot(model="openrouter:deepseek/deepseek-v4-flash")
```

## Virtuelle Umgebung anlegen und Abhängigkeiten installieren

```bash
uv sync
```

## Prüfen, ob die .env gefunden wird und der Key des Providers da ist

```bash
uv run rag-project --show-config
```

## Prüfen, ob Provider + Modell sauber aufgelöst werden

```bash
uv run rag-project --check
```

## Katalog anzeigen (das * markiert den Default je Provider)

```bash
uv run rag-project --models
```

## Tests laufen lassen

```bash
uv run pytest -q
```

## führt den Chatbot aus

```bash
uv run python -m rag_project.main_chatbot
```

## importiert nur die Klasse, startet nichts

```bash
uv run python -c "from rag_project.main_chatbot import ChatBot; print(ChatBot)"
```
