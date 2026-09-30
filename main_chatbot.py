"""Terminal-Chatbot auf Basis der bestehenden Provider-Schicht.

Enthaelt:
    * :class:`ChatBot` -- kapselt Nachrichtenverlauf, Streaming und Reset
    * eine kleine Terminal-REPL mit Slash-Befehlen

Dieses Modul veraendert keine anderen Dateien des Projekts. Es nutzt
ausschliesslich die oeffentliche API von ``rag_project.config`` und
``rag_project.providers``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from rag_project.providers import build_llm, provider_report, resolve_target

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = "You are a friendly Python tutor."

#: Erlaubte Ausgabemodi der REPL.
MODES = ("stream", "invoke")

BANNER = """\
Python-Tutor Chatbot
====================
Befehle:
  /help              Diese Uebersicht anzeigen
  /system <Text>     System-Prompt ersetzen (leer = Default wiederherstellen)
  /show              Aktuellen System-Prompt anzeigen
  /reset             Verlauf loeschen (System-Prompt bleibt)
  /model             Aktives Modell und Provider anzeigen
  /mode <modus>      Ausgabemodus wechseln: stream | invoke
  /mode              Aktuellen Ausgabemodus anzeigen
  /exit  oder /quit  Beenden
"""


# ---------------------------------------------------------------------------
# ChatBot
# ---------------------------------------------------------------------------

@dataclass
class ChatBot:
    """Zustandsbehafteter Chatbot mit Streaming- und Invoke-Modus.

    Der Nachrichtenverlauf wird als LangChain-Nachrichtenliste gehalten:
    System-Nachricht an Position 0, danach abwechselnd Human/AI.
    """

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model: str | None = None
    temperature: float | None = None

    _llm: object = field(default=None, init=False, repr=False)

    # --------------------------------------------------------------- Aufbau
    def __post_init__(self) -> None:
        # streaming=True wird immer gesetzt: der Client muss den Stream
        # unterstuetzen, auch wenn die REPL gerade invoke verwendet.
        self._llm = build_llm(self.model, streaming=True, **self._llm_overrides())
        self.history: list = [self._system_message()]

    def _llm_overrides(self) -> dict:
        """Optionale Client-Parameter, die nur gesetzt werden, wenn vorhanden.

        Ohne diesen Helfer muesste derselbe Ausdruck an zwei Stellen stehen
        (__post_init__ und reset).
        """
        if self.temperature is None:
            return {}
        return {"temperature": self.temperature}

    def _system_message(self):
        from langchain_core.messages import SystemMessage

        return SystemMessage(content=self.system_prompt)

    # ---------------------------------------------------------------- Reset
    def reset(self, system_prompt: str | None = None) -> None:
        """Verlauf leeren und optional den System-Prompt ersetzen.

        :param system_prompt: Neuer Prompt. ``None`` oder ein leerer String
            stellt :data:`DEFAULT_SYSTEM_PROMPT` wieder her.
        """
        if system_prompt is not None:
            text = system_prompt.strip()
            self.system_prompt = text or DEFAULT_SYSTEM_PROMPT
            self._llm = build_llm(
                self.model, streaming=True, **self._llm_overrides()
            )
        self.history = [self._system_message()]

    # -------------------------------------------------------------- Nutzung
    def invoke(self, message: str) -> str:
        """Einmaliger Aufruf: Anfrage senden, vollstaendige Antwort erhalten.

        Geeignet fuer kurze Antworten, Skripte und Tests.
        """
        from langchain_core.messages import HumanMessage

        self.history.append(HumanMessage(content=message))
        response = self._llm.invoke(self.history)
        self.history.append(response)
        return _as_text(response)

    def stream(self, message: str) -> Iterator[str]:
        """Streaming-Aufruf: gibt Textstuecke aus, sobald sie eintreffen.

        Der Verlauf wird erst nach dem letzten Stueck ergaenzt, damit ein
        abgebrochener Stream keine halbe AI-Nachricht hinterlaesst.
        """
        from langchain_core.messages import AIMessage, HumanMessage

        self.history.append(HumanMessage(content=message))

        pieces: list[str] = []
        for chunk in self._llm.stream(self.history):
            text = _as_text(chunk)
            if text:
                pieces.append(text)
                yield text

        self.history.append(AIMessage(content="".join(pieces)))

    # -------------------------------------------------------------- Auskunft
    @property
    def model_label(self) -> str:
        target = resolve_target(self.model)
        return target.summary()

    def turn_count(self) -> int:
        """Anzahl der Nutzer-Turns ohne System-Nachricht."""
        return sum(1 for m in self.history if m.__class__.__name__ == "HumanMessage")


def _as_text(message: object) -> str:
    """Zieht Klartext aus einem LangChain-Objekt oder String.

    ``content`` kann je nach Anbieter ein String oder eine Liste von
    Bloecken sein -- beides wird hier auf Text reduziert.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Terminal-REPL
# ---------------------------------------------------------------------------

def handle_command(bot: ChatBot, line: str, state: dict) -> bool:
    """Verarbeitet Slash-Befehle.

    :param state: veraenderbarer Zustand der REPL. Enthaelt ``mode``
        (``"stream"`` oder ``"invoke"``). Die Uebergabe per dict statt per
        Nicht-lokal-Deklaration haelt die Funktion testbar.

    :return: ``True``, wenn die REPL beendet werden soll.
    """
    raw = line.strip()
    command, _, argument = raw.partition(" ")
    command = command.lower()
    argument = argument.strip().lower()

    if command in {"/exit", "/quit"}:
        print("Bis spaeter.")
        return True

    if command == "/help":
        print(BANNER)

    elif command == "/system":
        if not argument:
            bot.reset()
            print(f"System-Prompt zurueckgesetzt auf: {bot.system_prompt!r}")
        else:
            # Gross-/Kleinschreibung des Prompts erhalten -> aus raw lesen
            original = raw.partition(" ")[2].strip()
            bot.reset(original)
            print(f"Neuer System-Prompt: {bot.system_prompt!r}")
            print("Verlauf wurde geleert, damit er ab jetzt gilt.")

    elif command == "/show":
        print(f"Aktueller System-Prompt: {bot.system_prompt!r}")

    elif command == "/reset":
        bot.reset()
        print("Verlauf geleert. System-Prompt unveraendert.")

    elif command == "/model":
        print(bot.model_label)
        info = provider_report()
        print(f"  Provider : {info['chat_provider']}")
        print(f"  Modell   : {info['resolved_model']}")

    elif command == "/mode":
        if not argument:
            print(f"Aktueller Modus: {state['mode']}")
            print("Verwendung: /mode stream | /mode invoke")
        elif argument in MODES:
            state["mode"] = argument
            print(f"Modus gewechselt auf: {argument}")
            if argument == "invoke":
                print("Antworten kommen jetzt vollstaendig statt stueckweise.")
        else:
            print(f"Unbekannter Modus: {argument!r}. Erlaubt: {' | '.join(MODES)}")

    else:
        print(f"Unbekannter Befehl: {command}. /help fuer die Uebersicht.")

    return False


def run(bot: ChatBot, argv: Sequence[str] | None = None) -> int:
    """Interaktive Schleife.

    Der Modus startet je nach ``--no-stream`` und laesst sich zur Laufzeit
    mit ``/mode`` umschalten.
    """
    argv = list(argv or [])
    state: dict = {"mode": "invoke" if "--no-stream" in argv else "stream"}

    print(BANNER)
    print(f"Modell   : {bot.model_label}")
    print(f"Modus    : {state['mode']}")
    print(f"System   : {bot.system_prompt!r}")
    print()

    while True:
        try:
            line = input("Du > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBis spaeter.")
            return 0

        if not line:
            continue
        if line.startswith("/"):
            if handle_command(bot, line, state):
                return 0
            continue

        print("Bot > ", end="", flush=True)
        try:
            if state["mode"] == "stream":
                for piece in bot.stream(line):
                    print(piece, end="", flush=True)
                print()
            else:
                print(bot.invoke(line))
        except KeyboardInterrupt:
            print("\n[abgebrochen]")
        except Exception as exc:  # Netzwerk-, Auth- oder API-Fehler
            print(f"\n[Fehler] {exc}", file=sys.stderr)
        print()


def main(argv: Sequence[str] | None = None) -> int:
    """Einstiegspunkt. Nutzt die Settings aus ``.env`` (Provider + Modell)."""
    argv = list(argv or [])
    model = None
    if "--model" in argv:
        index = argv.index("--model")
        if index + 1 < len(argv):
            model = argv[index + 1]

    bot = ChatBot(model=model)
    return run(bot, argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
