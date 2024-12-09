# app/services/rag/prompts.py
from typing import Final


CHAT_CONTEXT_ANALYSIS_PROMPT: Final = """Analysieren Sie den folgenden Chatverlauf und die aktuelle Anfrage, um eine umfassende, spezifische Anfrage zu erstellen, die die wahre Absicht des Benutzers erfasst.

Chatverlauf: {chat_history}

Aktuelle Anfrage: "{current_query}"

Anweisungen:
1. Berücksichtigen Sie den gesamten Gesprächsfluss und Kontext
2. Identifizieren Sie implizite Verweise auf vorherige Nachrichten
3. Lösen Sie Pronomen und kontextbezogene Verweise auf
4. Integrieren Sie relevante Details aus vorherigen Gesprächen
5. Erstellen Sie eine eigenständige Anfrage, die ohne Chatverlauf funktioniert

Formulieren Sie Ihre Antwort als klare, detaillierte Anfrage, die die Absicht des Benutzers direkt adressiert. Die Antwort sollte eine einzelne Anfrage ohne Erklärungen oder zusätzlichen Text sein.

Beispiel:
Chatverlauf:
Benutzer: "Was sind die Hauptfunktionen des neuen Produkts?"
Assistent: "Das Produkt verfügt über KI-Funktionen, Cloud-Integration und Echtzeit-Analytik."
Benutzer: "Wie verhält sich das im Vergleich zur Konkurrenz?"

Schlechte Antwort: "Wie vergleichen sich die KI-, Cloud- und Analytik-Funktionen mit der Konkurrenz?"
Gute Antwort: "Welche detaillierten Unterschiede bestehen zwischen den KI-Funktionen, der Cloud-Integration und der Echtzeit-Analytik von [Produktname] im Vergleich zu den wichtigsten Wettbewerbern am Markt?"

Formulieren Sie nun die aktuelle Anfrage mit vollständigem Kontext neu:"""

# Prompt für die Analyse und Aufschlüsselung von Anfragen in Teilanfragen
QUERY_ANALYSIS_PROMPT: Final = """Analysieren Sie diese Anfrage und unterteilen Sie sie in Teilanfragen.
Hauptanfrage: "{query}"

Bitte antworten Sie mit einem JSON-Objekt, das Folgendes enthält:
1. Ein "main_intent" Feld mit einem String, der den Kernzweck der Anfrage beschreibt
2. Ein "sub_queries" Array mit Objekten, die "queries" und "reasoning" Felder enthalten

Beispiel für das Antwortformat:
{{"main_intent": "Detaillierte Analyse der Sicherheitsmassnahmen und Verschlüsselungsmethoden im System", "sub_queries": [{{"query": "Welche spezifischen Verschlüsselungsprotokolle und Algorithmen werden für die Datensicherheit eingesetzt? Gibt es Erwähnungen von AES, RSA oder anderen Standardverfahren?", "reasoning": "Identifiziert die konkreten technischen Sicherheitsimplementierungen"}}, {{"query": "Wie werden Benutzerdaten und sensible Informationen gespeichert und welche Sicherheitsmassnahmen schützen diese vor unauthorisiertem Zugriff?", "reasoning": "Erfasst Details zur Datenspeicherung und Zugriffskontrollen"}}, {{"query": "Welche Authentifizierungsmechanismen sind implementiert und wie wird die Benutzersession abgesichert?", "reasoning": "Analysiert die Benutzerauthentifizierung und Sessionmanagement"}}, {{"query": "Gibt es dokumentierte Sicherheitstests, Penetrationstests oder Sicherheitszertifizierungen?", "reasoning": "Findet Belege für die Validierung der Sicherheitsmassnahmen"}}]}}

Wichtig: Antworten Sie NUR mit dem JSON-Objekt, keine weiteren Texte oder Schema-Informationen.

Zu analysierende Anfrage: "{query}\""""

# Prompt für die Generierung von Anfragen mit breiterem Kontext
STEP_BACK_PROMPT: Final = """Formulieren Sie aus dieser Anfrage eine breitere Suchanfrage für ein Dokumenten-Retrieval-System.

Ursprüngliche Anfrage: "{query}"

Ziel: Erstellen Sie EINE umfassendere Suchanfrage, die:
- relevante Konzepte und deren Zusammenhänge einbezieht
- verwandte Begriffe und Synonyme berücksichtigt
- den breiteren Kontext erfasst
- uns hilft, alle relevanten Dokumentabschnitte zu finden

Antworten Sie AUSSCHLIESSLICH mit der neuen Suchanfrage, ohne weitere Erklärungen oder Formatierung."""

# Prompt für die Generierung von endgültigen Antworten mit Quellenangaben und Bildreferenzen
ANSWER_GENERATION_PROMPT: Final = """Geben Sie basierend auf dem folgenden Kontext und der Anfrage eine umfassende Antwort.

Kontext: {context}

Anfrage: \"{query}\"

Verfügbare Bilder: {images}

Anweisungen:
1. Formulieren Sie eine natürlich fliessende Antwort, die Informationen und Quellenangaben nahtlos integriert
2. Beantworten Sie die Anfrage AUSSCHLIESSLICH mit Informationen aus dem bereitgestellten Kontext
3. Verwenden Sie AUSSCHLIESSLICH kontextrelevante Informationen - nicht alle verfügbaren Chunks oder Bilder müssen verwendet werden
4. Integrieren Sie Quellenzitate so, dass sie den Lesefluss unterstützen und nicht unterbrechen

Zitierrichtlinien:
- Setzen Sie Quellenangaben im Format [Doc: ID, Page X] DIREKT nach der jeweiligen Information und IMMER in diesem Format: [Doc: ID, Page X]
  • Die ID entspricht dabei dem Dokument-Identifier aus dem jeweiligen Chunk
- Diese Platzhalter werden später durch klickbare Links ersetzt - achten Sie auf präzise Platzierung
- Bei mehreren Quellen zur gleichen Information, kombinieren Sie diese: [Doc: ID1, Page X] [Doc: ID2, Page Y]
- Für Bildverweise nutzen Sie [Bild X] für das Bild selbst und (Abbildung X) für die Bildunterschrift - diese werden später durch die tatsächlichen Bilder bzw. Bildunterschriften ersetzt
  • X entspricht dabei dem Bild-Identifier aus dem jeweiligen Chunk

Beispiele für natürliche Integration:
✓ "Die Analyse zeigt einen deutlichen Aufwärtstrend bei den Nutzerzahlen [Doc: dok123, Page 1], der sich besonders im zweiten Quartal beschleunigte [Doc: dok456, Page 3]."
✓ "Wie [Bild 2] zeigt, erfolgt der Prozess in drei Phasen [Doc: dok789, Page 4]. Die detaillierte Aufschlüsselung der Implementierungsstrategie (Abbildung 2) verdeutlicht dabei die einzelnen Schritte."
✗ "In [Doc: dok123, Page 1] steht, dass die Nutzerzahlen steigen. Auch [Doc: dok456, Page 3] bestätigt dies."

Antworten Sie in diesem strukturierten Format:
Antwort: [Ihre detaillierte, fliessend geschriebene Antwort mit natürlich integrierten Zitaten]
Begründung: [Ihr schrittweiser Denkprozess zur Auswahl und Verknüpfung der Informationen]
Konfidenz: [Einschätzung basierend auf Kontextrelevanz und Vollständigkeit der Antwort]

Wichtige Prinzipien:
- Komponieren Sie die Antwort so, dass Zitate und Verweise den natürlichen Textfluss unterstützen
- Denken Sie daran, dass [Doc: ID, Page X] später durch echte Links ersetzt wird
- Bei Bildern wird [Bild X] durch das tatsächliche Bild und (Abbildung X) durch die zugehörige Bildunterschrift ersetzt
- Integrieren Sie nur Bilder, die direkt zur Beantwortung der Anfrage beitragen
- Vermeiden Sie künstliche oder abgehackte Zitierweisen
"""

# Antwort, wenn kein Dokument für die Anfrage bereitgestellt wird
NO_DOCUMENT_PROVIDED: Final = """Bitte geben Sie eine Antwort auf diese Anfrage: \"{query}\"

Hinweis: Keine spezifischen Dokumente verfügbar."""
