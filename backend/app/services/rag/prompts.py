# app/services/rag/prompts.py
from typing import Final


CHAT_CONTEXT_ANALYSIS_PROMPT: Final = """Analyze the following chat history and the current query to create a comprehensive, specific query that captures the user's true intention.

Chat History:
{chat_history}

Current Query: "{current_query}"

Instructions:
1. Consider the entire conversation flow and context
2. Identify implicit references to previous messages
3. Resolve any pronouns or contextual references
4. Incorporate relevant details from previous exchanges
5. Create a self-contained query that would work without requiring chat history

Format your response as a clear, detailed query that directly addresses the user's intention. The response should be a single query without explanations or additional text.

Example:
Chat History:
User: "What are the key features of the new product?"
Assistant: "The product has AI capabilities, cloud integration, and real-time analytics."
User: "How does that compare to competitors?"

Bad Response: "How do the AI, cloud, and analytics features compare to competitors?"
Good Response: "What is a detailed comparison of [Product Name]'s AI capabilities, cloud integration, and real-time analytics features against its main competitors in the market?"

Now, rephrase the current query with full context:"""

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
