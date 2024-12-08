# app/services/rag/prompts.py
from typing import Final

# Prompt für die Analyse und Aufschlüsselung von Anfragen in Teilanfragen
QUERY_ANALYSIS_PROMPT: Final = """Analysieren Sie diese Anfrage und unterteilen Sie sie in Teilanfragen.
Hauptanfrage: \"{query}\"

Bitte antworten Sie mit einem JSON-Objekt, das Folgendes enthält:
1. Ein "main_intent" Feld mit einem String, der den Kernzweck der Anfrage beschreibt
2. Ein "sub_queries" Array mit Objekten, die "queries" und "reasoning" Felder enthalten

Beispiel für das Antwortformat:
{{"main_intent": "verstehen, was der Benutzer mit X meint", "sub_queries": [{{"query": "was ist X?", "reasoning": "müssen das grundlegende Konzept klären"}}]}}

Wichtig: Antworten Sie NUR mit dem JSON-Objekt, keine weiteren Texte oder Schema-Informationen.

Zu analysierende Anfrage: \"{query}\""""

# Prompt für die Generierung von Anfragen mit breiterem Kontext
STEP_BACK_PROMPT: Final = """Bevor wir die Anfrage direkt beantworten, lassen Sie uns einen Schritt zurücktreten und den breiteren Kontext betrachten.
Anfrage: \"{query}\"

Welche übergeordneten Themen oder Konzepte sollten wir berücksichtigen, um eine umfassendere Antwort zu geben?
Konzentrieren Sie sich darauf, eine allgemeinere Anfrage zu generieren, die hilft, relevanten Kontext zu ermitteln."""

# Prompt für die Generierung von endgültigen Antworten mit Quellenangaben und Bildreferenzen
ANSWER_GENERATION_PROMPT: Final = """Geben Sie basierend auf dem folgenden Kontext und der Anfrage eine umfassende Antwort. 

Kontext:
{context}

Anfrage: \"{query}\"

Verfügbare Bilder:
{images}

Anweisungen:
1. Beantworten Sie die Anfrage NUR mit Informationen aus dem bereitgestellten Kontext
2. Zitieren Sie ALLE Quellen ausschliesslich im Format [Doc: ID, Page X] DIREKT nach jeder Information
3. Verwenden Sie bei Bildreferenzen das Format (Abbildung X), wobei X der Bildunterschrift entspricht
4. Geben Sie explizit an, aus welchem Dokument und welcher Seite die jeweilige Information stammt
5. Verwenden Sie Direktzitate in ihrer Originalsprache

Beispiel für das Zitierformat:
"Das Dokument besagt 'Der Prozess beginnt mit...' [Doc: dok123, Page 1] und erläutert weiter 'Die Analyse zeigt...' [Doc: dok456, Page 3]"

Antworten Sie in diesem strukturierten Format:
Antwort: [Ihre detaillierte Antwort mit eingebetteten Zitaten]
Begründung: [Ihr schrittweiser Denkprozess]
Konfidenz: [Punktzahl zwischen 0-1 basierend auf Kontextrelevanz]

Wenn einer der abgerufenen Textabschnitte zugehörige Bilder oder Abbildungen enthält, die das Konzept verdeutlichen würden,
fügen Sie unbedingt einen Verweis darauf in Ihrer Antwort mit der Notation [Bild X] ein. Zum Beispiel:
- Bei der Erklärung eines Diagramms: "Wie wir in [Bild 1] sehen können, zeigt der Prozess..."
- Wenn ein Bild als Beleg dient: "Das Dokument veranschaulicht dies in [Bild 2]"
Verweisen Sie nur auf Bilder, die für die Beantwortung der Anfrage direkt relevant sind."""

# Antwort, wenn kein Dokument für die Anfrage bereitgestellt wird
NO_DOCUMENT_PROVIDED: Final = """Bitte geben Sie eine Antwort auf diese Anfrage: \"{query}\"

Hinweis: Keine spezifischen Dokumente verfügbar."""
