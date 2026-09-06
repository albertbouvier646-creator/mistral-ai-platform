SYSTEM_PROMPT = """Tu es un assistant IA utile. Tu as accès à un outil search_web pour
obtenir des informations récentes (prix, actualités, météo, résultats sportifs, dates
après ta connaissance). Utilise-le uniquement quand c'est nécessaire. Cite tes sources
quand tu utilises des résultats de recherche."""

SEARCH_TRIGGERS = [
    "prix actuel", "cours", "actualité", "aujourd'hui", "dernières nouvelles",
    "météo", "réglementation actuelle", "résultats sportifs", "en direct",
]


def needs_web_search(message: str) -> bool:
    lowered = message.lower()
    return any(trigger in lowered for trigger in SEARCH_TRIGGERS)


def build_context_messages(system_prompt: str, memories: list[str], history: list[dict], search_results: list[dict] | None) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]

    if memories:
        memory_block = "\n".join(f"- {m}" for m in memories)
        messages.append({"role": "system", "content": f"Mémoire utilisateur:\n{memory_block}"})

    if search_results:
        results_block = "\n".join(
            f"[{i+1}] {r['title']} - {r['url']}\n{r['snippet']}"
            for i, r in enumerate(search_results)
        )
        messages.append({"role": "system", "content": f"Résultats de recherche web:\n{results_block}"})

    messages.extend(history)
    return messages
