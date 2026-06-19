"""
Generates synthetic "AI-style" paragraphs for the starter training corpus.

IMPORTANT / HONEST LIMITATION:
We have no network access to a real labeled human-vs-AI-text dataset
(no Hugging Face Hub, no Kaggle, no OpenAI API in this environment), so the
"AI" class is built from a template-driven generator that reproduces common
LLM stylistic tells: hedging language, transition-heavy structure, balanced
"on one hand / on the other hand" framing, listy enumeration, and generic
intensifiers. This is a reasonable stand-in for demonstrating the feature
engineering + classifier pipeline, but it is NOT real LLM output. For
production-grade accuracy, retrain on a real corpus such as the HC3 dataset
or Kaggle's "LLM - Detect AI Generated Text" competition data (see README).
"""

import random
import re

_CONTRACTION_MAP = {
    "it's": "it is", "that's": "that is", "there's": "there is",
    "don't": "do not", "doesn't": "does not", "isn't": "is not",
    "aren't": "are not", "can't": "cannot", "won't": "will not",
    "it'll": "it will", "they're": "they are",
}


def _vary_contractions(text: str, rng: random.Random) -> str:
    """Randomly expand contractions so contraction-presence isn't a spurious
    shortcut feature correlated 1:1 with the AI label (every hand-written
    template originally used contractions like "it's")."""
    if rng.random() < 0.5:
        return text
    for contracted, expanded in _CONTRACTION_MAP.items():
        text = re.sub(re.escape(contracted), expanded, text, flags=re.IGNORECASE)
    return text


TOPICS = [
    "remote work", "climate change", "artificial intelligence", "personal finance",
    "healthy eating", "time management", "electric vehicle adoption", "social media",
    "online education", "mental health", "renewable energy", "cybersecurity",
    "urban gardening", "cryptocurrency", "sleep hygiene", "minimalism",
    "freelancing", "language learning", "home automation", "sustainable fashion",
    "meditation", "small business marketing", "space exploration", "the vegan diet",
    "productivity software", "public transportation", "cloud computing", "yoga",
    "digital privacy", "career change", "fitness tracking", "remote learning",
    "the plant-based diet", "smart home technology", "work-life balance", "podcasting",
    "e-commerce", "water conservation", "team collaboration software",
    "personal branding", "investment strategy", "stress management",
    "open source software", "urban planning", "wildlife conservation",
    "intermittent fasting", "video game design", "supply chain management",
    "telemedicine", "the co-working space model",
]

INTROS = [
    "In today's fast-paced world, {topic} has become an increasingly important topic for many people.",
    "When it comes to {topic}, there are several factors worth considering.",
    "{topic_cap} is a subject that has gained significant attention in recent years.",
    "It's important to note that {topic} can have a meaningful impact on everyday life.",
    "Many experts agree that {topic} plays a crucial role in modern society.",
    "Over the past decade, {topic} has evolved considerably, shaping how people live and work.",
    "Understanding {topic} requires looking at it from multiple perspectives.",
    "As awareness around {topic} continues to grow, more people are looking for practical guidance.",
]

BODY_TEMPLATES = [
    "On one hand, {topic} offers a number of clear benefits. On the other hand, it's worth acknowledging the potential drawbacks as well.",
    "There are several key factors to consider, including cost, accessibility, and long-term impact.",
    "Furthermore, it's essential to recognize that individual circumstances can significantly influence outcomes.",
    "Additionally, experts generally recommend taking a balanced and informed approach.",
    "That said, it's worth mentioning that results can vary depending on a range of variables.",
    "In many cases, the benefits tend to outweigh the challenges, provided that certain best practices are followed.",
    "Moreover, ongoing research continues to shed new light on this evolving area.",
    "It's also worth noting that consistency and patience are often key to achieving meaningful results.",
    "To put this into perspective, consider the following key points: accessibility, affordability, and overall effectiveness.",
    "While there is no one-size-fits-all solution, a thoughtful and well-informed strategy can make a significant difference.",
    "Generally speaking, small, incremental changes tend to be more sustainable than drastic ones.",
    "At the same time, it's crucial to remain mindful of potential limitations and adjust accordingly.",
]

CONCLUSIONS = [
    "In conclusion, {topic} is a multifaceted subject that deserves careful thought and ongoing attention.",
    "Overall, taking a balanced, informed approach to {topic} can lead to meaningful, lasting benefits.",
    "Ultimately, the key to success with {topic} lies in staying informed and adapting to individual needs.",
    "To summarize, {topic} offers significant potential, provided it is approached thoughtfully and consistently.",
    "All things considered, {topic} remains a valuable area worth exploring further.",
]

LIST_INTROS = [
    "Here are a few key points to keep in mind:",
    "Consider the following key takeaways:",
    "A few practical tips can help, such as:",
]

LIST_ITEMS = [
    "Start small and build up gradually over time.",
    "Stay consistent, even when progress feels slow.",
    "Seek guidance from reputable, well-informed sources.",
    "Track your progress to stay motivated.",
    "Be mindful of common pitfalls and plan accordingly.",
    "Set realistic, achievable goals from the outset.",
]


def _fill(template: str, topic: str) -> str:
    return template.format(topic=topic, topic_cap=topic[0].upper() + topic[1:])


def generate_ai_paragraph(rng: random.Random) -> str:
    topic = rng.choice(TOPICS)
    parts = [_fill(rng.choice(INTROS), topic)]

    n_body = rng.randint(2, 4)
    body_pool = BODY_TEMPLATES[:]
    rng.shuffle(body_pool)
    for t in body_pool[:n_body]:
        parts.append(_fill(t, topic))

    if rng.random() < 0.35:
        parts.append(rng.choice(LIST_INTROS))
        items = LIST_ITEMS[:]
        rng.shuffle(items)
        for item in items[: rng.randint(2, 3)]:
            parts.append(item)

    parts.append(_fill(rng.choice(CONCLUSIONS), topic))
    return _vary_contractions(" ".join(parts), rng)


def generate_ai_paragraphs(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    seen = set()
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        p = generate_ai_paragraph(rng)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


if __name__ == "__main__":
    samples = generate_ai_paragraphs(5)
    for s in samples:
        print(s)
        print("---")
