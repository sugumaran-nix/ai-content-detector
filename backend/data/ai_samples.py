"""
Generates synthetic "AI-style" paragraphs for the starter training corpus.

HONEST LIMITATION:
The "AI" class is built from a template-driven generator covering six distinct
stylistic registers that reproduce real LLM stylistic tells. This is a reasonable
stand-in for demonstrating the feature pipeline, but NOT real LLM output.
For production-grade accuracy, retrain on a real corpus such as the HC3 dataset
or Kaggle's "LLM - Detect AI Generated Text" competition data.

Registers included:
  1. Hedge/self-help   — "it's important to note", "key takeaways"
  2. Tech-jargon       — "leverage", "synthesize", "algorithmically"
  3. Academic          — "this paper examines", "the findings suggest"
  4. Business/corporate — "actionable insights", "cross-functional alignment"
  5. Journalistic AI   — balanced "on one hand / on the other" structure
  6. Listicle          — numbered points, "First,", "Additionally,"
"""

import random
import re

# ---------------------------------------------------------------------------
# Contraction randomiser (prevent 100% contraction-presence as a tell)
# ---------------------------------------------------------------------------
_CONTRACTION_MAP = {
    "it's": "it is", "that's": "that is", "there's": "there is",
    "don't": "do not", "doesn't": "does not", "isn't": "is not",
    "aren't": "are not", "can't": "cannot", "won't": "will not",
    "it'll": "it will", "they're": "they are", "we're": "we are",
    "you're": "you are", "I've": "I have", "we've": "we have",
}


def _vary_contractions(text: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return text
    for contracted, expanded in _CONTRACTION_MAP.items():
        text = re.sub(re.escape(contracted), expanded, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Shared topic pool
# ---------------------------------------------------------------------------
TOPICS = [
    "remote work", "climate change", "artificial intelligence", "personal finance",
    "healthy eating", "time management", "electric vehicle adoption", "social media",
    "online education", "mental health", "renewable energy", "cybersecurity",
    "urban gardening", "cryptocurrency", "sleep hygiene", "minimalism",
    "freelancing", "language learning", "home automation", "sustainable fashion",
    "meditation", "small business marketing", "space exploration", "the vegan diet",
    "productivity software", "public transportation", "cloud computing", "yoga",
    "digital privacy", "career change", "fitness tracking", "remote learning",
    "smart home technology", "work-life balance", "podcasting", "e-commerce",
    "water conservation", "team collaboration software", "personal branding",
    "investment strategy", "stress management", "open source software",
    "urban planning", "wildlife conservation", "intermittent fasting",
    "video game design", "supply chain management", "telemedicine",
    "blockchain technology", "quantum computing",
]

# ---------------------------------------------------------------------------
# Register 1: Hedge / self-help
# ---------------------------------------------------------------------------
_HEDGE_INTROS = [
    "In today's fast-paced world, {topic} has become an increasingly important topic for many people.",
    "When it comes to {topic}, there are several factors worth considering carefully.",
    "{topic_cap} is a subject that has gained significant attention in recent years.",
    "It's important to note that {topic} can have a meaningful impact on everyday life.",
    "Many experts agree that {topic} plays a crucial role in modern society.",
    "Understanding {topic} is essential for anyone looking to thrive in today's environment.",
    "The growing interest in {topic} reflects a broader shift in how people approach daily challenges.",
]
_HEDGE_BODIES = [
    "To put this into perspective, consider the following key points: accessibility, affordability, and overall effectiveness.",
    "On one hand, {topic} offers a number of clear benefits. On the other hand, it's worth acknowledging the potential drawbacks as well.",
    "At the same time, it's crucial to remain mindful of potential limitations and adjust accordingly.",
    "Research consistently shows that individuals who engage with {topic} report higher levels of satisfaction and well-being.",
    "By taking a holistic approach, it becomes possible to harness the full potential of {topic} while mitigating associated risks.",
    "Furthermore, it's worth noting that the impact of {topic} extends far beyond immediate outcomes, influencing long-term trajectories.",
    "It is also important to recognize that different individuals may experience {topic} differently depending on their unique circumstances.",
]
_HEDGE_OUTROS = [
    "In conclusion, {topic} is a multifaceted subject that deserves careful thought and ongoing attention.",
    "Ultimately, the key takeaway is that {topic} requires a balanced and informed perspective.",
    "Moving forward, it will be essential to continue exploring the evolving landscape of {topic}.",
    "As our understanding of {topic} continues to grow, so too will our ability to leverage its benefits responsibly.",
    "The evidence strongly suggests that a proactive and thoughtful approach to {topic} yields the most favorable outcomes.",
]


def _hedge_paragraph(topic: str, rng: random.Random) -> str:
    tc = topic.capitalize()
    intro = rng.choice(_HEDGE_INTROS).format(topic=topic, topic_cap=tc)
    bodies = rng.sample(_HEDGE_BODIES, k=rng.randint(2, 4))
    body = " ".join(b.format(topic=topic, topic_cap=tc) for b in bodies)
    outro = rng.choice(_HEDGE_OUTROS).format(topic=topic, topic_cap=tc)
    return f"{intro} {body} {outro}"


# ---------------------------------------------------------------------------
# Register 2: Tech-jargon / corporate AI
# ---------------------------------------------------------------------------
_TECH_TEMPLATES = [
    (
        "The implementation of {topic} leverages cutting-edge methodologies to synthesize actionable insights "
        "from complex, multi-dimensional datasets. By algorithmically optimizing the underlying infrastructure, "
        "organizations can achieve unprecedented levels of operational efficiency. "
        "The integration of {topic} into existing workflows facilitates seamless scalability, "
        "enabling stakeholders to derive maximum value from their technology investments. "
        "Furthermore, the deployment of advanced {topic} solutions necessitates a robust governance framework "
        "to ensure compliance, security, and long-term sustainability across all verticals."
    ),
    (
        "Harnessing the transformative potential of {topic} requires a paradigm shift in how organizations "
        "conceptualize and operationalize their strategic objectives. "
        "The convergence of {topic} with adjacent technological domains creates synergistic opportunities "
        "that drive innovation and competitive differentiation. "
        "Stakeholders must proactively align cross-functional teams to orchestrate a cohesive implementation "
        "roadmap that maximizes return on investment while minimizing disruption to mission-critical processes. "
        "Ultimately, the strategic adoption of {topic} positions enterprises to capitalize on emerging market dynamics."
    ),
    (
        "From a technical standpoint, {topic} represents a significant advancement in how systems process, "
        "analyze, and respond to complex inputs. "
        "The architecture underlying modern {topic} solutions is designed for horizontal scalability, "
        "fault tolerance, and low-latency throughput. "
        "By abstracting implementation complexity behind intuitive interfaces, {topic} democratizes access "
        "to sophisticated capabilities that were previously limited to well-resourced organizations. "
        "This paradigm shift has profound implications for productivity, automation, and the future of work."
    ),
]


def _tech_paragraph(topic: str, rng: random.Random) -> str:
    template = rng.choice(_TECH_TEMPLATES)
    return template.format(topic=topic, topic_cap=topic.capitalize())


# ---------------------------------------------------------------------------
# Register 3: Academic / research style
# ---------------------------------------------------------------------------
_ACADEMIC_TEMPLATES = [
    (
        "This analysis examines the multifaceted dimensions of {topic}, drawing upon a synthesis of "
        "empirical evidence and theoretical frameworks. "
        "The findings suggest that {topic} exhibits a complex, non-linear relationship with key outcome variables, "
        "underscoring the necessity of context-sensitive approaches. "
        "Methodologically, this study employs a mixed-methods design to triangulate qualitative and quantitative data sources. "
        "The implications for policy and practice are discussed in relation to existing literature, "
        "with particular attention to the socioeconomic determinants that mediate the observed effects of {topic}."
    ),
    (
        "Scholarly discourse on {topic} has evolved considerably over the past decade, "
        "reflecting both advances in empirical methodology and shifts in theoretical orientation. "
        "A growing body of evidence indicates that {topic} is associated with statistically significant "
        "improvements in the dependent variables of interest, though effect sizes vary substantially across subgroups. "
        "Future research should prioritize longitudinal designs capable of establishing temporal precedence "
        "and ruling out confounding explanations. "
        "Furthermore, cross-cultural replication studies are needed to assess the generalizability of current findings."
    ),
]


def _academic_paragraph(topic: str, rng: random.Random) -> str:
    template = rng.choice(_ACADEMIC_TEMPLATES)
    return template.format(topic=topic, topic_cap=topic.capitalize())


# ---------------------------------------------------------------------------
# Register 4: Business / corporate communication
# ---------------------------------------------------------------------------
_BUSINESS_TEMPLATES = [
    (
        "Organizations that embrace {topic} as a strategic priority are better positioned to achieve "
        "sustainable competitive advantage in an increasingly dynamic marketplace. "
        "Key success factors include executive sponsorship, cross-functional alignment, and a commitment "
        "to continuous improvement through data-driven decision-making. "
        "To maximize impact, it is recommended that teams establish clear KPIs aligned with organizational objectives "
        "and implement regular performance reviews to ensure accountability. "
        "The business case for {topic} is compelling: early adopters consistently report measurable gains "
        "in efficiency, revenue growth, and customer satisfaction."
    ),
    (
        "A strategic approach to {topic} begins with a comprehensive assessment of current-state capabilities "
        "and a clear articulation of desired outcomes. "
        "Stakeholder engagement is critical at every phase of the implementation lifecycle, "
        "from initial scoping through post-deployment optimization. "
        "Organizations should allocate sufficient resources to change management, "
        "recognizing that cultural adoption is as important as technical execution. "
        "When executed effectively, {topic} initiatives deliver measurable ROI within twelve to eighteen months, "
        "creating a foundation for continued innovation."
    ),
]


def _business_paragraph(topic: str, rng: random.Random) -> str:
    template = rng.choice(_BUSINESS_TEMPLATES)
    return template.format(topic=topic, topic_cap=topic.capitalize())


# ---------------------------------------------------------------------------
# Register 5: Journalistic / balanced-analysis style
# ---------------------------------------------------------------------------
_JOURNALIST_INTROS = [
    "The debate surrounding {topic} has intensified in recent months, with experts divided on its long-term implications.",
    "Proponents of {topic} argue that the benefits far outweigh the risks, while critics maintain a more cautious stance.",
    "{topic_cap} has emerged as one of the most polarizing topics in contemporary discourse.",
    "A nuanced examination of {topic} reveals both promising opportunities and significant challenges that warrant careful consideration.",
]
_JOURNALIST_BODIES = [
    "Supporters point to robust empirical evidence demonstrating that {topic} yields measurable improvements across multiple dimensions.",
    "However, detractors highlight a series of unresolved concerns, including equity, accessibility, and unintended consequences.",
    "The available data presents a mixed picture: while short-term outcomes are largely positive, the long-term trajectory remains uncertain.",
    "Policymakers have been slow to respond, in part because the evidence base for {topic} continues to evolve rapidly.",
    "Independent analysts note that the effectiveness of {topic} is highly dependent on implementation context and stakeholder buy-in.",
]
_JOURNALIST_OUTROS = [
    "What seems clear is that {topic} will continue to shape the conversation for years to come.",
    "Resolving these tensions will require sustained dialogue, rigorous research, and a willingness to adapt policies as new evidence emerges.",
    "The coming years will be decisive in determining whether {topic} fulfills its promise or falls short of expectations.",
]


def _journalist_paragraph(topic: str, rng: random.Random) -> str:
    tc = topic.capitalize()
    intro = rng.choice(_JOURNALIST_INTROS).format(topic=topic, topic_cap=tc)
    bodies = rng.sample(_JOURNALIST_BODIES, k=rng.randint(2, 3))
    body = " ".join(b.format(topic=topic) for b in bodies)
    outro = rng.choice(_JOURNALIST_OUTROS).format(topic=topic)
    return f"{intro} {body} {outro}"


# ---------------------------------------------------------------------------
# Register 6: Listicle / structured enumeration
# ---------------------------------------------------------------------------
_LISTICLE_INTROS = [
    "There are several compelling reasons why {topic} deserves your attention.",
    "Understanding {topic} requires examining it through multiple lenses.",
    "The following points highlight the most important considerations related to {topic}.",
    "Experts consistently identify these core principles as central to any meaningful engagement with {topic}.",
]
_LISTICLE_POINTS = [
    "First, {topic} provides a structured framework for addressing challenges that previously lacked clear solutions.",
    "Second, the scalability of {topic} makes it applicable across a wide range of contexts and organizational sizes.",
    "Additionally, {topic} fosters collaboration by creating shared vocabulary and common standards among diverse stakeholders.",
    "Moreover, the evidence base supporting {topic} has grown substantially, lending credibility to its adoption.",
    "Furthermore, {topic} aligns with broader trends toward sustainability, efficiency, and human-centered design.",
    "Finally, early adopters of {topic} consistently report positive outcomes that exceed initial projections.",
]
_LISTICLE_OUTROS = [
    "Taken together, these points make a compelling case for prioritizing {topic} in both personal and professional contexts.",
    "Each of these dimensions reinforces the conclusion that {topic} is not merely a trend but a durable paradigm shift.",
    "The cumulative weight of this evidence leaves little doubt that {topic} merits serious and sustained attention.",
]


def _listicle_paragraph(topic: str, rng: random.Random) -> str:
    intro = rng.choice(_LISTICLE_INTROS).format(topic=topic)
    points = rng.sample(_LISTICLE_POINTS, k=rng.randint(3, 5))
    body = " ".join(p.format(topic=topic) for p in points)
    outro = rng.choice(_LISTICLE_OUTROS).format(topic=topic)
    return f"{intro} {body} {outro}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_GENERATORS = [
    _hedge_paragraph,
    _tech_paragraph,
    _academic_paragraph,
    _business_paragraph,
    _journalist_paragraph,
    _listicle_paragraph,
]


def generate_ai_paragraphs(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)  # nosec B311 - deterministic training-data generation, not security-sensitive
    topics = list(TOPICS)
    paragraphs = []
    while len(paragraphs) < n:
        rng.shuffle(topics)
        for topic in topics:
            if len(paragraphs) >= n:
                break
            gen = rng.choice(_GENERATORS)
            text = gen(topic, rng)
            text = _vary_contractions(text, rng)
            if len(text.split()) >= 30:
                paragraphs.append(text)
    return paragraphs[:n]
