"""Financial copilot RAG over corporate 10-K statement notes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LLMProvider = Literal["local", "openai", "gemini", "ollama"]
SearchMode = Literal["hybrid", "keyword", "semantic"]


SAMPLE_VARIANCE_QUESTIONS: list[str] = [
    "Why did SG&A expense increase?",
    "Explain the Gross Margin contraction",
    "What factors contributed to the increase in Cost of Goods Sold?",
    "How did foreign exchange impact net income?",
    "What drove the change in Research and Development spending?",
]


SAMPLE_10K_NOTES: list[dict[str, str]] = [
    {
        "section": "MD&A",
        "title": "SG&A Expense Analysis",
        "content": (
            "Selling, general and administrative (SG&A) expense increased by $142 million, or 11.4%, "
            "to $1.39 billion in fiscal 2025 compared to $1.25 billion in fiscal 2024. The increase "
            "was primarily driven by higher sales and marketing investments of $68 million to support "
            "new product launches, a $34 million increase in stock-based compensation due to retention "
            "grants, and $24 million of additional cloud infrastructure and SaaS licensing costs. "
            "Professional services and audit fees rose $16 million following the ERP implementation. "
            "Partially offsetting these increases, travel and entertainment expense declined $8 million "
            "as the company continued hybrid work policies."
        ),
    },
    {
        "section": "MD&A",
        "title": "Gross Margin Discussion",
        "content": (
            "Gross margin contracted 280 basis points to 58.2% in fiscal 2025 from 61.0% in fiscal 2024. "
            "The decline was attributable to unfavorable product mix shift toward lower-margin hardware bundles, "
            "which reduced gross margin by approximately 150 basis points. Input cost inflation, including "
            "semiconductor components and freight surcharges, contributed an additional 90 basis points of "
            "pressure. Promotional pricing in the enterprise segment during Q3 reduced margin by 60 basis points. "
            "Manufacturing efficiency gains and supplier renegotiations partially offset the decline by "
            "20 basis points."
        ),
    },
    {
        "section": "Note 2",
        "title": "Summary of Significant Accounting Policies - Revenue Recognition",
        "content": (
            "The Company recognizes revenue when control of goods or services transfers to the customer. "
            "For software subscriptions, revenue is recognized ratably over the contract term. During fiscal 2025, "
            "the Company modified its breakage estimate for prepaid service credits, resulting in a $12 million "
            "increase in recognized revenue. Multi-element arrangements are allocated based on standalone selling "
            "prices. Contract modifications are assessed to determine whether they should be treated as separate "
            "contracts or as part of the existing arrangement."
        ),
    },
    {
        "section": "Note 8",
        "title": "Cost of Revenue and Inventory",
        "content": (
            "Cost of revenue increased $219 million, or 14.8%, year-over-year to $1.70 billion. Higher unit volumes "
            "accounted for $96 million of the increase. Elevated component costs and expedited freight added "
            "$74 million. Inventory obsolescence charges increased $31 million due to discontinuation of legacy SKUs. "
            "Standard costs were updated in Q2 to reflect supplier price increases, contributing $18 million to the "
            "year-over-year change."
        ),
    },
    {
        "section": "MD&A",
        "title": "Foreign Currency and Other Income (Expense)",
        "content": (
            "Foreign exchange fluctuations negatively impacted pre-tax income by approximately $27 million in fiscal 2025. "
            "The U.S. dollar strengthened against the Euro and British Pound, reducing translated revenue from EMEA "
            "operations. The Company maintains a hedging program for forecasted transactions; however, hedge "
            "ineffectiveness and mark-to-market adjustments on unhedged balances resulted in a net FX loss of "
            "$11 million recorded in other income (expense). A $16 million remeasurement loss arose from intercompany "
            "loan balances denominated in foreign currencies."
        ),
    },
    {
        "section": "MD&A",
        "title": "Research and Development",
        "content": (
            "Research and development expense increased $58 million, or 9.7%, to $655 million in fiscal 2025. "
            "The increase reflects expanded headcount in AI and data platform engineering, higher contractor spend "
            "for prototype development, and increased cloud compute used for model training. R&D as a percentage of "
            "revenue was 12.4% compared to 12.1% in the prior year. The Company capitalized $22 million of "
            "software development costs compared to $15 million in fiscal 2024."
        ),
    },
    {
        "section": "Note 12",
        "title": "Income Taxes",
        "content": (
            "The effective tax rate was 22.8% in fiscal 2025 versus 21.5% in fiscal 2024. The increase was primarily "
            "due to reduced benefit from foreign-derived intangible income (FDII), geographic mix shift toward higher "
            "tax jurisdictions, and a $9 million valuation allowance established against certain state net operating "
            "loss carryforwards. These items were partially offset by $14 million of R&D tax credits."
        ),
    },
    {
        "section": "Note 15",
        "title": "Debt and Liquidity",
        "content": (
            "Total debt increased to $2.1 billion from $1.8 billion, reflecting the issuance of $400 million in "
            "senior notes partially offset by scheduled repayments. Interest expense rose $19 million due to higher "
            "average borrowings and elevated reference rates. The Company ended the year with $890 million in cash "
            "and equivalents and a net leverage ratio of 1.8x adjusted EBITDA."
        ),
    },
    {
        "section": "MD&A",
        "title": "Segment Performance - Enterprise",
        "content": (
            "Enterprise segment revenue grew 7.2% year-over-year, while operating margin declined from 24.1% to 21.6%. "
            "Margin compression was driven by competitive discounting on renewal contracts, higher implementation "
            "costs for large deployments, and increased customer success staffing to improve retention metrics."
        ),
    },
    {
        "section": "Note 6",
        "title": "Restructuring Charges",
        "content": (
            "During fiscal 2025, the Company recorded restructuring charges of $41 million related to workforce "
            "optimization and facility consolidation. Severance and related costs were $29 million, while asset "
            "impairments and lease termination fees totaled $12 million. The Company expects annualized savings of "
            "approximately $55 million beginning in fiscal 2026."
        ),
    },
]


@dataclass(frozen=True)
class DocumentChunk:
    """Indexed chunk from a 10-K financial statement note."""

    chunk_id: str
    section: str
    title: str
    content: str
    source_label: str


@dataclass(frozen=True)
class RetrievedChunk:
    """Search result with relevance score."""

    chunk: DocumentChunk
    score: float
    keyword_score: float
    semantic_score: float


@dataclass
class CopilotResponse:
    """Answer returned by the financial copilot."""

    question: str
    answer: str
    sources: list[RetrievedChunk]
    provider: LLMProvider
    search_mode: SearchMode


@dataclass
class FinancialCopilot:
    """
    RAG copilot for corporate 10-K financial statement notes.

    Defaults to local hybrid keyword + vector search with template-based answers.
    Optionally uses OpenAI, Gemini, or Ollama when credentials are supplied.
    """

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    ollama_model: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 3
    keyword_weight: float = 0.4
    semantic_weight: float = 0.6

    _documents: list[DocumentChunk] = field(default_factory=list, init=False, repr=False)
    _corpus: list[str] = field(default_factory=list, init=False, repr=False)
    _tfidf_vectorizer: TfidfVectorizer | None = field(default=None, init=False, repr=False)
    _tfidf_matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    _embedding_model: object | None = field(default=None, init=False, repr=False)
    _embedding_matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    _is_indexed: bool = field(default=False, init=False, repr=False)

    def get_sample_questions(self) -> list[str]:
        """Return pre-loaded variance analysis questions."""
        return list(SAMPLE_VARIANCE_QUESTIONS)

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Scale scores to 0-1 range for hybrid ranking."""
        if scores.size == 0:
            return scores
        min_score = float(scores.min())
        max_score = float(scores.max())
        if np.isclose(min_score, max_score):
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    @staticmethod
    def _chunk_text(text: str, max_words: int = 120) -> list[str]:
        """Split long note text into retrieval-friendly chunks."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[str] = []
        current: list[str] = []
        word_count = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words > max_words and current:
                chunks.append(" ".join(current))
                current = [sentence]
                word_count = sentence_words
            else:
                current.append(sentence)
                word_count += sentence_words
        if current:
            chunks.append(" ".join(current))
        return chunks or [text]

    def _build_document_chunks(self, notes: list[dict[str, str]]) -> list[DocumentChunk]:
        """Convert raw 10-K note records into indexed document chunks."""
        documents: list[DocumentChunk] = []
        for note_index, note in enumerate(notes):
            section = note["section"]
            title = note["title"]
            content = note["content"]
            split_contents = self._chunk_text(content)
            for chunk_index, chunk_content in enumerate(split_contents):
                chunk_id = f"10k-{note_index + 1:02d}-{chunk_index + 1:02d}"
                documents.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        section=section,
                        title=title,
                        content=chunk_content,
                        source_label=f"{section} | {title}",
                    )
                )
        return documents

    def _load_embedding_model(self) -> object | None:
        """Lazy-load a local sentence-transformer model when available."""
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        except Exception:
            self._embedding_model = None
        return self._embedding_model

    def index_10k_notes(self, notes: list[dict[str, str]] | None = None) -> int:
        """
        Index corporate 10-K financial statement notes for retrieval.

        Args:
            notes: Optional note records with section, title, and content fields.
                Uses built-in sample notes when omitted.

        Returns:
            Number of indexed chunks.
        """
        source_notes = notes if notes is not None else SAMPLE_10K_NOTES
        self._documents = self._build_document_chunks(source_notes)
        self._corpus = [
            f"{doc.section} {doc.title} {doc.content}" for doc in self._documents
        ]

        self._tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self._corpus)

        embedding_model = self._load_embedding_model()
        if embedding_model is not None:
            embeddings = embedding_model.encode(
                self._corpus,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._embedding_matrix = np.asarray(embeddings, dtype=float)
        else:
            self._embedding_matrix = None

        self._is_indexed = True
        return len(self._documents)

    def _ensure_indexed(self) -> None:
        """Build the default index when ask/search is called before indexing."""
        if not self._is_indexed:
            self.index_10k_notes()

    def _keyword_search(self, query: str) -> np.ndarray:
        """Return TF-IDF cosine similarity scores for each indexed chunk."""
        if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            raise RuntimeError("Index is not initialized.")
        query_vector = self._tfidf_vectorizer.transform([query])
        return cosine_similarity(query_vector, self._tfidf_matrix).flatten()

    def _semantic_search(self, query: str) -> np.ndarray:
        """Return embedding cosine similarity scores for each indexed chunk."""
        if self._embedding_matrix is None:
            return np.zeros(len(self._documents), dtype=float)
        embedding_model = self._load_embedding_model()
        if embedding_model is None:
            return np.zeros(len(self._documents), dtype=float)
        query_embedding = embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_embedding, dtype=float)
        return cosine_similarity(query_vector, self._embedding_matrix).flatten()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        search_mode: SearchMode = "hybrid",
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant 10-K note chunks for a query.

        Args:
            query: Natural-language finance question.
            top_k: Number of chunks to return.
            search_mode: One of hybrid, keyword, or semantic.

        Returns:
            Ranked list of RetrievedChunk objects.
        """
        self._ensure_indexed()
        limit = top_k or self.top_k

        keyword_scores = self._normalize_scores(self._keyword_search(query))
        semantic_scores = self._normalize_scores(self._semantic_search(query))

        if search_mode == "keyword":
            combined_scores = keyword_scores
        elif search_mode == "semantic":
            combined_scores = semantic_scores if semantic_scores.sum() > 0 else keyword_scores
        else:
            if semantic_scores.sum() > 0:
                combined_scores = (
                    self.keyword_weight * keyword_scores
                    + self.semantic_weight * semantic_scores
                )
            else:
                combined_scores = keyword_scores

        ranked_indices = np.argsort(combined_scores)[::-1][:limit]
        results: list[RetrievedChunk] = []
        for index in ranked_indices:
            score = float(combined_scores[index])
            if score <= 0:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._documents[int(index)],
                    score=score,
                    keyword_score=float(keyword_scores[int(index)]),
                    semantic_score=float(semantic_scores[int(index)]),
                )
            )
        return results

    def _resolve_provider(self) -> LLMProvider:
        """Select the highest-priority configured LLM provider."""
        if self.openai_api_key:
            return "openai"
        if self.gemini_api_key:
            return "gemini"
        if self.ollama_model:
            return "ollama"
        return "local"

    def _build_context(self, retrieved: list[RetrievedChunk]) -> str:
        """Format retrieved note excerpts for answer generation."""
        if not retrieved:
            return "No relevant 10-K note excerpts were retrieved."
        blocks = []
        for item in retrieved:
            blocks.append(
                f"[{item.chunk.source_label} | relevance={item.score:.2f}]\n{item.chunk.content}"
            )
        return "\n\n".join(blocks)

    def _synthesize_local_answer(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        """Generate a zero-API-key answer from retrieved 10-K excerpts."""
        if not retrieved:
            return (
                "I could not locate relevant 10-K note language for that question. "
                "Try rephrasing the question or indexing additional disclosures."
            )

        lead = retrieved[0].chunk.content
        supporting = [item.chunk.content for item in retrieved[1:]]
        answer_parts = [
            f"Based on indexed 10-K financial statement notes, here is an analysis of: {question}",
            "",
            f"Primary disclosure: {lead}",
        ]
        if supporting:
            answer_parts.append("")
            answer_parts.append("Supporting context:")
            for index, snippet in enumerate(supporting, start=1):
                answer_parts.append(f"{index}. {snippet}")
        answer_parts.append("")
        answer_parts.append(
            "This response was generated locally from retrieved note excerpts without an external LLM."
        )
        return "\n".join(answer_parts)

    def _generate_llm_answer(
        self,
        question: str,
        context: str,
        provider: LLMProvider,
    ) -> str:
        """Generate an answer using an optional external or open-source LLM."""
        prompt = (
            "You are a financial reporting analyst. Answer the question using ONLY the provided "
            "10-K note excerpts. Be concise, cite the relevant note section, and quantify drivers "
            "when available.\n\n"
            f"Question:\n{question}\n\n"
            f"10-K Note Excerpts:\n{context}\n\n"
            "Answer:"
        )

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=self.openai_api_key,
            )
            response = llm.invoke(prompt)
            return str(response.content)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=self.gemini_api_key,
            )
            response = llm.invoke(prompt)
            return str(response.content)

        if provider == "ollama":
            from langchain_community.chat_models import ChatOllama

            llm = ChatOllama(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=0.1,
            )
            response = llm.invoke(prompt)
            return str(response.content)

        return self._synthesize_local_answer(question, self.search(question))

    def ask(
        self,
        question: str,
        search_mode: SearchMode = "hybrid",
        top_k: int | None = None,
        force_local: bool = False,
    ) -> CopilotResponse:
        """
        Answer a finance variance question using retrieved 10-K note context.

        Args:
            question: User question, e.g. a pre-loaded variance prompt.
            search_mode: Retrieval strategy (hybrid, keyword, or semantic).
            top_k: Number of note chunks to retrieve.
            force_local: When True, skip external LLM calls even if keys exist.

        Returns:
            CopilotResponse with answer text, sources, and provider metadata.
        """
        retrieved = self.search(question, top_k=top_k, search_mode=search_mode)
        provider = "local" if force_local else self._resolve_provider()
        context = self._build_context(retrieved)

        if provider == "local":
            answer = self._synthesize_local_answer(question, retrieved)
        else:
            try:
                answer = self._generate_llm_answer(question, context, provider)
            except Exception as exc:
                answer = (
                    f"LLM provider '{provider}' failed ({exc}). "
                    "Falling back to local excerpt synthesis.\n\n"
                    + self._synthesize_local_answer(question, retrieved)
                )
                provider = "local"

        return CopilotResponse(
            question=question,
            answer=answer,
            sources=retrieved,
            provider=provider,
            search_mode=search_mode,
        )


def build_default_copilot(
    openai_api_key: str | None = None,
    gemini_api_key: str | None = None,
    ollama_model: str | None = None,
) -> FinancialCopilot:
    """
    Create and index a ready-to-use FinancialCopilot instance.

    Args:
        openai_api_key: Optional OpenAI API key.
        gemini_api_key: Optional Google Gemini API key.
        ollama_model: Optional local Ollama model name.

    Returns:
        Indexed FinancialCopilot instance.
    """
    copilot = FinancialCopilot(
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        ollama_model=ollama_model,
    )
    copilot.index_10k_notes()
    return copilot


if __name__ == "__main__":
    copilot = build_default_copilot()
    print("Indexed chunks:", len(copilot._documents))
    print("Sample questions:", copilot.get_sample_questions())
    for sample_question in copilot.get_sample_questions()[:2]:
        response = copilot.ask(sample_question)
        print("\nQuestion:", response.question)
        print("Provider:", response.provider)
        print("Top source:", response.sources[0].chunk.source_label if response.sources else "None")
        print("Answer preview:", response.answer[:220], "...")
