"""RAG (Retrieval-Augmented Generation) module."""
import json
import os
from typing import Optional, List, Dict, Generator, Tuple
from openai import OpenAI
from src.services.embeddings import EmbeddingService
from src.services.vectorstore import ZillizVectorStore
from src.services.llm_client_factory import build_llm_client
from src.utils.prompt_loader import read_prompt

# Prompt mặc định cho luồng RAG — sống ở một nơi duy nhất, dùng chung cho cả generate_answer()
# (non-stream) lẫn generate_answer_stream() (stream) qua _build_rag_messages().
_DEFAULT_RAG_SYSTEM_PROMPT = read_prompt(
    "teacher_agent", "rag_system_prompt.txt",
    """Bạn là một giáo viên hữu ích, nhiệt tình và giàu kinh nghiệm.
Nhiệm vụ của bạn là giúp học sinh học tập dựa trên tài liệu khóa học được cung cấp.

Hướng dẫn:
- Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu và NGẮN GỌN — đi thẳng vào điều học sinh hỏi
- Với câu hỏi định nghĩa/khái niệm đơn giản ("X là gì?", "X khác Y thế nào?"), trả lời trong
  khoảng 2-5 câu; KHÔNG tự thêm những phần học sinh không hỏi (quy tắc đặt tên, phân loại chi
  tiết, quy trình các bước, lời chào/khuyến khích ở cuối)
- Chỉ mở rộng dài hơn khi câu hỏi vốn cần nhiều bước giải thích, hoặc học sinh hỏi thêm/yêu cầu
  chi tiết hơn
- Sử dụng thông tin từ tài liệu được cung cấp để trả lời
- Nếu thông tin không có trong tài liệu, hãy nói rõ điều đó
- Giải thích các khái niệm phức tạp một cách đơn giản
- Chỉ đưa ví dụ hoặc câu hỏi ôn tập khi thực sự cần thiết để làm rõ câu trả lời, không thêm mặc định""",
)

_CLASSIFY_PROMPT = read_prompt(
    "teacher_agent", "classify_prompt.txt",
    'Phân loại câu hỏi dựa trên tài liệu truy xuất được thành "chat", "learning", '
    '"cross_module" hoặc "out_of_corpus". Chỉ trả JSON '
    '{"label": "...", "suggested_module_id": null}.',
)

_VALID_LABELS = ("chat", "learning", "cross_module", "out_of_corpus")


class RAGService:
    """Service for RAG-based question answering."""

    def __init__(self, llm_provider: str = "open_ai"):
        """Initialize RAG service with embedding and vector store."""
        self.embedding_service = EmbeddingService()
        self.vector_store = ZillizVectorStore()
        self.llm_client, self.model = build_llm_client(llm_provider)
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        # Ngưỡng cosine chỉ để LỌC RÁC khỏi ngữ cảnh của bộ phân loại — KHÔNG phải ranh giới
        # quyết định từ chối/trả lời (đó là việc của _classify_with_evidence). Chọn 0.30 vì dữ
        # liệu đo cho thấy chunk hợp lệ thấp nhất ~0.44-0.45, còn chunk out-of-corpus/cross-module
        # thấp nhất ~0.27-0.32 — 0.30 loại rác mà không cắt nhầm chunk hợp lệ nào.
        self.min_score = float(os.getenv("RAG_MIN_SCORE", "0.30"))
        # Số chunk lấy khi quét TOÀN khóa (module_id=None) để tìm bằng chứng cross-module. Cao hơn
        # nhiều so với top_k trong-module vì top-5 toàn khóa có thể chỉ toàn chunk của chính module
        # hiện tại, không còn gì để phát hiện cross-module.
        self.cross_top_k = int(os.getenv("RAG_CROSS_TOP_K", "20"))
        # Biên an toàn cho quyết định "cross_module": chỉ CHẤP NHẬN nhãn này nếu điểm cosine cao
        # nhất bên module khác vượt điểm cao nhất module hiện tại ít nhất ngần này — xem
        # `_classify_with_evidence`. Phát hiện từ vụ tài liệu "Điều kiện" (M3) bị upload nhầm, trùng
        # nội dung với "Lập trình là gì?" (M1): điểm cosine của chunk trùng lặp gần như giống hệt
        # chunk gốc, khiến LLM phân loại chọn nhầm "cross_module" dù module hiện tại đã đủ trả lời
        # (evaluated 2026-07-28, eval/MEMORY.md §7 — 8/33 câu in-corpus bị gán nhầm nhãn này).
        self.cross_module_margin = float(os.getenv("RAG_CROSS_MODULE_MARGIN", "0.05"))

    def retrieve(
        self,
        query: str,
        course_id: Optional[str] = None,
        module_id: Optional[str] = None,
        top_k: int = 5,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict]:
        """Retrieve relevant documents from vector store.

        Args:
            query: User's question or search query.
            course_id: Filter by course ID (optional).
            module_id: Filter by module ID (optional).
            top_k: Number of results to retrieve.
            query_embedding: Pre-computed embedding of `query` (optional). Pass this in when the
                caller already embedded the query (e.g. to search multiple scopes without
                re-embedding, see `analyze()`); if omitted, the query is embedded here as before.

        Returns:
            List of retrieved documents with metadata.
        """
        if query_embedding is None:
            query_embedding = self.embedding_service.embed_text(query)

        # Search in vector store with filters
        results = self.vector_store.search(
            query_embedding=query_embedding,
            course_id=course_id,
            module_id=module_id,
            top_k=top_k
        )

        return results

    def generate_answer(
        self,
        query: str,
        context_docs: List[Dict],
        history: Optional[List[Dict]] = None,
        system_prompt: str = None
    ) -> str:
        """Generate answer using LLM with retrieved context and conversation history.

        Args:
            query: User's question.
            context_docs: Retrieved documents from vector store.
            history: Conversation history (optional).
            system_prompt: Custom system prompt (optional).

        Returns:
            Generated answer from LLM.
        """
        messages = self._build_rag_messages(query, context_docs, history, system_prompt)

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        return response.choices[0].message.content

    def answer_question(
        self,
        query: str,
        context_docs: List[Dict],
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        """Generate a RAG answer for ALREADY-RETRIEVED, already-decided-relevant documents.

        The retrieve-then-classify decision (`analyze()`) is the single place that decides
        whether a question gets answered at all; by the time this is called, `context_docs` is
        exactly what should ground the answer. There is deliberately no "if not context_docs"
        fallback here anymore — an empty/irrelevant retrieval is handled upstream by refusing
        before generation is ever attempted (see `analyze()` / `refusals.py`).

        Args:
            query: User's question.
            context_docs: Retrieved documents to ground the answer in (from `analyze()`).
            history: Conversation history (optional).

        Returns:
            Dictionary with answer and sources.
        """
        answer = self.generate_answer(query, context_docs, history=history)
        return {
            "answer": answer,
            "sources": context_docs,
        }

    # ------------------------------------------------------------------
    # Retrieve-first classification
    # ------------------------------------------------------------------

    def analyze(
        self,
        query: str,
        course_id: Optional[str] = None,
        module_id: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        top_k: int = 5,
    ) -> Dict:
        """Retrieve first, then classify using the retrieved chunks as evidence.

        Thay vì "phân loại ý định rồi mới truy xuất", hàm này truy xuất TRƯỚC, rồi dùng chính các
        chunk truy xuất được (cả trong module hiện tại lẫn các module khác trong cùng khóa) làm
        bằng chứng cho một lời gọi LLM phân loại duy nhất. Điều này cho phép phân biệt:
          - "learning": module hiện tại có đủ thông tin để trả lời -> sinh câu trả lời RAG.
          - "cross_module": module khác trong khóa có thông tin -> VẪN trả lời (ngữ cảnh gồm cả
            chunk của module kia), kèm gợi ý mở module đó; server nối câu gợi ý vào cuối.
          - "out_of_corpus": không đâu trong khóa có thông tin -> từ chối, không gợi ý.
          - "chat": không cần nội dung môn học (chào hỏi, hỏi cách dùng hệ thống).

        Returns:
            {
              "label": "chat"|"learning"|"cross_module"|"out_of_corpus",
              "intent": "chat"|"learning" (hình chiếu 2 giá trị của label, để tương thích ngược),
              "suggested_module_id": Optional[str],
              "docs": List[Dict]  (ngữ cảnh DÙNG ĐỂ SINH câu trả lời và làm `sources` — xem
                       `_context_for`; luôn [] với "chat" và "out_of_corpus"),
            }
        """
        # Embed một lần, dùng cho cả hai lượt truy xuất bên dưới.
        query_embedding = self.embedding_service.embed_text(query)

        in_docs = [
            d for d in self.retrieve(query, course_id, module_id, top_k, query_embedding=query_embedding)
            if (d.get("score") or 0.0) >= self.min_score
        ]

        cross_docs: List[Dict] = []
        if course_id:
            course_hits = self.retrieve(
                query, course_id, None, self.cross_top_k, query_embedding=query_embedding
            )
            current = (module_id or "").strip()
            cross_docs = [
                d for d in course_hits
                if (d.get("score") or 0.0) >= self.min_score
                and (d.get("module_id") or "").strip() != current
            ][:top_k]

        label, suggested_module_id = self._classify_with_evidence(
            query, in_docs, cross_docs, history, module_id
        )

        return {
            "label": label,
            "intent": "learning" if label in ("learning", "cross_module", "out_of_corpus") else "chat",
            "suggested_module_id": suggested_module_id,
            "docs": self._context_for(label, in_docs, cross_docs),
        }

    @staticmethod
    def _context_for(label: str, in_docs: List[Dict], cross_docs: List[Dict]) -> List[Dict]:
        """Ngữ cảnh dùng để sinh câu trả lời, theo từng nhãn.

        - "learning": chỉ chunk của module hiện tại (cách ly module như cũ).
        - "cross_module": chunk module hiện tại GHÉP chunk của module chứa nội dung. Nội dung được
          hỏi nằm ở module kia, nên nếu chỉ đưa chunk module hiện tại thì câu trả lời sẽ rỗng nghĩa
          — mà luồng này đã bỏ từ chối, buộc phải trả lời có cơ sở. Ưu tiên chunk module hiện tại
          đứng trước để câu trả lời bám ngữ cảnh học sinh đang học.
        - còn lại ("chat", "out_of_corpus"): không có ngữ cảnh.
        """
        if label == "learning":
            return in_docs
        if label != "cross_module":
            return []
        merged: List[Dict] = []
        seen = set()
        for d in list(in_docs) + list(cross_docs):
            key = d.get("id")
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            merged.append(d)
        return merged

    def _classify_with_evidence(
        self,
        query: str,
        in_docs: List[Dict],
        cross_docs: List[Dict],
        history: Optional[List[Dict]],
        module_id: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        """Single LLM call: classify using retrieved chunks (both scopes) as evidence."""
        user_block = self._build_evidence_block(query, in_docs, cross_docs, history, module_id)
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _CLASSIFY_PROMPT},
                    {"role": "user", "content": user_block},
                ],
                temperature=0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception:
            # Không bao giờ để lỗi phân loại làm hỏng cả lượt chat — lùi về mặc định an toàn.
            return ("learning" if in_docs else "out_of_corpus"), None

        label = str(data.get("label", "")).strip().lower()
        if label not in _VALID_LABELS:
            label = "learning" if in_docs else "out_of_corpus"

        # Chốt chặn quyết định: KHÔNG tin "cross_module" trừ khi bằng chứng module khác THỰC SỰ
        # nổi trội hơn rõ rệt so với module hiện tại. LLM phân loại thỉnh thoảng bị nhiễu bởi chunk
        # cross-module có điểm cosine gần bằng (hoặc do lỗi dữ liệu — tài liệu upload nhầm trùng nội
        # dung module khác — gần như giống hệt) chunk trong-module, dù module hiện tại đã đủ để trả
        # lời. Vì in_docs không rỗng nghĩa là module hiện tại đã có bằng chứng đạt ngưỡng min_score,
        # hạ về "learning" an toàn hơn: câu trả lời vẫn đúng module đang học, chỉ mất câu gợi ý chéo
        # module (tiện ích phụ, không ảnh hưởng tính đúng của nội dung).
        if label == "cross_module" and in_docs:
            top_in = max((d.get("score") or 0.0) for d in in_docs)
            top_cross = max((d.get("score") or 0.0) for d in cross_docs) if cross_docs else 0.0
            if top_cross <= top_in + self.cross_module_margin:
                label = "learning"

        suggested = data.get("suggested_module_id")
        suggested = str(suggested).strip() if suggested else None

        if label == "cross_module":
            # Chốt chặn: KHÔNG chấp nhận module_id mà LLM tự bịa — phải nằm trong chính các chunk
            # đã truy xuất được từ các module khác.
            valid_ids = {(d.get("module_id") or "").strip() for d in cross_docs}
            valid_ids.discard("")
            if suggested not in valid_ids:
                # Dự phòng: lấy module của chunk ĐIỂM CAO NHẤT. Phải duyệt theo thứ tự cross_docs
                # (Milvus trả về đã xếp hạng giảm dần) chứ KHÔNG lấy phần tử tuỳ ý của `valid_ids`
                # — set không có thứ tự và hash của str được randomize mỗi tiến trình, nên cùng một
                # đầu vào có thể gợi ý module khác nhau giữa các lần chạy.
                suggested = next(
                    ((d.get("module_id") or "").strip() for d in cross_docs
                     if (d.get("module_id") or "").strip()),
                    None,
                )
            if suggested is None:
                # Không có gì cụ thể để gợi ý -> hạ về out_of_corpus (vẫn từ chối, không gợi ý sai).
                label = "out_of_corpus"
        else:
            suggested = None

        return label, suggested

    def _build_evidence_block(
        self,
        query: str,
        in_docs: List[Dict],
        cross_docs: List[Dict],
        history: Optional[List[Dict]],
        module_id: Optional[str],
    ) -> str:
        """Build the user-turn evidence block for `_classify_with_evidence`."""
        parts: List[str] = []

        if history:
            recent = history[-3:] if len(history) > 3 else history
            if recent:
                parts.append("=== NGỮ CẢNH HỘI THOẠI GẦN ĐÂY ===")
                for msg in recent:
                    speaker = "học sinh" if msg.get("role") == "user" else "trợ giảng"
                    parts.append(f"{speaker}: {msg.get('content', '')}")
                parts.append("")

        parts.append(f"=== ĐOẠN TÀI LIỆU TỪ MODULE HIỆN TẠI (module_id = {module_id}) ===")
        if in_docs:
            for i, d in enumerate(in_docs[:5], 1):
                parts.append(f"[H{i}] (điểm tương đồng {d.get('score', 0):.2f}) "
                             f"nguồn: {d.get('from_src', '')}")
                parts.append((d.get("text") or "")[:700])
        else:
            parts.append("(KHÔNG có đoạn nào đủ liên quan trong module hiện tại)")
        parts.append("")

        parts.append("=== ĐOẠN TÀI LIỆU TỪ CÁC MODULE KHÁC TRONG CÙNG KHÓA HỌC ===")
        if cross_docs:
            for i, d in enumerate(cross_docs[:5], 1):
                parts.append(f"[K{i}] module_id = {d.get('module_id', '')} "
                             f"(điểm tương đồng {d.get('score', 0):.2f}) nguồn: {d.get('from_src', '')}")
                parts.append((d.get("text") or "")[:700])
        else:
            parts.append("(KHÔNG có đoạn nào đủ liên quan ở các module khác)")
        parts.append("")

        parts.append("=== CÂU HỎI CỦA HỌC SINH ===")
        parts.append(query)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Streaming variants
    # ------------------------------------------------------------------

    def _build_rag_messages(
        self,
        query: str,
        context_docs: List[Dict],
        history: Optional[List[Dict]] = None,
        system_prompt: str = None
    ) -> list:
        """Build the messages list for a RAG call (shared by stream and non-stream)."""
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[Tài liệu {i}]")
            if doc.get("course_id"):
                context_parts.append(f"Khóa học: {doc['course_id']}")
            if doc.get("module_id"):
                context_parts.append(f"Module: {doc['module_id']}")
            context_parts.append(f"Nội dung: {doc['text']}")
            context_parts.append("")

        context = "\n".join(context_parts)

        if not system_prompt:
            system_prompt = _DEFAULT_RAG_SYSTEM_PROMPT

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            recent_history = history[-15:] if len(history) > 15 else history
            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": f"""Dựa trên các tài liệu sau:

{context}

Câu hỏi: {query}

Hãy trả lời câu hỏi của học sinh một cách ngắn gọn, đúng trọng tâm và dễ hiểu."""})

        return messages

    def generate_answer_stream(
        self,
        query: str,
        context_docs: List[Dict],
        history: Optional[List[Dict]] = None,
        system_prompt: str = None
    ) -> Generator[str, None, None]:
        """Stream answer token-by-token using OpenAI streaming.

        Yields:
            String chunks (text deltas) as they arrive from OpenAI.
        """
        messages = self._build_rag_messages(query, context_docs, history, system_prompt)

        stream = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def answer_question_stream(
        self,
        query: str,
        context_docs: List[Dict],
        history: Optional[List[Dict]] = None,
    ) -> Generator[Dict, None, None]:
        """Streaming RAG answer for ALREADY-RETRIEVED, already-decided-relevant documents.

        Mirror of `answer_question()` for the SSE path — see its docstring for why there is no
        "if not context_docs" fallback here: that decision belongs to `analyze()`, upstream.

        Yields:
            {"type": "token", "delta": "..."} for each text chunk.
            {"type": "done", "sources": [...]}  as the final item.
        """
        for text_delta in self.generate_answer_stream(query, context_docs, history=history):
            yield {"type": "token", "delta": text_delta}

        yield {"type": "done", "sources": context_docs}

