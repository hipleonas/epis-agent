"""Câu từ chối CỐ ĐỊNH cho gia sư AI — KHÔNG bao giờ được LLM sinh ra.

Đây là chốt chặn "zero rò rỉ" của luồng retrieve-first: khi bộ phân loại (`RAGService.analyze`)
kết luận câu hỏi thuộc `out_of_corpus` — không đoạn tài liệu nào trong CẢ khóa học nói về nội dung
được hỏi — agent trả thẳng hằng số này, không gọi thêm lời sinh câu trả lời nào, nên không có đường
nào để kiến thức chung của LLM lọt vào.

`cross_module` KHÔNG còn là từ chối: nội dung có thật trong khóa học, chỉ nằm ở module khác, nên hệ
thống vẫn trả lời (ngữ cảnh gồm cả chunk của module đó) rồi gợi ý học sinh mở module tương ứng —
xem `RAGService._context_for` và `ChatService.buildModuleSuggestion` bên gsai-server.

QUAN TRỌNG: câu từ chối phải chứa các cụm mà `eval/epislib/judge.py::_REFUSAL_MARKERS` dùng để
nhận diện từ chối ("xin lỗi", "không tìm thấy"/"không có thông tin", "không thể" …). Nếu đổi lời
văn, đối chiếu lại `_REFUSAL_MARKERS` — nếu không, chỉ số `correct_refusal_rate` của bộ đo sẽ hỏng
âm thầm dù hành vi thực tế đã đúng.
"""

REFUSAL_OUT_OF_CORPUS = (
    "Xin lỗi, mình không tìm thấy thông tin về nội dung này trong tài liệu của khóa học, "
    "nên mình không thể trả lời. Bạn hãy thử hỏi về nội dung đang học trong module nhé."
)
