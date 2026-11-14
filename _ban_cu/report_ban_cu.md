# Báo cáo Lab 9 – SSA-VN (Modern NLP)
**MSSV:** 24DH190314 · **Họ tên:** Nguyễn Hoàng Phúc · **Hình thức:** cá nhân · **check_code:** `927D30B4` · seed 58202 · noise `code_mix` · topic_pair ['thi_diem', 'thu_vien_cntt']
**Cấu hình:** embed `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; backbone `xlm-roberta-base` (LoRA r=8, α=16); few-shot k=6; rag_top_k=2; ablation A4.

## 1. Bảng kết quả
**Phân loại chủ đề (test):**

|  | acc clean | F1 clean | acc noisy | F1 noisy |
|---|---|---|---|---|
| TF-IDF+LR | 0.653 | 0.634 | 0.508 | 0.487 |
| Embedding+LR | 0.636 | 0.620 | 0.614 | 0.593 |

**Truy hồi (`retr_test`):**

|  | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| BM25 – clean | 0.727 | 0.841 | 0.894 | 0.970 | 0.804 | 0.844 |
| BM25 – noisy | 0.644 | 0.818 | 0.894 | 0.962 | 0.750 | 0.801 |
| Dense – clean | 0.576 | 0.758 | 0.818 | 0.909 | 0.674 | 0.730 |
| Dense – noisy | 0.545 | 0.712 | 0.795 | 0.909 | 0.649 | 0.711 |
| Hybrid (α=0.7) – clean | 0.758 | 0.909 | 0.962 | 0.985 | 0.838 | 0.874 |
| Hybrid (α=0.7) – noisy | 0.758 | 0.894 | 0.947 | 0.977 | 0.832 | 0.867 |

**RAG / từ chối / prompting / Phần E:** answer_hit_rate = 0.879; abstain_rate_ood = 0.300; false_abstain_rate = 0.068; zero-shot = 0.400, 6-shot = 0.533 (mô hình `Qwen/Qwen2.5-1.5B-Instruct`, greedy). LoRA q,v = 294,912 tham số (1 ma trận 12,288 = 1/48 ma trận đầy đủ); DPO margin h = 8.40, loss = 2.25e-04; toy: P@5 = 0.40, R@5 = 1.00, RR = 0.50, nDCG@5 = 0.6934. Recall@3 trên 20 câu tự viết (hybrid) = 0.900.

## 2. Ba phát hiện chính
1. **Truy hồi.** Trên `retr_test` sạch, hệ tốt nhất là Hybrid (α=0.7) (MRR@10 = 0.838, Recall@3 = 0.909). Khi sang tập nhiễu `code_mix`, MRR@10 của BM25 giảm 0.054, dense giảm 0.025, hybrid giảm 0.006 – embedding bền với nhiễu hơn so với khớp từ khóa.
2. **Phân loại.** Tách theo mục khiến bài toán khó (TF-IDF+LR chỉ đạt acc 0.653 sạch / 0.508 nhiễu). Embedding đóng băng+LR đạt 0.636 / 0.614 (-0.017 / +0.106 so với TF-IDF).
3. **RAG và từ chối.** RAG trích xuất (top-2) đạt answer_hit_rate = 0.879 (chỉ top-1: 0.795) trên 132 câu; ngưỡng τ = 0.452 chốt trên dev từ chối 30% câu ngoài phạm vi nhưng từ chối nhầm 7% câu hợp lệ (câu *near_topic* khó tách hơn câu *totally_off*). Zero-shot / 6-shot LLM trên 30 câu: 0.400 / 0.533.

## 3. Ablation A4 – Quét trọng số hybrid alpha = 0.0, 0.1, ..., 1.0 trên tập dev
Chọn α trên dev theo MRR@10 → **α = 0.7**. Bảng dev (MRR@10): α=0.0: 0.699, α=0.1: 0.705, α=0.2: 0.711, α=0.3: 0.721, α=0.4: 0.729, α=0.5: 0.732, α=0.6: 0.737, α=0.7: 0.738, α=0.8: 0.718, α=0.9: 0.673, α=1.0: 0.608. Test (α=0 / chọn / α=1), MRR@10 sạch: 0.804 / 0.838 / 0.674; nhiễu: 0.750 / 0.832 / 0.649. BM25 mạnh ở khớp từ chính xác, dense mạnh ở diễn đạt khác từ; hybrid kết hợp hai thang điểm sau min-max (chi tiết số liệu ở notebook F1).

## 4. Hạn chế, rủi ro, cải tiến
Hạn chế: dữ liệu giả lập, KB chỉ 80 đoạn theo khuôn mẫu; dev/test nhỏ nên α, τ nhiều phương sai; chỉ một kiểu nhiễu. Rủi ro: ảo giác khi truy hồi sai (vẫn trả lời tự tin, ngưỡng τ chỉ chặn câu hoàn toàn ngoài lề); quyền riêng tư (câu hỏi chứa MSSV/hoàn cảnh cá nhân); thiên lệch với teencode/không dấu/phương ngữ. Sẽ làm tiếp: reranker (cross-encoder) và fine-tune embedding bằng hard negative cùng chủ đề; bộ phát hiện OOD; DPO với cặp trung thành/ảo giác; tăng cường nhiều kiểu nhiễu.

## 5. Nhật ký sử dụng AI
- **Công cụ:** Claude (Anthropic, qua ứng dụng Claude/Cowork).
- **AI hỗ trợ:** đọc đề và các lab tham khảo (Lab 7, Lab 8); dựng khung và viết code các phần C–F trong notebook; soạn lời giải tay E1–E3 để đối chiếu; viết 20 câu hỏi `own_queries.csv`; soạn nháp báo cáo này (số liệu được sinh tự động từ notebook).
- **Phần có sẵn từ giảng viên giữ nguyên:** `normalize`, `BM25`, các hàm metric truy hồi, baseline TF-IDF+LR, `save_results`.
- **Kiểm chứng:** mọi hàm tính tay (E1–E3) được kiểm bằng `assert` và đối chiếu thư viện (scikit-learn, SciPy, peft); không chọn siêu tham số bằng tập test (α, τ chốt trên dev).
- **Trách nhiệm:** người nộp chạy lại notebook (Restart & Run All) và tự giải thích từng dòng khi vấn đáp.

## 6. Phân công
Làm cá nhân: toàn bộ Phần A–F do MSSV 24DH190314 thực hiện.
