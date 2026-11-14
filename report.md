# Báo cáo Lab 9 – SSA-VN (Modern NLP)
**MSSV nhóm trưởng (thẻ chung): 24DH190314** · check_code `927D30B4` · seed 58202 · noise `code_mix` · topic_pair ['thi_diem', 'thu_vien_cntt']  
**Nhóm 3 sinh viên:** Nguyễn Hoàng Phúc (24DH190314), Nguyễn Bảo Duy (24DH190309), Lê Trần Nhật Huy (24DH190312)  
**Cấu hình thẻ:** embed `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; backbone `xlm-roberta-base` (LoRA r=8, alpha=16); few-shot k=6; rag_top_k=2; ablation A4.

## 1. Bảng kết quả
**Phân loại chủ đề (clf_test):**

| | acc sạch | F1 sạch | acc nhiễu | F1 nhiễu |
|---|---|---|---|---|
| TF-IDF+LR | 0.653 | 0.634 | 0.508 | 0.487 |
| Embedding+LR | 0.636 | 0.620 | 0.614 | 0.593 |

LoRA (C4, nâng cao): chưa chạy vì không có GPU.

**Truy hồi (retr_test):**

| | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| BM25 – sạch | 0.727 | 0.841 | 0.894 | 0.970 | 0.804 | 0.844 |
| BM25 – nhiễu | 0.644 | 0.818 | 0.894 | 0.962 | 0.750 | 0.801 |
| Dense – sạch | 0.576 | 0.758 | 0.818 | 0.909 | 0.674 | 0.730 |
| Dense – nhiễu | 0.545 | 0.712 | 0.795 | 0.909 | 0.649 | 0.711 |
| Hybrid α=0.7 – sạch | 0.758 | 0.909 | 0.962 | 0.985 | 0.838 | 0.874 |
| Hybrid α=0.7 – nhiễu | 0.758 | 0.894 | 0.947 | 0.977 | 0.832 | 0.867 |

**LLM và RAG:** zero-shot = 0.400, 6-shot = 0.533 (`Qwen/Qwen2.5-1.5B-Instruct`, temperature 0, 30 câu); answer_hit_rate = 0.879 (132 câu, top-2); ngưỡng τ = 0.452: abstain_rate_ood = 0.300, false_abstain_rate = 0.068. Recall@3 trên 20 câu tự viết = 0.900.

**Phần E (mỗi thành viên một thẻ, code khớp lời giải tay):**

| Thành viên | E1: tham số LoRA | E2: margin h | E2: loss | P@5 | R@5 | RR | nDCG@5 |
|---|---|---|---|---|---|---|---|
| Nguyễn Hoàng Phúc (24DH190314) | 294,912 | 8.40 | 0.0002248 | 0.40 | 1.00 | 0.50 | 0.6934 |
| Nguyễn Bảo Duy (24DH190309) | 294,912 | 1.31 | 0.2389 | 0.40 | 1.00 | 1.00 | 0.8503 |
| Lê Trần Nhật Huy (24DH190312) | 147,456 | 1.80 | 0.153 | 0.20 | 0.50 | 0.25 | 0.2641 |

## 2. Ba phát hiện chính
1. **Hybrid tốt hơn từng thành phần và bền với nhiễu.** MRR@10 trên retr_test sạch: BM25 0.804, dense 0.674, hybrid 0.838. Sang tập nhiễu `code_mix`, BM25 giảm 0.054, dense giảm 0.025, hybrid chỉ giảm 0.006.
2. **Nhiễu code_mix làm hỏng mô hình khớp từ, ít ảnh hưởng embedding.** TF-IDF+LR giảm accuracy từ 0.653 xuống 0.508 vì từ tiếng Anh là từ lạ; Embedding+LR chỉ giảm từ 0.636 xuống 0.614. Tách theo mục khiến bài toán phân loại khó: lỗi tập trung ở các mục test có từ vựng chưa gặp trong train.
3. **RAG trả lời đúng 88% số câu nhưng ngưỡng từ chối còn yếu.** Ngưỡng τ chốt trên dev chỉ chặn được 30% câu ngoài phạm vi (các câu hoàn toàn lạc đề) và từ chối nhầm 6.8% câu hợp lệ; câu ngoài phạm vi nhưng gần chủ đề có cosine cao như câu hợp lệ.

## 3. Ablation A4 – Quét trọng số hybrid alpha = 0.0, 0.1, ..., 1.0 trên tập dev
MRR@10 trên dev theo α: 0.0: 0.699, 0.1: 0.705, 0.2: 0.711, 0.3: 0.721, 0.4: 0.729, 0.5: 0.732, 0.6: 0.737, 0.7: 0.738, 0.8: 0.718, 0.9: 0.673, 1.0: 0.608 → chọn **α = 0.7**.
Trên retr_test sạch: bỏ dense (α=0) 0.804; hybrid 0.838; bỏ BM25 (α=1) 0.674. Bỏ BM25 mất 0.164, bỏ dense mất 0.034: BM25 quan trọng hơn vì câu hỏi hay nhắc đúng tên thủ tục, dense bổ sung cho các câu nói đời thường và từ tiếng Anh.

## 4. Hạn chế, rủi ro, cải tiến
Hạn chế: dữ liệu giả lập, KB 80 đoạn theo khuôn mẫu; dev/test nhỏ nên α, τ dao động; RAG trích xuất bằng regex phụ thuộc cách viết của đoạn. Rủi ro: ảo giác – khi truy hồi sai, hệ thống vẫn trả lời tự tin bằng thông tin của quy định khác. Cải tiến: fine-tune mô hình embedding bằng cặp (câu hỏi, đoạn đúng) của train để cải thiện dense, hybrid và ngưỡng từ chối.

## 5. Nhật ký sử dụng AI
- **Công cụ:** Claude (Anthropic).
- **AI hỗ trợ:** đọc đề; viết và rút gọn code các phần C–F trong notebook; soạn lời giải tham khảo E1–E3 để đối chiếu; gợi ý 20 câu hỏi `own_queries.csv`; soạn nháp báo cáo (số liệu sinh tự động từ notebook).
- **Giữ nguyên code của giảng viên:** `normalize`, `BM25`, các hàm metric truy hồi, baseline TF-IDF+LR, `save_results`.
- **Nhóm tự làm và kiểm chứng:** chạy lại toàn bộ notebook, làm tay E1–E3 theo thẻ riêng, đọc và phân loại nguyên nhân các câu sai; α và τ chỉ chọn trên dev.

## 6. Phân công công việc
| Họ tên | MSSV | Vai trò | Phần phụ trách |
|---|---|---|---|
| Nguyễn Hoàng Phúc | 24DH190314 | Nhóm trưởng | Phần A, B; ghép notebook, results.json, báo cáo; Phần E theo thẻ riêng |
| Nguyễn Bảo Duy | 24DH190309 | Thành viên | Phần C (dense, hybrid, phân loại embedding) và F1 (ablation); Phần E theo thẻ riêng |
| Lê Trần Nhật Huy | 24DH190312 | Thành viên | Phần D (prompting, RAG, từ chối) và F2–F4; Phần E theo thẻ riêng |

Mọi thành viên đều chạy lại được notebook và giải thích được tất cả các phần.
