# Lab 9 – Phần E (E1–E3): lời giải tay cho 3 thành viên

**Giả định E1 (nêu rõ khi nộp):** cả `xlm-roberta-base` và `vinai/phobert-base-v2` là cỡ "base" → **L = 12 lớp, d = hidden_size = 768**
(đúng bảng `CLF_BACKBONES` trong `lab9_tools.py`). Kiểm lại bằng:
`cfg = AutoConfig.from_pretrained("<tên backbone>"); cfg.num_hidden_layers, cfg.hidden_size`.

Công thức chung
- E1: 1 ma trận LoRA = A (r×d) + B (d×r) = **2·r·d**; 1 lớp = n_targets·2rd; tổng = L·n_targets·2rd; so với d².
- E2: h = β·[(logπ_w − logref_w) − (logπ_l − logref_l)]; loss = −ln σ(h) = ln(1 + e^(−h)); gradient ∝ β·σ(−h).
- E3: P@5 = (#liên quan trong top 5)/5; R@5 = (#liên quan trong top 5)/(#liên quan tổng); RR = 1/hạng của mục liên quan đầu tiên;
  DCG@5 = Σ rel_i / log2(i+1); IDCG@5 = DCG của min(#liên quan, 5) mục xếp đầu; nDCG@5 = DCG/IDCG.

---------------------------------------------------------------------
## Thành viên 1 — Nguyễn Hoàng Phúc — MSSV 24DH190314 — check_code 927D30B4

### E1
(a) Đề cho: xlm-roberta-base (L=12, d=768 – giả định base); r=8, alpha=16; target = {query, value} → n_targets = 2.
(b) 1 ma trận = 2rd; 1 lớp = 2·(2rd); tổng = 12·(1 lớp); đối chiếu d² = 768².
(c) Thay số:
- 1 ma trận: 2·8·768 = 16·768 = **12 288**
- 1 lớp: 2 × 12 288 = **24 576**
- 12 lớp: 12 × 24 576 = **294 912**
- d² = 768·768 = **589 824**
- 1 ma trận LoRA / d² = 12 288 / 589 824 = 1/48 ≈ **2,0833 %**
- tổng LoRA / d² = 294 912 / 589 824 = 1/2 = **50 %**
(d) Kết quả: tổng **294 912** tham số LoRA ≈ **một nửa** một ma trận đầy đủ 768×768.
(e) Mỗi ma trận LoRA chỉ bằng 2r/d = 16/768 = 1/48 ma trận gốc; cả 24 ma trận LoRA cộng lại mới bằng 1/2 một ma trận d×d. (alpha chỉ đổi hệ số α/r = 2, không đổi số tham số.)

### E2
(a) Đề cho: β = 0,5; logπ_chosen = −18,5; logπ_rejected = −42,8; logref_chosen = −24,8; logref_rejected = −32,3.
(b) h = β·[(logπ_w − logref_w) − (logπ_l − logref_l)]; loss = ln(1 + e^(−h)).
(c) Thay số:
- chosen: −18,5 − (−24,8) = **+6,3**
- rejected: −42,8 − (−32,3) = **−10,5**
- hiệu: 6,3 − (−10,5) = 16,8
- h = 0,5 × 16,8 = **8,4**
- e^(−8,4) = e^(−8)·e^(−0,4) = 3,3546·10⁻⁴ × 0,67032 = 2,2487·10⁻⁴
- loss = ln(1 + 2,2487·10⁻⁴) ≈ 2,2487·10⁻⁴ − (2,2487·10⁻⁴)²/2 = **2,248·10⁻⁴** (≈ 0,0002248)
- σ(−h) = 1/(1+e^(8,4)) ≈ 2,248·10⁻⁴
(d) Kết quả: h = 8,4 (> 0); loss ≈ 2,248·10⁻⁴.
(e) h > 0: so với mô hình tham chiếu, policy đã tăng xác suất câu chosen (+6,3) và giảm xác suất câu rejected (−10,5) → đã "ưa thích đúng". Loss ≈ 0 và gradient ∝ β·σ(−h) ≈ 1,1·10⁻⁴ → ví dụ này gần như đã học xong, hầu như không còn cập nhật.

### E3  (toy_ranking = [0,1,1,0,0,0,0,0])
(a) Đề cho: 8 mục, k = 5; mục liên quan ở hạng 2 và 3 → tổng 2 mục liên quan.
(b) Công thức như phần chung.
(c) Thay số:
- trong top 5: 2 mục liên quan → P@5 = 2/5 = **0,4**; R@5 = 2/2 = **1,0**
- mục liên quan đầu tiên ở hạng 2 → RR = **0,5**
- DCG@5 = 1/log2(3) + 1/log2(4) = 1/1,58496 + 1/2 = 0,63093 + 0,5 = **1,13093**
- IDCG@5 (min(2,5) = 2 mục đầu) = 1/log2(2) + 1/log2(3) = 1 + 0,63093 = **1,63093**
- nDCG@5 = 1,13093 / 1,63093 = **0,6934**
(d) P@5 = 0,4; R@5 = 1,0; RR = 0,5; nDCG@5 = 0,6934.
(e) Cả 2 mục liên quan đều vào top 5 nên recall đầy đủ, nhưng chúng nằm ở hạng 2–3 chứ không phải 1–2 nên RR = 0,5 và nDCG < 1.

---------------------------------------------------------------------
## Thành viên 2 — Nguyễn Bảo Duy — MSSV 24DH190309 — check_code CD4A20E5

### E1
(a) xlm-roberta-base (L=12, d=768 – giả định base); r=8, alpha=16; target {query, value}.
(b) như trên.
(c) 1 ma trận: 2·8·768 = **12 288**; 1 lớp: 2×12 288 = **24 576**; 12 lớp: **294 912**; d² = **589 824**;
1 ma trận/d² = 1/48 ≈ **2,0833 %**; tổng/d² = 1/2 = **50 %**.
(d) Tổng **294 912** tham số LoRA = 50 % một ma trận 768×768.
(e) Cùng backbone và cùng r với Phúc nên kết quả giống hệt; số tham số không phụ thuộc alpha hay dropout.

### E2
(a) β = 0,1; logπ_chosen = −21,7; logπ_rejected = −33,1; logref_chosen = −23,6; logref_rejected = −21,9.
(b) như trên.
(c) Thay số:
- chosen: −21,7 − (−23,6) = **+1,9**
- rejected: −33,1 − (−21,9) = **−11,2**
- hiệu: 1,9 − (−11,2) = 13,1
- h = 0,1 × 13,1 = **1,31**
- e^(−1,31) = 0,26982
- loss = ln(1 + 0,26982) = ln 1,26982 = **0,2389**
- σ(−h) = 0,26982 / 1,26982 = 0,2125
(d) h = 1,31 (> 0); loss ≈ 0,2389.
(e) h > 0: policy ưa chosen hơn rejected so với ref (chosen tăng nhẹ +1,9, rejected giảm mạnh −11,2 – phần lớn "biên độ" đến từ việc hạ rejected). Loss còn 0,24 và gradient ∝ β·σ(−h) = 0,1×0,2125 ≈ 0,021 → còn học được thêm, chưa bão hòa như trường hợp β lớn.

### E3  (toy_ranking = [1,0,0,0,1,0,0,0])
(a) 8 mục, k = 5; mục liên quan ở hạng 1 và 5 → tổng 2.
(c) Thay số:
- top 5 có 2 mục liên quan → P@5 = 2/5 = **0,4**; R@5 = 2/2 = **1,0**
- mục liên quan đầu ở hạng 1 → RR = **1,0**
- DCG@5 = 1/log2(2) + 1/log2(6) = 1 + 1/2,58496 = 1 + 0,38685 = **1,38685**
- IDCG@5 = 1 + 0,63093 = **1,63093**
- nDCG@5 = 1,38685 / 1,63093 = **0,8503**
(d) P@5 = 0,4; R@5 = 1,0; RR = 1,0; nDCG@5 = 0,8503.
(e) Mục đầu xếp hạng 1 (RR tối đa) nhưng mục thứ hai bị đẩy xuống hạng 5 (lý tưởng là hạng 2) nên nDCG chỉ 0,85.

---------------------------------------------------------------------
## Thành viên 3 — Lê Trần Nhật Huy — MSSV 24DH190312 — check_code 74695247

### E1
(a) vinai/phobert-base-v2 (L=12, d=768 – giả định base, kiến trúc RoBERTa); r=4, alpha=8; target {query, value}.
(c) Thay số:
- 1 ma trận: 2·4·768 = **6 144**
- 1 lớp: 2 × 6 144 = **12 288**
- 12 lớp: 12 × 12 288 = **147 456**
- d² = **589 824**
- 1 ma trận/d² = 6 144/589 824 = 1/96 ≈ **1,0417 %**
- tổng/d² = 147 456/589 824 = 1/4 = **25 %**
(d) Tổng **147 456** tham số LoRA = 25 % một ma trận 768×768.
(e) r nhỏ một nửa so với hai bạn kia nên số tham số cũng giảm một nửa (2rd tuyến tính theo r).

### E2
(a) β = 0,1; logπ_chosen = −20,6; logπ_rejected = −38,1; logref_chosen = −22,8; logref_rejected = −22,3.
(c) Thay số:
- chosen: −20,6 − (−22,8) = **+2,2**
- rejected: −38,1 − (−22,3) = **−15,8**
- hiệu: 2,2 − (−15,8) = 18,0
- h = 0,1 × 18,0 = **1,8**
- e^(−1,8) = 0,16530
- loss = ln(1 + 0,16530) = ln 1,16530 = **0,1530**
- σ(−h) = 0,16530 / 1,16530 = 0,1419
(d) h = 1,8 (> 0); loss ≈ 0,1530.
(e) h > 0: policy đã ưa chosen hơn rejected so với ref. Lưu ý hiệu log-ratio (18,0) lớn hơn của Phúc (16,8) nhưng h nhỏ hơn nhiều vì β = 0,1 thay vì 0,5 → β quyết định độ lớn h, loss và gradient.

### E3  (toy_ranking = [0,0,0,1,0,0,0,1])
(a) 8 mục, k = 5; mục liên quan ở hạng 4 và 8 → tổng 2.
(c) Thay số:
- top 5 chỉ có hạng 4 → P@5 = 1/5 = **0,2**; R@5 = 1/2 = **0,5**
- mục liên quan đầu ở hạng 4 → RR = 1/4 = **0,25**
- DCG@5 = 1/log2(5) = 1/2,32193 = **0,43068** (mục hạng 8 nằm ngoài top 5, không tính)
- IDCG@5 = 1 + 0,63093 = **1,63093** (vẫn lấy min(2,5) = 2 mục)
- nDCG@5 = 0,43068 / 1,63093 = **0,2641**
(d) P@5 = 0,2; R@5 = 0,5; RR = 0,25; nDCG@5 = 0,2641.
(e) Một trong hai mục liên quan rơi ngoài top 5 nên recall chỉ 0,5; mục còn lại ở hạng 4 nên bị chiết khấu mạnh, nDCG thấp nhất nhóm.

---------------------------------------------------------------------
## Bảng tổng hợp

| Thành viên | E1: tổng tham số LoRA | E1: tổng / d² | E2: h | E2: loss | P@5 | R@5 | RR | nDCG@5 |
|---|---|---|---|---|---|---|---|---|
| Phúc (24DH190314) | 294 912 | 1/2 = 50 % | 8,4 | 2,248·10⁻⁴ | 0,4 | 1,0 | 0,5 | 0,6934 |
| Duy (24DH190309) | 294 912 | 1/2 = 50 % | 1,31 | 0,2389 | 0,4 | 1,0 | 1,0 | 0,8503 |
| Huy (24DH190312) | 147 456 | 1/4 = 25 % | 1,8 | 0,1530 | 0,2 | 0,5 | 0,25 | 0,2641 |

(1 ma trận LoRA / d²: Phúc, Duy = 1/48 ≈ 2,0833 %; Huy = 1/96 ≈ 1,0417 %.)

---------------------------------------------------------------------
## Vấn đáp thử

**Phúc**
1. *Vì sao mỗi ma trận LoRA chỉ có 2rd tham số?* — ΔW = B·A với A ∈ R^(r×d), B ∈ R^(d×r); W gốc đóng băng; alpha/r chỉ là hệ số nhân (=2).
2. *h = 8,4 > 0 nghĩa là gì, vì sao loss ≈ 0?* — policy đã tăng chosen và giảm rejected so với ref; σ(h) ≈ 1 nên −ln σ(h) ≈ 2,2·10⁻⁴, gradient ≈ 0.
3. *[Đổi tham số] Nếu β = 0,1 (giữ nguyên các logp)?* — h = 0,1×16,8 = **1,68**; loss = ln(1+e^(−1,68)) = **0,1709**; σ(−h) = 0,157 → gradient lớn hơn nhiều.

**Duy**
1. *[Đổi tham số] Nếu r gấp đôi (r = 16)?* — 1 ma trận = 24 576; 1 lớp = 49 152; tổng = **589 824** = đúng **100 %** d² (gấp đôi, tuyến tính theo r).
2. *Nếu đổi chỗ chosen/rejected thì h và ý nghĩa?* — h = **−1,31**, loss = ln(1+e^(1,31)) = **1,5489**; h < 0: policy đang ưa câu bị từ chối hơn so với ref → loss lớn, gradient lớn (σ(−h)=0,79).
3. *Vì sao P@5 = 0,4 mà RR = 1 và nDCG chỉ 0,8503?* — mục đúng đầu ở hạng 1, nhưng mục thứ hai ở hạng 5 thay vì 2: DCG = 1,38685 < IDCG = 1,63093.

**Huy**
1. *Vì sao dùng L=12, d=768 cho PhoBERT-base-v2 và cách kiểm lại?* — cỡ base kiểu RoBERTa; kiểm bằng `AutoConfig.from_pretrained("vinai/phobert-base-v2")` đọc num_hidden_layers, hidden_size. Embedding/classifier không tính vào số tham số LoRA.
2. *Vì sao R@5 = 0,5 và DCG chỉ có một số hạng?* — mục liên quan thứ hai ở hạng 8 (ngoài top 5) nên không tính vào hit@5 và DCG@5; mẫu số recall vẫn là 2.
3. *[Đổi tham số] Nếu mục liên quan đầu tiên chuyển từ hạng 4 lên hạng 1 ([1,0,0,0,0,0,0,1])?* — P@5 = 0,2; R@5 = 0,5 (không đổi); RR = **1**; DCG@5 = 1; nDCG@5 = 1/1,63093 = **0,6131**.

---------------------------------------------------------------------
## Ghi chú / điểm cần lưu ý
- Không chạy được `AutoConfig` và `peft` trong môi trường kiểm (chưa cài transformers/peft, hết dung lượng). L = 12, d = 768 lấy từ bảng `CLF_BACKBONES` của kit + giả định cỡ base. Cell `lab9_E_check_cell.py` sẽ tự dùng AutoConfig/peft nếu máy bạn có; không thì in cảnh báo rồi dùng 12/768.
- Phần peft: nếu đếm toàn bộ tham số trainable của `get_peft_model` thì sẽ lớn hơn (có thêm đầu classifier); chỉ tính tên có `"lora_"`.
