# %% Lab 9 - Phần E (E1-E3): cell kiểm tra số tay
# Cách dùng: đặt MSSV của MÌNH, đứng ở thư mục chứa student_pack_<MSSV>/ rồi chạy cell.
# (Nếu chưa có thư mục: python lab9_tools.py pack --mssv <MSSV>)
import json, math, pathlib

MSSV = "24DH190314"          # <-- đổi: 24DH190314 | 24DH190309 | 24DH190312

# ---- số TAY của từng người (E1: tổng tham số; E2: h, loss; E3: P@5, R@5, RR, nDCG@5)
HAND = {
    "24DH190314": dict(lora=294912, h=8.4,  loss=2.2484e-4, P=0.4, R=1.0, RR=0.5,  ndcg=0.6934),
    "24DH190309": dict(lora=294912, h=1.31, loss=0.2389,    P=0.4, R=1.0, RR=1.0,  ndcg=0.8503),
    "24DH190312": dict(lora=147456, h=1.8,  loss=0.1530,    P=0.2, R=0.5, RR=0.25, ndcg=0.2641),
}

# ---- nạp thẻ (không hard-code số liệu)
card = json.loads(pathlib.Path(f"student_pack_{MSSV}/personal_card.json").read_text(encoding="utf-8"))
assert card["mssv"] == MSSV
print("check_code:", card["check_code"], "| backbone:", card["clf_backbone"])

# ---- 3 hàm theo công thức
def lora_trainable(layers, hidden, r, n_targets=2):
    """Mỗi ma trận đích: A (r x d) + B (d x r) = 2*r*d; tổng = layers * n_targets * 2*r*d."""
    return layers * n_targets * 2 * r * hidden

def dpo_loss(beta, lp_pi_w, lp_pi_l, lp_ref_w, lp_ref_l):
    """h = beta*[(logpi_w - logref_w) - (logpi_l - logref_l)];  loss = -ln sigma(h) = ln(1+e^-h)."""
    h = beta * ((lp_pi_w - lp_ref_w) - (lp_pi_l - lp_ref_l))
    return h, math.log1p(math.exp(-h))

def toy_metrics(rels, k=5):
    n_rel = sum(rels)
    hits = sum(rels[:k])
    rr = next((1 / (i + 1) for i, x in enumerate(rels) if x), 0.0)
    dcg = sum(x / math.log2(i + 2) for i, x in enumerate(rels[:k]))              # mục hạng i chia log2(i+1)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, k)))
    return dict(P=hits / k, R=hits / n_rel, RR=rr, DCG=dcg, IDCG=idcg, ndcg=dcg / idcg)

# ---- layers/hidden: lấy từ config của đúng backbone (cần mạng); không được thì dùng giả định base = 12/768
LAYERS, HIDDEN = 12, 768
cfg = None
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(card["clf_backbone"])
    LAYERS, HIDDEN = cfg.num_hidden_layers, cfg.hidden_size
    print("AutoConfig:", LAYERS, "lớp, hidden", HIDDEN)
except Exception as e:
    print("!! Không đọc được AutoConfig (", type(e).__name__, ") -> dùng giả định 12 lớp, hidden 768")

# ---- E1
r, nt = card["lora"]["r"], len(card["lora"]["target_modules"])
total = lora_trainable(LAYERS, HIDDEN, r, nt)
print(f"E1: tổng LoRA = {total:,} | 1 ma trận = {2*r*HIDDEN:,} | d^2 = {HIDDEN**2:,} "
      f"| 1 ma trận/d^2 = {2*r*HIDDEN/HIDDEN**2:.4%} | tổng/d^2 = {total/HIDDEN**2:.4%}")
assert total == HAND[MSSV]["lora"]

# ---- E2
d = card["dpo"]
h, loss = dpo_loss(d["beta"], d["logp_pi_chosen"], d["logp_pi_rejected"], d["logp_ref_chosen"], d["logp_ref_rejected"])
print(f"E2: h = {h:.4f} | loss = {loss:.6g} | sigma(-h) = {1/(1+math.exp(h)):.6g}")
assert math.isclose(h, HAND[MSSV]["h"], abs_tol=1e-9)
assert math.isclose(loss, HAND[MSSV]["loss"], rel_tol=1e-3)
try:
    from scipy.special import log_expit
    assert math.isclose(loss, -log_expit(h), rel_tol=1e-12)
    print("   scipy.log_expit khớp")
except ImportError:
    print("   (bỏ qua scipy: chưa cài)")

# ---- E3
m = toy_metrics(card["toy_ranking"], 5)
print("E3:", {k: round(v, 5) for k, v in m.items()})
H = HAND[MSSV]
assert (m["P"], m["R"], m["RR"]) == (H["P"], H["R"], H["RR"])
assert abs(m["ndcg"] - H["ndcg"]) < 5e-5
try:
    from sklearn.metrics import ndcg_score
    n = len(card["toy_ranking"])
    sk = ndcg_score([card["toy_ranking"]], [[n - i for i in range(n)]], k=5)   # điểm giảm dần = đúng thứ tự đã xếp
    assert math.isclose(sk, m["ndcg"], rel_tol=1e-12)
    print("   sklearn.ndcg_score khớp")
except ImportError:
    print("   (bỏ qua sklearn: chưa cài)")

# ---- E1 đối chiếu peft: chỉ đếm tham số có "lora_" (peft còn mở thêm đầu classifier)
try:
    from transformers import AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType
    if cfg is None:                                   # không có mạng: dựng config roberta nhỏ (vocab nhỏ cho nhẹ)
        from transformers import AutoConfig
        cfg = AutoConfig.for_model("roberta", num_hidden_layers=LAYERS, hidden_size=HIDDEN,
                                   num_attention_heads=12, intermediate_size=3072, vocab_size=1000)
    cfg.num_labels = 8
    base = AutoModelForSequenceClassification.from_config(cfg)          # khởi tạo ngẫu nhiên, không tải trọng số
    pm = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=card["lora"]["alpha"],
                                         lora_dropout=card["lora"]["dropout"],
                                         target_modules=card["lora"]["target_modules"]))
    n_lora = sum(p.numel() for n_, p in pm.named_parameters() if "lora_" in n_ and p.requires_grad)
    print("   peft đếm 'lora_':", f"{n_lora:,}")
    assert n_lora == total
except ImportError:
    print("   (bỏ qua peft/transformers: chưa cài)")
print("OK - tất cả assert đã qua cho", MSSV)
