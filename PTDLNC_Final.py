# %% [markdown]
# # Tiểu luận cuối kỳ – Phân tích dữ liệu nâng cao (261MIE401601)
# ## Phân bổ ngân sách coupon cá nhân hóa cho chương trình khách hàng thân thiết: từ dự báo và nhân quả đến tối ưu hóa và định tuyến hành động
#
# **Dữ liệu:** dunnhumby – *The Complete Journey* (2.500 hộ gia đình, 711 ngày, 30 chiến dịch coupon).
#
# **Cách chạy:** một file duy nhất, chạy tuần tự từ trên xuống dưới (Colab hoặc PyCharm).
#
# | Biến `RUN_STAGE` | Ý nghĩa |
# |---|---|
# | `"dev"` | Giai đoạn phát triển: chỉ dùng Train + Validation. Locked Test và Out-of-Time **chưa bị đụng tới**. |
# | `"final"` | Đóng băng mô hình, ghi biên bản khóa (thời gian + SHA-256), rồi mới đánh giá một lần trên Locked Test và Out-of-Time. |
#
# **Chu trình 8 giai đoạn của bài:** (1) Thiết kế bài toán và dữ liệu → (2) Dự báo và hiệu chỉnh → (3) Giải thích mô hình → (4) Nhân quả → (5) Predict-Then-Optimize / SPO+ → (6) MILP → (7) Action Router + Guardrails + HITL → (8) Triển khai, đo lường, giám sát.

# %% [markdown]
# ## 0. Thiết lập môi trường và cấu hình

# %%
import os, sys, json, time, hashlib, platform, warnings, math, itertools, datetime as dt
from pathlib import Path
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # tránh cảnh báo đếm CPU của joblib trên Windows
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
SHOW_FIGS = "ipykernel" in sys.modules          # Colab/Jupyter: hiển thị hình; PyCharm/terminal: chỉ lưu file (không chặn chương trình)
if not SHOW_FIGS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats, optimize, sparse
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error
import joblib

try:                                   # tránh lỗi tiếng Việt trên console Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------- CẤU HÌNH CHÍNH -----------------------------
RUN_STAGE = os.environ.get("PTDLNC_RUN_STAGE", "dev")   # "dev" = chưa động tới Locked/OOT | "final" = khóa rồi đánh giá một lần
SEED = 2026
rng = np.random.default_rng(SEED)

# Tham số bài toán (khóa trong Decision Contract)
LOOKBACK = 90              # cửa sổ nhìn lại để dựng đặc trưng (ngày)
WAVE_LEN = 28              # độ dài một đợt quyết định = độ dài kỳ kết quả (ngày)
FIRST_DECISION_DAY = 197   # ngày quyết định của đợt đầu tiên (chiến dịch đầu tiên bắt đầu ngày 224)
N_WAVES = 18               # 18 đợt: ngày 197, 225, ..., 673
MARGIN = 0.25              # tỷ suất lợi nhuận gộp GIẢ ĐỊNH (có phân tích độ nhạy 0,15–0,35)
CONTACT_COST = 0.10        # chi phí liên hệ mỗi lần gửi coupon (USD) – GIẢ ĐỊNH
PS_CLIP = (0.03, 0.97)     # miền positivity cho propensity score
WINSOR_Q = 0.995           # winsorize chi tiêu ở phân vị này (tính trên Train)

# Chia thời gian theo ĐỢT, có 1 đợt đệm (embargo) giữa các tập vì nhãn của một đợt chỉ biết sau 28 ngày
SPLIT_WAVES = {
    "train":  list(range(0, 8)),
    "val":    [9, 10],
    "locked": [12, 13],
    "oot":    [15, 16, 17],
}
EMBARGO_WAVES = [8, 11, 14]

# Cổng phát hành (Release Gates) – KHAI BÁO TRƯỚC khi khóa Locked Test (không chỉnh sau khi thấy kết quả)
#   key = mã cổng ; (mô tả, toán tử, ngưỡng)
RELEASE_GATES = {
    "G1_brier_rel":  ("Brier mô hình không tệ hơn baseline logistic quá 2% (tương đối)", "<=", 0.02),
    "G2_ece":        ("ECE sau hiệu chỉnh ≤ 0,05", "<=", 0.05),
    "G3_coverage":   ("Độ phủ conformal 90% thực tế ≥ 0,85", ">=", 0.85),
    "G4_psi":        ("PSI lớn nhất THEO TỪNG ĐỢT của đặc trưng/điểm dự báo so với Train < 0,25", "<", 0.25),
    "G5_p95_ms":     ("Độ trễ online (đặc trưng + suy luận + luật) P95 ≤ 250 ms", "<=", 250.0),
    "G6_noninf":     ("Chính sách Router không kém chính sách lịch sử quá 3 USD/hộ-đợt (cận dưới CI95 của chênh lệch giá trị ≥ −3)", ">=", -3.0),
    "G7_cost_cut":   ("Router giảm chi phí coupon kỳ vọng ≥ 50% so với chính sách lịch sử", ">=", 0.50),
    "G8_hitl":       ("Nhu cầu HITL (ứng viên TRƯỚC khi áp trần công suất) ≤ 8% số hộ", "<=", 0.08),
}

# ----------------------------- ĐƯỜNG DẪN -----------------------------
IN_COLAB = "google.colab" in sys.modules
def _find_data_dir():
    cands = []
    try:
        here = Path(__file__).resolve().parent
        cands += [here / "Dataset", here]
    except NameError:
        pass
    cands += [Path.cwd() / "Dataset", Path.cwd(),
              Path("/content/Dataset"), Path("/content/drive/MyDrive/PTDLNC_Final/Dataset")]
    for c in cands:
        if (c / "transaction_data.csv").exists():
            return c
    raise FileNotFoundError(
        "Không thấy transaction_data.csv. Trên Colab: upload thư mục Dataset vào /content hoặc mount Drive "
        "(from google.colab import drive; drive.mount('/content/drive')) rồi đặt DATA_DIR thủ công.")
DATA_DIR = _find_data_dir()
OUT_DIR = Path(os.environ["PTDLNC_OUT_DIR"]) if os.environ.get("PTDLNC_OUT_DIR") else DATA_DIR.parent / "outputs"
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 100, "display.width", 180)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.titlesize": 11, "axes.labelsize": 10})

# ----------------------------- HÀM TIỆN ÍCH -----------------------------
def show_df(obj, title=None, rows=None):
    """Hiển thị bảng đẹp trên Colab/Jupyter, vẫn chạy được trên PyCharm/console."""
    if title:
        print(f"\n{title}")
    view = obj.head(rows) if rows is not None else obj
    try:
        from IPython.display import display
        display(view)
    except ImportError:
        print(view.to_string() if isinstance(view, pd.Series) else view.to_string(index=True))

def save_table(df, name):
    df.to_csv(OUT_DIR / "tables" / f"{name}.csv", encoding="utf-8-sig")

def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / f"{name}.png", bbox_inches="tight")
    if SHOW_FIGS:
        plt.show()
    plt.close("all")

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _round_obj(o, nd=6):
    """Làm tròn số thực (đệ quy) để mã băm không lệch vì sai khác dấu phẩy động giữa các máy."""
    if isinstance(o, float): return round(o, nd)
    if isinstance(o, dict): return {k: _round_obj(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_round_obj(v, nd) for v in o]
    return o

def sha256_obj(obj):
    return hashlib.sha256(json.dumps(_round_obj(obj), sort_keys=True, default=str).encode("utf-8")).hexdigest()

def expected_calibration_error(y, p, bins=10):
    """ECE theo bin phân vị đều (equal-frequency) để tránh bin rỗng."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)

def psi(expected, actual, bins=10):
    """Population Stability Index; bin theo phân vị của mẫu tham chiếu."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))

def tvd(p, q):
    """Total Variation Distance giữa hai phân phối rời rạc (ví dụ phân phối hành động)."""
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())

_seed_everything = lambda s=SEED: (np.random.seed(s))
_seed_everything()

print(f"Python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} | "
      f"sklearn {sklearn.__version__} | scipy {scipy.__version__}")
print(f"RUN_STAGE = {RUN_STAGE!r} | SEED = {SEED}")
print(f"DATA_DIR  = {DATA_DIR}")
print(f"OUT_DIR   = {OUT_DIR}")

# %% [markdown]
# # GIAI ĐOẠN 1. Thiết kế bài toán quyết định, kỹ nghệ dữ liệu và phòng chống rò rỉ
#
# ### 1.1 Decision Contract (khóa trước khi mô hình hóa)
#
# | Thành phần | Nội dung |
# |---|---|
# | Chủ thể quyết định | Quản lý chương trình khách hàng thân thiết của một chuỗi siêu thị (kênh omnichannel: thẻ thành viên + coupon gửi qua mail/ứng dụng) |
# | Đơn vị phân tích | **Hộ gia đình × đợt 28 ngày** (chỉ xét hộ có ít nhất một lần mua trong 90 ngày trước thời điểm quyết định) |
# | Thời điểm quyết định | Ngày đầu tiên của mỗi đợt (197, 225, …, 673). Đặc trưng chỉ lấy từ dữ liệu **trước** ngày đó |
# | Kỳ kết quả | 28 ngày kể từ ngày quyết định |
# | Biến can thiệp (T) | Hộ nhận ≥ 1 coupon từ chiến dịch bắt đầu trong đợt |
# | Biến kết quả (Y) | Chi tiêu 28 ngày sau quyết định (USD); phụ: có quay lại mua hay không |
# | Hành động | Gửi coupon / không gửi (mở rộng thành 5 hành động ở Action Router) |
# | Hàm giá trị | Giá trị = MARGIN × (chi tiêu tăng thêm do coupon) − (chiết khấu coupon kỳ vọng + chi phí liên hệ) |
# | Ràng buộc | Ngân sách mỗi đợt, giới hạn tần suất liên hệ, trần theo phân khúc, không gửi khi vi phạm positivity |
#
# **Lưu ý trung thực:** coupon trong dữ liệu *không* được gán ngẫu nhiên. Hộ chi tiêu nhiều được nhắm nhiều hơn rất rõ. Vì vậy toàn bộ phần nhân quả dựa trên giả định *không có nhiễu chưa quan sát* (conditional ignorability) và được kiểm tra bằng overlap, placebo và độ nhạy.

# %%
# ---------------------------------------------------------------------------
# 1.2 Đọc dữ liệu và hợp đồng dữ liệu đầu vào
# ---------------------------------------------------------------------------
t0 = time.time()
tx = pd.read_csv(DATA_DIR / "transaction_data.csv",
                 usecols=["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "SALES_VALUE",
                          "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC", "WEEK_NO"])
camp_desc = pd.read_csv(DATA_DIR / "campaign_desc.csv")
camp_tab = pd.read_csv(DATA_DIR / "campaign_table.csv")
redempt = pd.read_csv(DATA_DIR / "coupon_redempt.csv")
demo = pd.read_csv(DATA_DIR / "hh_demographic.csv")
product = pd.read_csv(DATA_DIR / "product.csv", usecols=["PRODUCT_ID", "DEPARTMENT", "BRAND"])

INPUT_FILES = ["transaction_data.csv", "campaign_desc.csv", "campaign_table.csv", "coupon_redempt.csv",
               "hh_demographic.csv", "product.csv"]
data_contract_in = []
for fn in INPUT_FILES:
    data_contract_in.append({
        "file": fn,
        "bytes": (DATA_DIR / fn).stat().st_size,
        "sha256": sha256_file(DATA_DIR / fn),
    })
data_contract_in = pd.DataFrame(data_contract_in)
data_contract_in["rows"] = [len(tx), len(camp_desc), len(camp_tab), len(redempt), len(demo), len(product)]
show_df(data_contract_in, "Hợp đồng dữ liệu đầu vào (kèm SHA-256)")
save_table(data_contract_in, "data_contract_input")

# Kiểm tra chất lượng dữ liệu cơ bản
dq = pd.Series({
    "Số hộ trong giao dịch": tx.household_key.nunique(),
    "Khoảng ngày": f"{tx.DAY.min()} – {tx.DAY.max()}",
    "Số chiến dịch": camp_desc.CAMPAIGN.nunique(),
    "Cặp hộ–chiến dịch bị trùng": int(camp_tab.duplicated(["household_key", "CAMPAIGN"]).sum()),
    "Dòng có SALES_VALUE < 0": int((tx.SALES_VALUE < 0).sum()),
    "Dòng có SALES_VALUE = 0": int((tx.SALES_VALUE == 0).sum()),
    "Hộ có nhân khẩu học": demo.household_key.nunique(),
    "Tổng doanh thu (USD)": round(tx.SALES_VALUE.sum(), 0),
})
show_df(dq.to_frame("giá trị"), "Kiểm tra chất lượng dữ liệu")

# %%
# ---------------------------------------------------------------------------
# 1.3 Ma trận hoạt động theo ngày (hộ × ngày) để tính đặc trưng nhanh và đúng thời điểm
# ---------------------------------------------------------------------------
MAXD = 760                                   # dư chỗ cho ngày kết thúc chiến dịch (≤ 719)
hh_ids = np.sort(tx.household_key.unique())
H = len(hh_ids)
hh_pos = {h: i for i, h in enumerate(hh_ids)}

FRESH = {"PRODUCE", "MEAT", "MEAT-PCKGD", "SEAFOOD", "SEAFOOD-PCKGD", "DELI", "PASTRY", "SALAD BAR"}
prod_priv = product.set_index("PRODUCT_ID").BRAND.eq("Private")
prod_fresh = product.set_index("PRODUCT_ID").DEPARTMENT.isin(FRESH)

tx["hi"] = tx.household_key.map(hh_pos).astype(np.int32)
tx["priv"] = tx.PRODUCT_ID.map(prod_priv).fillna(False).astype(bool)
tx["fresh"] = tx.PRODUCT_ID.map(prod_fresh).fillna(False).astype(bool)
tx["coupon_used"] = -(tx.COUPON_DISC + tx.COUPON_MATCH_DISC)      # chiết khấu coupon (dương)
tx["promo_disc"] = -tx.RETAIL_DISC                                 # chiết khấu khuyến mãi của cửa hàng (dương)
tx["priv_sales"] = tx.SALES_VALUE * tx.priv
tx["fresh_sales"] = tx.SALES_VALUE * tx.fresh

def build_daily_matrices(tx_in, camp_tab_in, redempt_in):
    """Dựng các ma trận (hộ × ngày). Dùng được cả khi cắt dữ liệu tại một ngày (kiểm tra rò rỉ)."""
    g = tx_in.groupby(["hi", "DAY"])
    agg = g.agg(spend=("SALES_VALUE", "sum"), coupon=("coupon_used", "sum"), promo=("promo_disc", "sum"),
                priv=("priv_sales", "sum"), fresh=("fresh_sales", "sum"), baskets=("BASKET_ID", "nunique")).reset_index()
    M = {}
    for c in ["spend", "coupon", "promo", "priv", "fresh", "baskets"]:
        a = np.zeros((H, MAXD), dtype=np.float64)
        a[agg.hi.values, agg.DAY.values] = agg[c].values
        M[c] = a
    # chiến dịch: ngày bắt đầu mỗi cặp hộ–chiến dịch, và dải ngày chiến dịch còn hiệu lực
    ct = camp_tab_in.merge(camp_desc[["CAMPAIGN", "START_DAY", "END_DAY"]], on="CAMPAIGN")
    ct["hi"] = ct.household_key.map(hh_pos).astype(np.int32)
    a = np.zeros((H, MAXD)); np.add.at(a, (ct.hi.values, ct.START_DAY.values), 1.0); M["targ"] = a
    diff = np.zeros((H, MAXD + 1))
    np.add.at(diff, (ct.hi.values, ct.START_DAY.values + 1), 1.0)       # còn hiệu lực từ ngày start+1 ...
    np.add.at(diff, (ct.hi.values, ct.END_DAY.values + 1), -1.0)        # ... đến hết end
    M["active_prev"] = np.cumsum(diff, axis=1)[:, :MAXD]
    rd = redempt_in.copy(); rd["hi"] = rd.household_key.map(hh_pos).astype(np.int32)
    a = np.zeros((H, MAXD)); np.add.at(a, (rd.hi.values, rd.DAY.values), 1.0); M["redeem"] = a
    # cộng dồn
    C = {k: np.cumsum(v, axis=1) for k, v in M.items() if k != "active_prev"}
    C["active_prev"] = M["active_prev"]
    # ngày mua gần nhất và ngày nhắm gần nhất (chạy max theo thời gian)
    day_idx = np.arange(MAXD)[None, :]
    C["last_visit"] = np.maximum.accumulate(np.where(M["baskets"] > 0, day_idx, 0), axis=1)
    C["last_targ"] = np.maximum.accumulate(np.where(M["targ"] > 0, day_idx, 0), axis=1)
    # Ngày cửa hàng đóng cửa (lễ): tổng giỏ hàng trong ngày < 10% trung vị 15 ngày TRƯỚC đó (chỉ dùng quá khứ → không rò rỉ).
    # Recency tính theo số ngày MỞ CỬA để kỳ nghỉ lễ không làm lệch phân phối của mọi hộ cùng lúc.
    tot = M["baskets"].sum(axis=0)
    med = pd.Series(tot).shift(1).rolling(15, min_periods=15).median().fillna(0).values
    closed = (day_idx[0] >= 1) & (day_idx[0] <= int(tx_in.DAY.max())) & (med > 50) & (tot < 0.10 * med)
    open_day = ((day_idx[0] >= 1) & ~closed).astype(float)
    C["cum_open"] = np.cumsum(open_day)
    C["closed_days"] = np.where(closed)[0]
    first_visit = np.where((M["baskets"] > 0).any(axis=1), (M["baskets"] > 0).argmax(axis=1), 10**6)
    return M, C, first_visit

MATS, CUM, FIRST_VISIT = build_daily_matrices(tx, camp_tab, redempt)
print(f"Dựng ma trận {H} hộ × {MAXD} ngày trong {time.time() - t0:.1f}s")
print("Ngày cửa hàng đóng cửa phát hiện được (tổng giỏ hàng < 10% trung vị 15 ngày trước):", CUM["closed_days"].tolist(),
      "→ khoảng cách 365 ngày giữa hai ngày này khớp lịch lễ hằng năm; recency được tính theo ngày mở cửa.")

# %%
# ---------------------------------------------------------------------------
# 1.4 Hàm dựng đặc trưng TẠI THỜI ĐIỂM QUYẾT ĐỊNH d (chỉ dùng ngày < d)
# ---------------------------------------------------------------------------
# Nhân khẩu học (chỉ ~32% hộ có) → mã hóa thứ tự, -1 = không rõ
age_order = ["19-24", "25-34", "35-44", "45-54", "55-64", "65+"]
inc_order = ["Under 15K", "15-24K", "25-34K", "35-49K", "50-74K", "75-99K", "100-124K", "125-149K",
             "150-174K", "175-199K", "200-249K", "250K+"]
dm = demo.set_index("household_key")
DEMO = pd.DataFrame(index=hh_ids)
DEMO["has_demo"] = DEMO.index.isin(dm.index).astype(int)
DEMO["age_ord"] = pd.Series(dm.AGE_DESC.map({v: i for i, v in enumerate(age_order)})).reindex(hh_ids).fillna(-1).values
DEMO["income_ord"] = pd.Series(dm.INCOME_DESC.map({v: i for i, v in enumerate(inc_order)})).reindex(hh_ids).fillna(-1).values
DEMO["hh_size"] = pd.Series(dm.HOUSEHOLD_SIZE_DESC.map({"1": 1, "2": 2, "3": 3, "4": 4, "5+": 5})).reindex(hh_ids).fillna(-1).values
DEMO["kids"] = pd.Series(dm.KID_CATEGORY_DESC.map({"None/Unknown": 0, "1": 1, "2": 2, "3+": 3})).reindex(hh_ids).fillna(-1).values
DEMO["owner"] = pd.Series(dm.HOMEOWNER_DESC.map({"Homeowner": 1, "Probable Owner": 1, "Renter": 0,
                                                    "Probable Renter": 0, "Unknown": -1})).reindex(hh_ids).fillna(-1).values
DEMO_ARR = DEMO.values.astype(float)
# Lịch chiến dịch đã được lập kế hoạch trước → số chiến dịch bắt đầu trong đợt là thông tin biết tại thời điểm quyết định
_cnt = np.zeros(MAXD + 1); np.add.at(_cnt, camp_desc.START_DAY.values, 1.0)
CAMP_CUM = np.cumsum(_cnt)
DEMO_COLS = list(DEMO.columns)

def wsum(key, hi, a, b):
    """Tổng theo cửa sổ [a, b] (bao gồm hai đầu) cho từng đơn vị; a, b là mảng ngày."""
    C = CUM[key]
    a = np.clip(a, 1, MAXD - 1); b = np.clip(b, 0, MAXD - 1)
    val = C[hi, b] - C[hi, a - 1]
    return np.where(b >= a, val, 0.0)

def make_features(hi, d, cum=None, first_visit=None):
    """Đặc trưng tại thời điểm quyết định d. hi: chỉ số hộ; d: mảng ngày quyết định.
    Mọi cửa sổ kết thúc ở d-1 → không rò rỉ thông tin tương lai."""
    global CUM, FIRST_VISIT
    if cum is not None:
        _old = (CUM, FIRST_VISIT); CUM, FIRST_VISIT = cum, first_visit
    hi = np.asarray(hi); d = np.asarray(d)
    f = {}
    s28, s90 = wsum("spend", hi, d - 28, d - 1), wsum("spend", hi, d - LOOKBACK, d - 1)
    s_prev28 = wsum("spend", hi, d - 56, d - 29)
    s180 = wsum("spend", hi, d - 180, d - 1)
    v28, v90 = wsum("baskets", hi, d - 28, d - 1), wsum("baskets", hi, d - LOOKBACK, d - 1)
    f["spend_28"], f["spend_prev28"], f["spend_90"], f["spend_180"] = s28, s_prev28, s90, s180
    f["spend_trend"] = s28 - s_prev28
    f["visits_28"], f["visits_90"] = v28, v90
    f["avg_basket_90"] = np.where(v90 > 0, s90 / np.maximum(v90, 1), 0.0)
    lv = CUM["last_visit"][hi, np.clip(d - 1, 0, MAXD - 1)]
    f["recency"] = CUM["cum_open"][np.clip(d - 1, 0, MAXD - 1)] - CUM["cum_open"][lv]      # số ngày MỞ CỬA kể từ lần mua gần nhất
    cp90 = wsum("coupon", hi, d - LOOKBACK, d - 1)
    pr90 = wsum("promo", hi, d - LOOKBACK, d - 1)
    f["coupon_used_90"] = cp90
    f["coupon_share_90"] = np.where(s90 > 0, cp90 / np.maximum(s90, 1e-9), 0.0)
    f["promo_share_90"] = np.where(s90 > 0, pr90 / np.maximum(s90 + pr90, 1e-9), 0.0)
    f["private_share_90"] = np.where(s90 > 0, wsum("priv", hi, d - LOOKBACK, d - 1) / np.maximum(s90, 1e-9), 0.0)
    f["fresh_share_90"] = np.where(s90 > 0, wsum("fresh", hi, d - LOOKBACK, d - 1) / np.maximum(s90, 1e-9), 0.0)
    f["redeem_90"] = wsum("redeem", hi, d - LOOKBACK, d - 1)
    # lịch sử bị nhắm theo CỬA SỔ (không dùng biến tích lũy từ ngày đầu vì chúng trôi theo lịch → phá vỡ tính dừng)
    f["targ_prev28"] = wsum("targ", hi, d - 28, d - 1)
    f["targ_prev90"] = wsum("targ", hi, d - LOOKBACK, d - 1)
    lt = CUM["last_targ"][hi, np.clip(d - 1, 0, MAXD - 1)]
    f["days_since_targ"] = np.where(lt > 0, np.minimum((d - 1) - lt, 180), 180)   # 180 = chưa bị nhắm trong 180 ngày qua
    f["active_camp_carry"] = CUM["active_prev"][hi, np.clip(d, 0, MAXD - 1)]
    # mùa vụ (lịch quyết định đã biết trước)
    f["camp_planned"] = CAMP_CUM[np.clip(d + WAVE_LEN - 1, 0, MAXD - 1)] - CAMP_CUM[np.clip(d - 1, 0, MAXD - 1)]
    f["season_sin"], f["season_cos"] = np.sin(2 * np.pi * d / 364.0), np.cos(2 * np.pi * d / 364.0)
    out = pd.DataFrame(f)
    dem = pd.DataFrame(DEMO_ARR[hi], columns=DEMO_COLS)
    out = pd.concat([out, dem], axis=1)
    if cum is not None:
        CUM, FIRST_VISIT = _old
    return out

FEATURES = list(make_features(np.array([0]), np.array([300])).columns)
print(f"{len(FEATURES)} đặc trưng:", FEATURES)

# %%
# ---------------------------------------------------------------------------
# 1.5 Dựng bảng hộ × đợt, biến T và Y; chia Train / Validation / Locked / OOT có đợt đệm
# ---------------------------------------------------------------------------
wave_days = FIRST_DECISION_DAY + WAVE_LEN * np.arange(N_WAVES)
assert wave_days[-1] + WAVE_LEN - 1 <= tx.DAY.max(), "Đợt cuối vượt quá dữ liệu"

rows = []
for k, d0 in enumerate(wave_days):
    hi = np.arange(H); d = np.full(H, d0)
    X = make_features(hi, d)
    X.insert(0, "household_key", hh_ids)
    X.insert(1, "wave", k)
    X.insert(2, "decision_day", d0)
    # kết quả và can thiệp (NHÌN VỀ TƯƠNG LAI – chỉ dùng làm nhãn, không bao giờ làm đặc trưng)
    X["T"] = (wsum("targ", hi, d, d + WAVE_LEN - 1) > 0).astype(int)
    X["n_camp_wave"] = wsum("targ", hi, d, d + WAVE_LEN - 1)
    X["Y"] = wsum("spend", hi, d, d + WAVE_LEN - 1)
    X["Y_active"] = (wsum("baskets", hi, d, d + WAVE_LEN - 1) > 0).astype(int)
    X["Y_coupon_cost"] = wsum("coupon", hi, d, d + WAVE_LEN - 1)
    X["Y_redeem"] = wsum("redeem", hi, d, d + WAVE_LEN - 1)
    X["Y_pre28"] = wsum("spend", hi, d - 28, d - 1)      # dùng cho kiểm tra placebo
    rows.append(X)
panel_all = pd.concat(rows, ignore_index=True)
panel_all["hi"] = panel_all.household_key.map(hh_pos).astype(int)

n_before = len(panel_all)
panel_all = panel_all[panel_all.spend_90 > 0].reset_index(drop=True)      # chỉ hộ còn hoạt động trong 90 ngày trước
print(f"Bảng hộ × đợt: {n_before:,} → {len(panel_all):,} đơn vị sau khi lọc hộ còn hoạt động (spend_90 > 0)")

wave2split = {k: s for s, ks in SPLIT_WAVES.items() for k in ks}
wave2split.update({k: "embargo" for k in EMBARGO_WAVES})
panel_all["split"] = panel_all.wave.map(wave2split)
assert panel_all.split.notna().all()

# Winsorize chi tiêu bằng ngưỡng tính trên Train (chống đuôi dài, tránh phụ thuộc Locked/OOT)
Y_CAP = float(panel_all.loc[panel_all.split == "train", "Y"].quantile(WINSOR_Q))
for c in ["Y", "Y_pre28"]:
    panel_all[c + "_w"] = panel_all[c].clip(upper=Y_CAP)

# --- NIÊM PHONG: Locked/OOT bị tách khỏi bảng làm việc cho tới khi RUN_STAGE == "final" ---
_sealed_panel = panel_all[panel_all.split.isin(["locked", "oot"])].copy()
panel = panel_all[panel_all.split.isin(["train", "val"])].reset_index(drop=True)
if RUN_STAGE != "final":
    panel_all = panel.copy()          # ở giai đoạn dev, mọi phân tích chỉ nhìn thấy Train + Val
    print("[DEV] Locked Test và OOT đã được niêm phong – không có hàng nào trong bảng làm việc.")

summary = (panel.groupby(["split", "wave"])
           .agg(decision_day=("decision_day", "first"), n=("T", "size"), treated=("T", "sum"),
                treat_rate=("T", "mean"), mean_Y=("Y", "mean"), active=("Y_active", "mean")).round(3))
show_df(summary, "Tóm tắt bảng hộ × đợt (Train + Validation)")
save_table(summary, "panel_summary_dev")

# %%
# ---------------------------------------------------------------------------
# 1.6 Kiểm tra rò rỉ: dựng lại đặc trưng chỉ từ dữ liệu < d và so sánh với bản đầy đủ
# ---------------------------------------------------------------------------
def leakage_test(n_units=400, waves=(1, 6, 10)):
    ok, rep = True, []
    for k in waves:
        d0 = int(wave_days[k])
        sub = panel_all[panel_all.wave == k] if k in panel_all.wave.values else _sealed_panel[_sealed_panel.wave == k]
        sub = sub.sample(min(n_units, len(sub)), random_state=SEED)
        tx_cut = tx[tx.DAY < d0]
        ct_cut = camp_tab.merge(camp_desc[["CAMPAIGN", "START_DAY"]], on="CAMPAIGN")
        ct_cut = ct_cut[ct_cut.START_DAY < d0][["household_key", "CAMPAIGN"]]
        rd_cut = redempt[redempt.DAY < d0]
        M2, C2, fv2 = build_daily_matrices(tx_cut, ct_cut, rd_cut)
        # active_prev dùng lịch chiến dịch đã công bố → giữ nguyên bản đầy đủ (kế hoạch chiến dịch biết trước)
        C2["active_prev"] = CUM["active_prev"]
        f_cut = make_features(sub.hi.values, np.full(len(sub), d0), cum=C2, first_visit=fv2)
        f_full = sub[FEATURES].reset_index(drop=True)
        diff = (f_cut.values - f_full.values)
        maxdiff = float(np.nanmax(np.abs(diff)))
        rep.append({"wave": k, "decision_day": d0, "n_units": len(sub), "max_abs_diff": maxdiff})
        ok &= maxdiff < 1e-6
    return ok, pd.DataFrame(rep)

leak_ok, leak_rep = leakage_test()
show_df(leak_rep, "Kiểm tra rò rỉ thời gian (đặc trưng dựng từ dữ liệu bị cắt tại d−1 phải trùng bản đầy đủ)")
assert leak_ok, "PHÁT HIỆN RÒ RỈ: đặc trưng phụ thuộc dữ liệu từ ngày quyết định trở đi"
print("→ Không phát hiện rò rỉ: đặc trưng chỉ dùng dữ liệu trước thời điểm quyết định.")

# %%
# ---------------------------------------------------------------------------
# 1.7 Khám phá nhanh: lựa chọn mẫu của chương trình (vì sao không so sánh thô được)
# ---------------------------------------------------------------------------
tr = panel[panel.split == "train"].copy()
tr["spend90_decile"] = pd.qcut(tr.spend_90, 10, labels=False, duplicates="drop") + 1
sel = tr.groupby("spend90_decile").agg(treat_rate=("T", "mean"), Y_treated=("Y", lambda s: np.nan), n=("T", "size"))
sel["Y_treated"] = tr[tr["T"] == 1].groupby("spend90_decile").Y.mean()
sel["Y_control"] = tr[tr["T"] == 0].groupby("spend90_decile").Y.mean()
show_df(sel.round(3), "Tỷ lệ được nhắm và chi tiêu theo decile chi tiêu 90 ngày trước (Train)")

naive = tr[tr["T"] == 1].Y.mean() - tr[tr["T"] == 0].Y.mean()
print(f"Chênh lệch thô (treated − control) = {naive:,.1f} USD/hộ/đợt  → bị thổi phồng do chọn mẫu (xem Giai đoạn 4)")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].bar(sel.index, sel.treat_rate, color="#4C78A8"); ax[0].set_xlabel("Decile chi tiêu 90 ngày trước"); ax[0].set_ylabel("Tỷ lệ được nhắm")
ax[0].set_title("Chương trình ưu tiên hộ chi tiêu nhiều")
ax[1].plot(sel.index, sel.Y_treated, "o-", label="Được nhắm"); ax[1].plot(sel.index, sel.Y_control, "s--", label="Không được nhắm")
ax[1].set_xlabel("Decile chi tiêu 90 ngày trước"); ax[1].set_ylabel("Chi tiêu 28 ngày sau (USD)"); ax[1].legend()
ax[1].set_title("Chênh lệch trong cùng decile nhỏ hơn nhiều so với chênh lệch thô")
save_fig("fig_1_selection")

wd = (pd.DataFrame({"wave": np.arange(N_WAVES), "day": wave_days})
      .assign(split=lambda x: x.wave.map(wave2split)))
fig, ax = plt.subplots(figsize=(11, 1.8))
cols = {"train": "#4C78A8", "val": "#F58518", "locked": "#E45756", "oot": "#54A24B", "embargo": "#BBBBBB"}
for r in wd.itertuples():
    ax.barh(0, WAVE_LEN, left=r.day, color=cols[r.split], edgecolor="white")
    ax.text(r.day + WAVE_LEN / 2, 0, str(r.wave), ha="center", va="center", fontsize=8, color="white")
ax.set_yticks([]); ax.set_xlabel("Ngày trong dữ liệu")
ax.set_ylim(-0.6, 0.6)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()],
          labels=["Train", "Validation", "Locked Test", "Out-of-Time", "Đợt đệm (embargo)"], ncol=5, loc="lower center",
          bbox_to_anchor=(0.5, 1.03), frameon=False)
save_fig("fig_1_split_timeline")

# %% [markdown]
# # GIAI ĐOẠN 2. Dự báo rủi ro, hiệu chỉnh xác suất và định lượng bất định
#
# Hai bài toán dự báo hỗ trợ quyết định:
# 1. **Xác suất hộ quay lại mua trong 28 ngày tới** `p_active` (rủi ro rời bỏ = 1 − p_active) → cần *xác suất đã hiệu chỉnh*.
# 2. **Chi tiêu 28 ngày tới** `ŷ` kèm **khoảng dự báo conformal** → cần *bất định định lượng được*.
#
# Nguyên tắc: chọn siêu tham số bằng *holdout theo thời gian bên trong Train*, hiệu chỉnh bằng dự báo *out-of-fold* (chia theo hộ), và chỉ chạm Validation để báo cáo.

# %%
# ---------------------------------------------------------------------------
# 2.1 Chọn siêu tham số bằng holdout theo thời gian bên trong Train
# ---------------------------------------------------------------------------
TR = panel[panel.split == "train"].reset_index(drop=True)
VA = panel[panel.split == "val"].reset_index(drop=True)
X_tr, X_va = TR[FEATURES], VA[FEATURES]
inner_fit, inner_ev = TR.wave <= 5, TR.wave >= 6          # đợt 0–5 huấn luyện, 6–7 kiểm tra nội bộ

HGB_GRID = {
    "nhỏ":   dict(max_iter=150, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0),
    "vừa":   dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=60, l2_regularization=1.0),
    "sâu":   dict(max_iter=350, learning_rate=0.04, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=2.0),
}
def hgb_clf(**kw): return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **kw)
def hgb_reg(**kw): return HistGradientBoostingRegressor(random_state=SEED, early_stopping=False, **kw)

sel_rows = []
for name, kw in HGB_GRID.items():
    c = hgb_clf(**kw).fit(X_tr[inner_fit], TR.Y_active[inner_fit])
    r = hgb_reg(**kw).fit(X_tr[inner_fit], TR.Y_w[inner_fit])
    p, yhat = c.predict_proba(X_tr[inner_ev])[:, 1], r.predict(X_tr[inner_ev])
    sel_rows.append({"cấu hình": name, "Brier": brier_score_loss(TR.Y_active[inner_ev], p),
                     "AUC": roc_auc_score(TR.Y_active[inner_ev], p),
                     "MAE chi tiêu": mean_absolute_error(TR.Y_w[inner_ev], yhat)})
sel_df = pd.DataFrame(sel_rows).set_index("cấu hình").round(4)
show_df(sel_df, "Chọn siêu tham số (holdout theo thời gian trong Train)")
BEST_CFG = sel_df["Brier"].idxmin()
BEST_KW = HGB_GRID[BEST_CFG]
print("→ Cấu hình được chọn:", BEST_CFG, BEST_KW)

# %%
# ---------------------------------------------------------------------------
# 2.2 Mô hình cơ sở, mô hình phi tuyến, bộ hiệu chỉnh (Platt / Isotonic) – đánh giá trên Validation
# ---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_tr, TR.Y_active, groups=TR.household_key))

# (a) dự báo out-of-fold trên Train → dùng để học bộ hiệu chỉnh mà không cần tách thêm dữ liệu
oof_p = np.zeros(len(TR))
for tri, tei in folds:
    m = hgb_clf(**BEST_KW).fit(X_tr.iloc[tri], TR.Y_active.iloc[tri])
    oof_p[tei] = m.predict_proba(X_tr.iloc[tei])[:, 1]

# (b) mô hình cuối cùng huấn luyện trên toàn bộ Train
clf_final = hgb_clf(**BEST_KW).fit(X_tr, TR.Y_active)
p_raw_va = clf_final.predict_proba(X_va)[:, 1]

# (c) bộ hiệu chỉnh
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
platt = LogisticRegression(C=1e6).fit(logit(oof_p).reshape(-1, 1), TR.Y_active)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof_p, TR.Y_active)
calib_platt = lambda p: platt.predict_proba(logit(p).reshape(-1, 1))[:, 1]
calib_iso = lambda p: iso.predict(p)

# (d) baseline logistic
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5)).fit(X_tr, TR.Y_active)
p_lr_va = lr.predict_proba(X_va)[:, 1]

cands = {
    "Hằng số (tỷ lệ Train)": np.full(len(VA), TR.Y_active.mean()),
    "Logistic (baseline)": p_lr_va,
    "Quy tắc recency (logistic 1 biến)": LogisticRegression().fit(TR[["recency"]], TR.Y_active).predict_proba(VA[["recency"]])[:, 1],
    "HGB thô": p_raw_va,
    "HGB + Platt": calib_platt(p_raw_va),
    "HGB + Isotonic": calib_iso(p_raw_va),
}
rows = []
for k, p in cands.items():
    rows.append({"mô hình": k,
                 "AUC": roc_auc_score(VA.Y_active, p) if np.std(p) > 0 else 0.5,
                 "PR-AUC (quay lại)": average_precision_score(VA.Y_active, p),
                 "PR-AUC (rời bỏ)": average_precision_score(1 - VA.Y_active, 1 - p),
                 "Brier": brier_score_loss(VA.Y_active, p),
                 "ECE": expected_calibration_error(VA.Y_active, p)})
cmp_df = pd.DataFrame(rows).set_index("mô hình").round(4)
show_df(cmp_df, "Chất lượng xác suất trên Validation")
save_table(cmp_df, "forecast_probability_validation")

# chọn bộ hiệu chỉnh theo Brier/ECE trên OOF (không dùng Validation để chọn)
cal_oof = {"HGB thô": oof_p, "HGB + Platt": calib_platt(oof_p), "HGB + Isotonic": calib_iso(oof_p)}
cal_oof_ece = {k: expected_calibration_error(TR.Y_active, v) for k, v in cal_oof.items()}
def _cv_cal_brier(kind):
    """Brier của bộ hiệu chỉnh khi fit trên các fold KHÁC (chia theo hộ) → không thiên vị isotonic như chọn trên chính tập đã fit."""
    pr = np.zeros(len(TR))
    for a, b in GroupKFold(n_splits=5).split(oof_p, TR.Y_active, groups=TR.household_key):
        if kind == "platt":
            m_ = LogisticRegression(C=1e6).fit(logit(oof_p[a]).reshape(-1, 1), TR.Y_active.iloc[a])
            pr[b] = m_.predict_proba(logit(oof_p[b]).reshape(-1, 1))[:, 1]
        else:
            m_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof_p[a], TR.Y_active.iloc[a])
            pr[b] = m_.predict(oof_p[b])
    return brier_score_loss(TR.Y_active, pr)
CV_CAL_BRIER = {"HGB + Platt": _cv_cal_brier("platt"), "HGB + Isotonic": _cv_cal_brier("iso")}
print("Brier chéo (fold khác) của hai bộ hiệu chỉnh:", {k: round(v, 5) for k, v in CV_CAL_BRIER.items()})
CAL_NAME = min(CV_CAL_BRIER, key=CV_CAL_BRIER.get)
CALIBRATOR = calib_platt if CAL_NAME == "HGB + Platt" else calib_iso
print(f"→ Bộ hiệu chỉnh được chọn (theo Brier chéo trên OOF): {CAL_NAME}")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Hiệu chỉnh hoàn hảo")
for name in ["Logistic (baseline)", "HGB thô", CAL_NAME]:
    p = cands[name]
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    xs = [p[idx == b].mean() for b in range(len(edges) - 1)]
    ys = [VA.Y_active.values[idx == b].mean() for b in range(len(edges) - 1)]
    ax[0].plot(xs, ys, "o-", ms=4, label=name)
ax[0].set_xlabel("Xác suất dự báo"); ax[0].set_ylabel("Tần suất thực tế"); ax[0].set_title("Reliability diagram (Validation)"); ax[0].legend(fontsize=8)
ax[1].hist(p_raw_va, bins=40, alpha=.6, label="HGB thô"); ax[1].hist(CALIBRATOR(p_raw_va), bins=40, alpha=.6, label=CAL_NAME)
ax[1].set_xlabel("Xác suất dự báo"); ax[1].set_title("Phân phối xác suất trước/sau hiệu chỉnh"); ax[1].legend(fontsize=8)
save_fig("fig_2_calibration")

# %%
# ---------------------------------------------------------------------------
# 2.3 Dự báo chi tiêu + khoảng dự báo conformal (Mondrian theo decile mức dự báo)
# ---------------------------------------------------------------------------
oof_y = np.zeros(len(TR))
for tri, tei in folds:
    m = hgb_reg(**BEST_KW).fit(X_tr.iloc[tri], TR.Y_w.iloc[tri])
    oof_y[tei] = np.clip(m.predict(X_tr.iloc[tei]), 0, None)
reg_final = hgb_reg(**BEST_KW).fit(X_tr, TR.Y_w)
yhat_va = np.clip(reg_final.predict(X_va), 0, None)

base_persist = np.clip(VA.spend_28.values, 0, Y_CAP)                   # baseline: lặp lại chi tiêu 28 ngày trước
ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X_tr, TR.Y_w)
yhat_ridge = np.clip(ridge.predict(X_va), 0, None)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
r2 = lambda a, b: float(1 - np.sum((np.asarray(a) - np.asarray(b)) ** 2) / np.sum((np.asarray(a) - np.mean(a)) ** 2))
reg_df = pd.DataFrame({
    "MAE": {"Lặp chi tiêu kỳ trước": mean_absolute_error(VA.Y_w, base_persist), "Ridge": mean_absolute_error(VA.Y_w, yhat_ridge),
            "HGB": mean_absolute_error(VA.Y_w, yhat_va)},
    "RMSE": {"Lặp chi tiêu kỳ trước": rmse(VA.Y_w, base_persist), "Ridge": rmse(VA.Y_w, yhat_ridge), "HGB": rmse(VA.Y_w, yhat_va)},
    "R²": {"Lặp chi tiêu kỳ trước": r2(VA.Y_w, base_persist), "Ridge": r2(VA.Y_w, yhat_ridge), "HGB": r2(VA.Y_w, yhat_va)},
}).round(3)
show_df(reg_df, "Dự báo chi tiêu 28 ngày (Validation)")
save_table(reg_df, "forecast_spend_validation")

# conformal Mondrian: mỗi decile của ŷ có phân vị phần dư riêng (OOF trên Train)
ALPHA = 0.10
cut = np.unique(np.quantile(oof_y, np.linspace(0, 1, 11)))
def _bin(y): return np.clip(np.searchsorted(cut, y, side="right") - 1, 0, len(cut) - 2)
res_oof = np.abs(TR.Y_w.values - oof_y)
conf_q = {b: np.quantile(res_oof[_bin(oof_y) == b], 1 - ALPHA) * (1 + 1 / max((_bin(oof_y) == b).sum(), 1)) for b in range(len(cut) - 1)}
def conformal_interval(yhat):
    q = np.array([conf_q[b] for b in _bin(yhat)])
    return np.clip(yhat - q, 0, None), yhat + q
lo_va, hi_va = conformal_interval(yhat_va)
cov = ((VA.Y_w.values >= lo_va) & (VA.Y_w.values <= hi_va))
cov_tab = pd.DataFrame({"coverage": cov, "width": hi_va - lo_va, "bin": _bin(yhat_va)}).groupby("bin").agg(coverage=("coverage", "mean"), width=("width", "mean"), n=("coverage", "size")).round(3)
print(f"Độ phủ conformal mục tiêu {1 - ALPHA:.0%} → thực tế trên Validation: {cov.mean():.1%} | độ rộng TB {np.mean(hi_va - lo_va):.1f} USD")
show_df(cov_tab, "Độ phủ theo decile mức dự báo (Validation)")
save_table(cov_tab, "conformal_coverage_validation")

# %%
# ---------------------------------------------------------------------------
# 2.4 Quản trị mô hình giai đoạn dự báo (phiên bản, tham số, dấu vết)
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "forecast_active": {"algo": "HistGradientBoostingClassifier", "params": BEST_KW, "calibrator": CAL_NAME,
                        "train_waves": SPLIT_WAVES["train"], "features": FEATURES},
    "forecast_spend": {"algo": "HistGradientBoostingRegressor", "params": BEST_KW, "conformal": f"Mondrian decile, alpha={ALPHA}",
                       "y_cap_winsor": Y_CAP},
}
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "features"} for k, v in MODEL_REGISTRY.items()}, indent=2, ensure_ascii=False))

# %% [markdown]
# # GIAI ĐOẠN 3. Khả năng giải thích mô hình và độ tin cậy
#
# Mục tiêu: người quản lý chương trình phải hiểu *vì sao* mô hình dự báo như vậy và biết lúc nào *không nên tin*. Có bốn lớp:
# toàn cục (Permutation Importance, nhóm đặc trưng), cục bộ (che từng đặc trưng cho một hộ), phản thực (what-if) và kiểm toán độ ổn định / theo phân khúc.
# Việc giải thích mô hình *tác động nhân quả* (CATE) được làm ở mục 4.4.

# %%
# ---------------------------------------------------------------------------
# 3.1 Giải thích toàn cục: Permutation Importance (đo bằng mức tăng Brier trên Validation) + nhóm đặc trưng
# ---------------------------------------------------------------------------
pfi = permutation_importance(clf_final, X_va, VA.Y_active, scoring="neg_brier_score", n_repeats=8, random_state=SEED)
pfi_df = (pd.DataFrame({"feature": FEATURES, "Δ Brier khi xáo trộn": -pfi.importances_mean, "sd": pfi.importances_std})
          .sort_values("Δ Brier khi xáo trộn", ascending=False).reset_index(drop=True))
show_df(pfi_df.head(12).round(5), "Top 12 đặc trưng theo Permutation Importance (Validation)")
save_table(pfi_df, "pfi_forecast")

FEATURE_GROUPS = {
    "Hoạt động & gần đây (recency, visits)": ["recency", "visits_28", "visits_90"],
    "Chi tiêu & xu hướng": ["spend_28", "spend_prev28", "spend_90", "spend_180", "spend_trend", "avg_basket_90"],
    "Nhạy cảm khuyến mãi / coupon": ["coupon_used_90", "coupon_share_90", "promo_share_90", "redeem_90"],
    "Cơ cấu giỏ hàng": ["private_share_90", "fresh_share_90"],
    "Lịch sử bị nhắm": ["targ_prev28", "targ_prev90", "days_since_targ", "active_camp_carry"],
    "Lịch & mùa vụ": ["camp_planned", "season_sin", "season_cos"],
    "Nhân khẩu học": ["has_demo", "age_ord", "income_ord", "hh_size", "kids", "owner"],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURES), "Nhóm đặc trưng phải phủ đủ đặc trưng"

def group_pfi(model, X, y, groups, n_repeats=8, seed=SEED):
    r = np.random.default_rng(seed)
    base = brier_score_loss(y, model.predict_proba(X)[:, 1])
    out = {}
    for g, cols in groups.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = r.permutation(len(X))
            Xp[cols] = X[cols].values[perm]
            deltas.append(brier_score_loss(y, model.predict_proba(Xp)[:, 1]) - base)
        out[g] = (np.mean(deltas), np.std(deltas))
    return pd.DataFrame(out, index=["Δ Brier", "sd"]).T.sort_values("Δ Brier", ascending=False)
gp = group_pfi(clf_final, X_va, VA.Y_active, FEATURE_GROUPS)
show_df(gp.round(5), "Tầm quan trọng theo nhóm đặc trưng (xáo trộn cả nhóm cùng lúc)")
save_table(gp, "pfi_forecast_groups")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
top = pfi_df.head(12).iloc[::-1]
ax[0].barh(top.feature, top["Δ Brier khi xáo trộn"], xerr=top.sd, color="#4C78A8"); ax[0].set_title("Top 12 đặc trưng (PFI)")
ax[0].set_xlabel("Tăng Brier khi xáo trộn")
ax[1].barh(gp.index[::-1], gp["Δ Brier"][::-1], color="#F58518"); ax[1].set_title("Theo nhóm đặc trưng"); ax[1].set_xlabel("Tăng Brier khi xáo trộn")
save_fig("fig_3_importance")

# %%
# ---------------------------------------------------------------------------
# 3.2 Giải thích cục bộ: che từng đặc trưng (thay bằng trung vị Train) cho 3 hộ minh họa
# ---------------------------------------------------------------------------
MED = X_tr.median()
def local_occlusion(model, x_row, med=MED):
    base = model.predict_proba(x_row.to_frame().T)[:, 1][0]
    deltas = {}
    for f in FEATURES:
        x2 = x_row.copy(); x2[f] = med[f]
        deltas[f] = base - model.predict_proba(x2.to_frame().T)[:, 1][0]
    return base, pd.Series(deltas).sort_values(key=np.abs, ascending=False)

p_cal_va = CALIBRATOR(p_raw_va)
order = np.argsort(p_cal_va)
exemplars = {"Rủi ro rời bỏ cao": order[int(0.02 * len(order))], "Trung bình": order[len(order) // 2], "Rủi ro thấp": order[int(0.98 * len(order))]}
local_tab = []
for lab, i in exemplars.items():
    base, d = local_occlusion(clf_final, X_va.iloc[i])
    for f in d.index[:4]:
        local_tab.append({"hộ minh họa": lab, "p_active (thô)": round(base, 3), "đặc trưng": f,
                          "giá trị": round(float(X_va.iloc[i][f]), 2), "đóng góp vào p": round(float(d[f]), 3)})
local_df = pd.DataFrame(local_tab)
show_df(local_df, "Giải thích cục bộ – 4 đặc trưng đóng góp lớn nhất cho mỗi hộ (đóng góp > 0: kéo xác suất quay lại lên)")
save_table(local_df, "local_explanations")

# %%
# ---------------------------------------------------------------------------
# 3.3 Phân tích phản thực (what-if) và đường phụ thuộc từng phần (PDP / ICE)
# ---------------------------------------------------------------------------
def pdp_ice(model, X, feat, grid, n_ice=60, seed=SEED):
    r = np.random.default_rng(seed)
    idx = r.choice(len(X), n_ice, replace=False)
    ice = np.zeros((n_ice, len(grid)))
    for j, v in enumerate(grid):
        Xs = X.iloc[idx].copy(); Xs[feat] = v
        ice[:, j] = model.predict_proba(Xs)[:, 1]
    pdp = np.zeros(len(grid))
    sub = X.sample(min(1500, len(X)), random_state=seed)
    for j, v in enumerate(grid):
        Xs = sub.copy(); Xs[feat] = v
        pdp[j] = model.predict_proba(Xs)[:, 1].mean()
    return pdp, ice
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, (feat, grid) in zip(ax, [("recency", np.linspace(0, 85, 18)), ("visits_90", np.linspace(1, 30, 15)), ("spend_28", np.linspace(0, 400, 17))]):
    pdp, ice = pdp_ice(clf_final, X_va, feat, grid)
    a.plot(grid, ice.T, color="grey", alpha=.15); a.plot(grid, pdp, color="#E45756", lw=2.2)
    a.set_xlabel(feat); a.set_ylabel("p(quay lại)"); a.set_title(f"PDP/ICE: {feat}")
save_fig("fig_3_pdp_ice")

# what-if: nếu một hộ "ngủ đông" quay lại 14 ngày sớm hơn thì xác suất thay đổi bao nhiêu (chỉ minh họa mô hình, không phải nhân quả)
whatif = []
base_p = clf_final.predict_proba(X_va)[:, 1]
for lab, changes in {"recency − 14 ngày": {"recency": lambda s: np.clip(s - 14, 0, None)},
                     "visits_90 + 3": {"visits_90": lambda s: s + 3},
                     "spend_28 × 1,2": {"spend_28": lambda s: s * 1.2}}.items():
    Xc = X_va.copy()
    for k, fn in changes.items():
        Xc[k] = fn(Xc[k])
    whatif.append({"kịch bản phản thực": lab, "Δ p_active trung bình": float((clf_final.predict_proba(Xc)[:, 1] - base_p).mean())})
whatif_df = pd.DataFrame(whatif).set_index("kịch bản phản thực").round(4)
show_df(whatif_df, "Phân tích phản thực trên mô hình dự báo (LƯU Ý: thể hiện hành vi mô hình, chưa phải hiệu ứng nhân quả)")

# %%
# ---------------------------------------------------------------------------
# 3.4 Độ tin cậy và kiểm toán: ổn định theo hạt giống/bootstrap, và chất lượng theo phân khúc
# ---------------------------------------------------------------------------
ranks, g_ranks = [], []
for b in range(5):
    r = np.random.default_rng(SEED + 100 + b)
    ib = r.choice(len(TR), len(TR), replace=True)
    mb = hgb_clf(**BEST_KW).fit(X_tr.iloc[ib], TR.Y_active.iloc[ib])
    pb = permutation_importance(mb, X_va, VA.Y_active, scoring="neg_brier_score", n_repeats=3, random_state=b)
    ranks.append(pd.Series(-pb.importances_mean, index=FEATURES))
    g_ranks.append(group_pfi(mb, X_va, VA.Y_active, FEATURE_GROUPS, n_repeats=3, seed=b)["Δ Brier"])
rank_df = pd.concat(ranks, axis=1)
rho = [stats.spearmanr(rank_df.iloc[:, i], rank_df.iloc[:, j])[0] for i in range(5) for j in range(i + 1, 5)]
print(f"Độ ổn định thứ hạng PFI giữa 5 lần bootstrap: Spearman trung bình = {np.mean(rho):.3f} (min {np.min(rho):.3f})")
g_rank_df = pd.concat(g_ranks, axis=1)
rho_g = [stats.spearmanr(g_rank_df.iloc[:, i], g_rank_df.iloc[:, j])[0] for i in range(5) for j in range(i + 1, 5)]
print(f"Ở mức NHÓM đặc trưng: Spearman trung bình = {np.mean(rho_g):.3f} → đặc trưng đơn lẻ tương quan mạnh nên xếp hạng cá thể kém ổn định; "
      f"kết luận nên dựa trên nhóm.")
print("Nhóm luôn đứng đầu:", g_rank_df.idxmax().value_counts().to_dict())
top5_sets = [set(rank_df.iloc[:, i].nlargest(5).index) for i in range(5)]
print("Đặc trưng luôn nằm trong top-5 ở mọi lần:", sorted(set.intersection(*top5_sets)))

# kiểm toán theo phân khúc (chất lượng dự báo không được tệ bất thường ở một nhóm)
seg = pd.Series("Chưa rõ nhân khẩu học", index=VA.index)
seg[(VA.income_ord >= 0) & (VA.income_ord <= 3)] = "Thu nhập thấp–TB (<50K)"
seg[VA.income_ord >= 4] = "Thu nhập khá+ (≥50K)"
tenure_b = pd.cut(VA.visits_90, [-1, 5, 12, 10_000], labels=["Ít ghé (≤5 lần/90 ngày)", "Vừa (6–12)", "Thường xuyên (>12)"])
aud = []
for name, g in [("phân khúc thu nhập", seg), ("tần suất ghé cửa hàng", tenure_b)]:
    for lv in pd.Series(g).dropna().unique():
        m = (g == lv).values
        if m.sum() >= 80 and VA.Y_active[m].nunique() > 1:
            aud.append({"nhóm": name, "mức": str(lv), "n": int(m.sum()), "tỷ lệ quay lại": VA.Y_active[m].mean(),
                        "AUC": roc_auc_score(VA.Y_active[m], p_cal_va[m]), "Brier": brier_score_loss(VA.Y_active[m], p_cal_va[m]),
                        "ECE": expected_calibration_error(VA.Y_active[m], p_cal_va[m])})
aud_df = pd.DataFrame(aud).set_index(["nhóm", "mức"]).round(3)
show_df(aud_df, "Kiểm toán chất lượng dự báo theo phân khúc (Validation, sau hiệu chỉnh)")
save_table(aud_df, "audit_segments")

# %% [markdown]
# # GIAI ĐOẠN 4. Phân tích nhân quả và mô hình hóa tác động biên
#
# Câu hỏi nhân quả: *nhận coupon trong đợt làm chi tiêu 28 ngày sau của một hộ thay đổi bao nhiêu (USD)?*
#
# Dữ liệu quan sát, coupon không ngẫu nhiên → cần (i) giả định nhận dạng rõ ràng, (ii) kiểm tra overlap, (iii) ước lượng ATE bằng Double ML / AIPW,
# (iv) ước lượng CATE (DR-learner) để xếp hạng can thiệp, (v) kiểm chứng ngoài mẫu (Validation) và (vi) placebo + độ nhạy.
#
# **Giả định nhận dạng:** *Conditional ignorability* – cho trước đặc trưng X (lịch sử mua, nhạy cảm khuyến mãi, lịch sử bị nhắm, mùa vụ, nhân khẩu học), việc được nhắm độc lập với chi tiêu tiềm năng;
# *Positivity* – 0 < P(T=1|X) < 1 (kiểm tra và cắt tỉa); *SUTVA* – coupon của hộ này không ảnh hưởng chi tiêu hộ khác.

# %%
# ---------------------------------------------------------------------------
# 4.1 Tiện ích: cross-fitting theo hộ, bootstrap theo cụm hộ
# ---------------------------------------------------------------------------
NUIS_REG = dict(max_iter=120, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0)
NUIS_CLF = dict(max_iter=150, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0)

def fit_nuisance(Xa, ya, ta, Xb):
    """Huấn luyện trên (Xa,ya,ta), dự báo cho Xb: e(x), m(x)=E[Y|x], mu1(x), mu0(x)."""
    e = hgb_clf(**NUIS_CLF).fit(Xa, ta).predict_proba(Xb)[:, 1]
    m = hgb_reg(**NUIS_REG).fit(Xa, ya).predict(Xb)
    mu1 = hgb_reg(**NUIS_REG).fit(Xa[ta == 1], ya[ta == 1]).predict(Xb)
    mu0 = hgb_reg(**NUIS_REG).fit(Xa[ta == 0], ya[ta == 0]).predict(Xb)
    return e, m, mu1, mu0

def crossfit(X, y, t, groups, n_splits=5):
    n = len(X)
    out = {k: np.zeros(n) for k in ["e", "m", "mu1", "mu0"]}
    for tri, tei in GroupKFold(n_splits=n_splits).split(X, y, groups):
        e, m, mu1, mu0 = fit_nuisance(X.iloc[tri], y[tri], t[tri], X.iloc[tei])
        for k, v in zip(["e", "m", "mu1", "mu0"], [e, m, mu1, mu0]):
            out[k][tei] = v
    return out

def dr_score(y, t, e, mu1, mu0, clip=PS_CLIP):
    ec = np.clip(e, *clip)
    return mu1 - mu0 + t * (y - mu1) / ec - (1 - t) * (y - mu0) / (1 - ec)

def cluster_se_mean(v, groups):
    """SE của trung bình với gom cụm theo hộ."""
    df = pd.DataFrame({"v": v, "g": groups})
    s = df.groupby("g").v.agg(["sum", "size"])
    mean = v.mean()
    resid = s["sum"] - mean * s["size"]
    return float(np.sqrt((resid ** 2).sum()) / len(v))

def plr_dml(y, t, m, e, groups, clip=None, mask=None):
    """Partially Linear Regression (Robinson/DML): θ = Σ ỹ·t̃ / Σ t̃²; SE gom cụm theo hộ."""
    if mask is None:
        mask = np.ones(len(y), bool)
    yt, tt = (y - m)[mask], (t - e)[mask]
    theta = (yt * tt).sum() / (tt ** 2).sum()
    psi = tt * (yt - theta * tt)
    df = pd.DataFrame({"psi": psi, "g": np.asarray(groups)[mask]})
    sg = df.groupby("g").psi.sum()
    se = float(np.sqrt((sg ** 2).sum()) / (tt ** 2).sum())
    return theta, se, int(mask.sum())

def boot_by_cluster(values, groups, B=400, seed=SEED, num=None, den=None):
    """Bootstrap theo cụm hộ cho trung bình (hoặc tỷ số num/den). Trả về (mean, lo, hi)."""
    r = np.random.default_rng(seed)
    g_codes, g_idx = np.unique(groups, return_inverse=True)
    G = len(g_codes)
    S = np.bincount(g_idx, weights=values, minlength=G)
    N = np.bincount(g_idx, minlength=G).astype(float)
    W = r.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)
    est = (W @ S) / np.maximum(W @ N, 1)
    return float(values.mean()), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))

# %%
# ---------------------------------------------------------------------------
# 4.2 Overlap, ATE: so sánh thô → hồi quy hiệu chỉnh → Double ML (PLR) → AIPW
# ---------------------------------------------------------------------------
import statsmodels.api as sm
y_tr, t_tr, g_tr = TR.Y_w.values, TR["T"].values, TR.household_key.values
t0 = time.time()
CF = crossfit(X_tr, y_tr, t_tr, g_tr)
print(f"Cross-fitting xong ({time.time() - t0:.0f}s)")

e_oof = CF["e"]
print(f"Propensity (OOF): AUC = {roc_auc_score(t_tr, e_oof):.3f} | Brier = {brier_score_loss(t_tr, e_oof):.4f}")
supp_tr = (e_oof >= PS_CLIP[0]) & (e_oof <= PS_CLIP[1])
print(f"Miền positivity [{PS_CLIP[0]}, {PS_CLIP[1]}]: giữ {supp_tr.mean():.1%} đơn vị Train "
      f"(treated giữ {supp_tr[t_tr == 1].mean():.1%}, control giữ {supp_tr[t_tr == 0].mean():.1%})")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hist(e_oof[t_tr == 0], bins=50, alpha=.6, density=True, label="Không được nhắm"); ax[0].hist(e_oof[t_tr == 1], bins=50, alpha=.6, density=True, label="Được nhắm")
ax[0].axvline(PS_CLIP[0], color="k", ls=":"); ax[0].axvline(PS_CLIP[1], color="k", ls=":"); ax[0].set_xlabel("Propensity score ước lượng (OOF)"); ax[0].legend(); ax[0].set_title("Overlap")
ax[1].hist(np.log10(np.clip(e_oof, 1e-4, 1)), bins=50, color="#4C78A8"); ax[1].set_xlabel("log10 propensity"); ax[1].set_title("Đuôi trái: vùng gần như không bao giờ được nhắm")
save_fig("fig_4_overlap")

# (1) so sánh thô, (2) OLS hiệu chỉnh, (3) PLR-DML, (4) AIPW
naive = y_tr[t_tr == 1].mean() - y_tr[t_tr == 0].mean()
se_naive = np.sqrt(y_tr[t_tr == 1].var(ddof=1) / (t_tr == 1).sum() + y_tr[t_tr == 0].var(ddof=1) / (t_tr == 0).sum())
ols = sm.OLS(y_tr, sm.add_constant(np.column_stack([t_tr, X_tr.drop(columns=["spend_trend"]).values]))).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g_tr)[0]})
th_all, se_all, n_all = plr_dml(y_tr, t_tr, CF["m"], e_oof, g_tr)
th_sup, se_sup, n_sup = plr_dml(y_tr, t_tr, CF["m"], e_oof, g_tr, mask=supp_tr)
GAM_tr = dr_score(y_tr, t_tr, e_oof, CF["mu1"], CF["mu0"])
aipw_all, aipw_se_all = GAM_tr.mean(), cluster_se_mean(GAM_tr, g_tr)
aipw_sup, aipw_se_sup = GAM_tr[supp_tr].mean(), cluster_se_mean(GAM_tr[supp_tr], g_tr[supp_tr])

def _row(name, est, se, n):
    return {"phương pháp": name, "ước lượng (USD/hộ/đợt)": est, "SE": se, "CI95 thấp": est - 1.96 * se, "CI95 cao": est + 1.96 * se, "n": n}
ate_df = pd.DataFrame([
    _row("So sánh thô (treated − control)", naive, se_naive, len(y_tr)),
    _row("OLS có đặc trưng (SE gom cụm hộ)", ols.params[1], ols.bse[1], len(y_tr)),
    _row("Double ML – PLR (toàn mẫu)", th_all, se_all, n_all),
    _row("Double ML – PLR (miền positivity)", th_sup, se_sup, n_sup),
    _row("AIPW (toàn mẫu, e cắt tỉa)", aipw_all, aipw_se_all, len(y_tr)),
    _row("AIPW (miền positivity)", aipw_sup, aipw_se_sup, int(supp_tr.sum())),
]).set_index("phương pháp").round(2)
show_df(ate_df, "Tác động trung bình (ATE) của coupon lên chi tiêu 28 ngày – Train")
save_table(ate_df, "ate_estimates")
print(f"→ So sánh thô cho {naive:,.1f} USD, trong khi Double ML trong miền positivity cho {th_sup:,.1f} USD "
      f"(CI95 [{th_sup - 1.96 * se_sup:,.1f}; {th_sup + 1.96 * se_sup:,.1f}]): " +
      ("gần như toàn bộ chênh lệch thô là do chọn mẫu." if abs(th_sup) < 0.2 * abs(naive) else "một phần lớn chênh lệch thô là do chọn mẫu."))

# %%
# ---------------------------------------------------------------------------
# 4.3 Kiểm tra độ vững: placebo (kết quả 'trước' can thiệp), độ nhạy (Cinelli–Hazlett) và 'đối chứng' bằng bỏ nhóm đặc trưng
# ---------------------------------------------------------------------------
# (a) Placebo kết quả: chi tiêu trong 28 ngày TRƯỚC thời điểm quyết định không thể bị coupon của đợt này gây ra.
#     Đặc trưng được dựng lại tại d−28 để không chứa chính kết quả placebo.
X_plc = make_features(TR.hi.values, TR.decision_day.values - WAVE_LEN)[FEATURES]
y_plc = TR.Y_pre28_w.values
CF_plc = crossfit(X_plc, y_plc, t_tr, g_tr)
th_plc, se_plc, _ = plr_dml(y_plc, t_tr, CF_plc["m"], CF_plc["e"], g_tr, mask=(CF_plc["e"] >= PS_CLIP[0]) & (CF_plc["e"] <= PS_CLIP[1]))
# placebo thô để so sánh: nếu không hiệu chỉnh thì 'tác động lên quá khứ' lớn đến mức nào
naive_plc = y_plc[t_tr == 1].mean() - y_plc[t_tr == 0].mean()

# (b) Độ nhạy: robustness value (Cinelli & Hazlett, 2020) từ thống kê t của PLR
dof = n_sup - len(FEATURES) - 1
tstat = th_sup / se_sup
f_cohen = abs(tstat) / np.sqrt(dof)
RV = 0.5 * (np.sqrt(f_cohen ** 4 + 4 * f_cohen ** 2) - f_cohen ** 2)
f_lo = max(abs(th_sup - 1.96 * se_sup) / se_sup, 0) / np.sqrt(dof) if (th_sup - 1.96 * se_sup) > 0 else 0.0
RV_a = 0.5 * (np.sqrt(f_lo ** 4 + 4 * f_lo ** 2) - f_lo ** 2) if f_lo > 0 else 0.0

# (c) Benchmark: bỏ từng nhóm đặc trưng quan sát được → ước lượng PLR đổi bao nhiêu (gợi ý quy mô nhiễu mà một biến 'mạnh' có thể gây ra)
fast_reg = dict(max_iter=80, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=60)
fast_clf = dict(max_iter=80, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=80)
def plr_with(cols, mask_support=True):
    Xc = X_tr[cols]
    m, e = np.zeros(len(Xc)), np.zeros(len(Xc))
    for tri, tei in GroupKFold(n_splits=4).split(Xc, y_tr, g_tr):
        m[tei] = hgb_reg(**fast_reg).fit(Xc.iloc[tri], y_tr[tri]).predict(Xc.iloc[tei])
        e[tei] = hgb_clf(**fast_clf).fit(Xc.iloc[tri], t_tr[tri]).predict_proba(Xc.iloc[tei])[:, 1]
    mk = (e >= PS_CLIP[0]) & (e <= PS_CLIP[1]) if mask_support else None
    th, se, n = plr_dml(y_tr, t_tr, m, e, g_tr, mask=mk)
    return th, se, n
bench = []
th_full, se_full, n_full = plr_with(FEATURES)
bench.append({"cấu hình": "Đủ đặc trưng (tham chiếu, bản nhanh)", "θ": th_full, "SE": se_full, "n hỗ trợ": n_full})
for g, cols in FEATURE_GROUPS.items():
    keep = [c for c in FEATURES if c not in cols]
    th, se, n = plr_with(keep)
    bench.append({"cấu hình": f"Bỏ nhóm: {g}", "θ": th, "SE": se, "n hỗ trợ": n})
bench_df = pd.DataFrame(bench).set_index("cấu hình").round(2)
show_df(bench_df, "Độ nhạy theo nhóm đặc trưng: bỏ nhóm nào thì ước lượng ATE (miền positivity) thay đổi")
save_table(bench_df, "ate_drop_group_benchmark")

rob_df = pd.DataFrame({
    "chỉ số": ["Placebo (kết quả 28 ngày trước) – PLR", "Placebo – so sánh thô", "Robustness value RV (ATE = 0)", "Robustness value RV (CI chạm 0)"],
    "giá trị": [f"{th_plc:.2f} ± {1.96 * se_plc:.2f} USD", f"{naive_plc:.2f} USD", f"{RV:.3%}", f"{RV_a:.3%}"],
    "diễn giải": ["≈ 0 ⇒ bộ đặc trưng hấp thụ phần lớn chọn mẫu theo lịch sử chi tiêu" if abs(th_plc) < 1.96 * se_plc
                  else "KHÁC 0 ⇒ còn nhiễu chưa hấp thụ hết; diễn giải ATE thận trọng",
                  "lệch lớn ⇒ hộ được nhắm vốn đã chi tiêu cao từ trước (nhiễu rõ rệt)",
                  "% phương sai của T và Y mà nhiễu ẩn phải giải thích để xóa hiệu ứng",
                  "ngưỡng để khoảng tin cậy 95% bao gồm 0"],
}).set_index("chỉ số")
show_df(rob_df, "Kiểm tra độ vững của nhận dạng nhân quả")
save_table(rob_df, "causal_robustness")

# %%
# ---------------------------------------------------------------------------
# 4.4 CATE: DR-learner (hồi quy điểm DR trên đặc trưng) và T-learner đối sánh
# ---------------------------------------------------------------------------
# mô hình nuisance cuối cùng học trên toàn Train để áp cho Validation / dữ liệu tương lai
e_f, m_f, mu1_f, mu0_f = fit_nuisance(X_tr, y_tr, t_tr, X_va)
y_va, t_va, g_va = VA.Y_w.values, VA["T"].values, VA.household_key.values
GAM_va = dr_score(y_va, t_va, e_f, mu1_f, mu0_f)
supp_va = (e_f >= PS_CLIP[0]) & (e_f <= PS_CLIP[1])
print(f"Validation: {supp_va.mean():.1%} đơn vị nằm trong miền positivity")

TAU_GRID = {
    "nông":  dict(max_iter=100, learning_rate=0.05, max_leaf_nodes=6, min_samples_leaf=200, l2_regularization=5.0),
    "vừa":   dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=10, min_samples_leaf=120, l2_regularization=5.0),
    "sâu":   dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=20, min_samples_leaf=80, l2_regularization=5.0),
}
class ConstTau:
    """Mô hình CATE hằng số (không dị biệt) – ứng viên 'null' để dữ liệu tự quyết định có cần dị biệt hay không."""
    def __init__(self, c): self.c = float(c)
    def predict(self, X): return np.full(len(X), self.c)

fit_i, ev_i = (TR.wave <= 5).values, (TR.wave >= 6).values
sel = {}
c_inner = GAM_tr[fit_i & supp_tr].mean()
sel["hằng số (ATE)"] = float(np.mean((GAM_tr[ev_i & supp_tr] - c_inner) ** 2))
for nm, kw in TAU_GRID.items():
    mt = HistGradientBoostingRegressor(random_state=SEED, early_stopping=False, **kw).fit(X_tr[fit_i & supp_tr], GAM_tr[fit_i & supp_tr])
    sel[nm] = float(np.mean((GAM_tr[ev_i & supp_tr] - mt.predict(X_tr[ev_i & supp_tr])) ** 2))   # MSE điểm DR (proxy hợp lệ để chọn mô hình CATE)
print("MSE điểm DR trên holdout nội bộ (đợt 6–7 của Train):", {k: round(v, 1) for k, v in sel.items()})
TAU_CFG = min(sel, key=sel.get)
print("→ Cấu hình CATE được chọn:", TAU_CFG)

# Bagging theo hộ: bất định của CATE (dùng cho Guardrail)
N_BAG = 15
r_bag = np.random.default_rng(SEED + 7)
idx_sup = np.where(supp_tr)[0]
_rows_by_hh = pd.DataFrame({"i": idx_sup, "g": g_tr[idx_sup]}).groupby("g").i.apply(lambda s: s.values).to_dict()
_uh = np.array(list(_rows_by_hh))
def boot_rows_support(rng):
    """Bootstrap THEO HỘ có hoàn lại (hộ trúng nhiều lần thì lặp nhiều lần) trên tập trong miền positivity."""
    pick = rng.choice(_uh, len(_uh), replace=True)
    return np.concatenate([_rows_by_hh[h] for h in pick])
def _fit_tau(ix, seed):
    if TAU_CFG == "hằng số (ATE)":
        return ConstTau(GAM_tr[ix].mean())
    return HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **TAU_GRID[TAU_CFG]).fit(X_tr.iloc[ix], GAM_tr[ix])
tau_model = _fit_tau(idx_sup, SEED)
bag_models = []
for b in range(N_BAG):
    bag_models.append(_fit_tau(boot_rows_support(r_bag), SEED + b))
def tau_predict(X):
    P = np.column_stack([m.predict(X) for m in bag_models])
    return P.mean(axis=1), P.std(axis=1)

tau_va = tau_model.predict(X_va)
tau_T_va = mu1_f - mu0_f                                         # T-learner (đối sánh)
tau_tr_hat = tau_model.predict(X_tr)
# Mô hình dị biệt có điều kiện (luôn huấn luyện để so sánh/diễn giải, kể cả khi dữ liệu chọn 'hằng số')
tau_het = HistGradientBoostingRegressor(random_state=SEED, early_stopping=False, **TAU_GRID["nông"]).fit(X_tr[supp_tr], GAM_tr[supp_tr])
tau_het_va = tau_het.predict(X_va)

tau_bag_va, tau_sd_va = tau_predict(X_va)
print(f"CATE (Validation, trong miền): trung bình={tau_va[supp_va].mean():.2f} | sd={tau_va[supp_va].std():.2f} | "
      f"sd bagging trung bình={tau_sd_va[supp_va].mean():.2f}")

# %%
# ---------------------------------------------------------------------------
# 4.5 Kiểm chứng dị biệt ngoài mẫu: hiệu chỉnh CATE (best linear predictor) và uplift/Qini trên Validation
# ---------------------------------------------------------------------------
sv = supp_va
# (a) Calibration test (Chernozhukov et al.): Γ ~ β0 + β1·τ̂  ; β1 ≈ 1 ⇒ dị biệt được nắm bắt, β1 ≈ 0 ⇒ chỉ là nhiễu
def blp(gamma, tau, groups):
    Xb = sm.add_constant(tau - tau.mean())
    fit = sm.OLS(gamma, Xb).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    return fit.params, fit.bse
blp_rows = []
for nm, tv in [("DR-learner dị biệt (nông)", tau_het_va), ("T-learner", tau_T_va)]:
    p, s = blp(GAM_va[sv], tv[sv], g_va[sv])
    blp_rows.append({"mô hình": nm, "β0 (≈ATE)": p[0], "β1 (độ hiệu chỉnh dị biệt)": p[1], "SE β1": s[1], "p-value (β1=0)": 2 * (1 - stats.norm.cdf(abs(p[1] / s[1])))})
blp_df = pd.DataFrame(blp_rows).set_index("mô hình").round(3)
show_df(blp_df, "Kiểm định hiệu chỉnh CATE trên Validation (β1≈1 là lý tưởng)")
save_table(blp_df, "cate_blp_validation")

# (a') ATE: Train so với Validation – kiểm tra tính ổn định theo thời gian của ước lượng
ate_va, ate_va_se = float(GAM_va[sv].mean()), cluster_se_mean(GAM_va[sv], g_va[sv])
ate_tv = pd.DataFrame({"ước lượng (USD/hộ/đợt)": [aipw_sup, ate_va], "SE": [aipw_se_sup, ate_va_se],
                       "CI95 thấp": [aipw_sup - 1.96 * aipw_se_sup, ate_va - 1.96 * ate_va_se],
                       "CI95 cao": [aipw_sup + 1.96 * aipw_se_sup, ate_va + 1.96 * ate_va_se], "n": [int(supp_tr.sum()), int(sv.sum())]},
                      index=["AIPW – Train (miền positivity)", "AIPW/DR – Validation (miền positivity)"]).round(2)
show_df(ate_tv, "ATE theo điểm DR: Train so với Validation (ổn định theo thời gian?)")
save_table(ate_tv, "ate_train_vs_validation")
# (a'') Chẩn đoán theo từng đợt Validation (chỉ để giải thích, KHÔNG dùng để chỉnh mô hình)
_wv = VA["wave"].values
_rows = []
for _w in sorted(np.unique(_wv)):
    _m = sv & (_wv == _w)
    _rows.append({"đợt": int(_w), "tỷ lệ nhắm (toàn đợt)": float(t_va[_wv == _w].mean()), "n trong miền positivity": int(_m.sum()),
                  "ATE (DR)": float(GAM_va[_m].mean()), "SE (gom cụm)": cluster_se_mean(GAM_va[_m], g_va[_m])})
ate_wave = pd.DataFrame(_rows).set_index("đợt").round(3)
show_df(ate_wave, "ATE (điểm DR) theo từng đợt Validation – chẩn đoán độ ổn định")
save_table(ate_wave, "ate_validation_by_wave")
_z = (ate_va - aipw_sup) / np.hypot(aipw_se_sup, ate_va_se)
print(f"→ Chênh lệch ATE Validation − Train = {ate_va - aipw_sup:+.1f} USD (z = {_z:.2f}): " +
      ("khác biệt có ý nghĩa thống kê ở mức 5%." if abs(_z) >= 1.96 else "chưa có ý nghĩa thống kê ở mức 5% nhưng rất lớn so với ngưỡng hòa vốn → ước lượng ATE KHÔNG ổn định theo thời gian.") +
      " ATE bị chi phối bởi các hộ propensity thấp nên rất nhiễu.")

# (b) Theo ngũ phân vị của τ̂: tác động DR thực nghiệm trong từng nhóm
q = pd.qcut(pd.Series(tau_het_va[sv]).rank(method="first"), 5, labels=False).values      # dùng CATE dị biệt (CATE được chọn có thể là hằng số → không chia nhóm được)
gp_df = (pd.DataFrame({"quintile": q + 1, "tau_hat": tau_het_va[sv], "gamma": GAM_va[sv], "g": g_va[sv]})
         .groupby("quintile").agg(tau_hat_mean=("tau_hat", "mean"), dr_effect=("gamma", "mean"), n=("gamma", "size")))
gp_df["SE"] = [cluster_se_mean(GAM_va[sv][q == i], g_va[sv][q == i]) for i in range(5)]
show_df(gp_df.round(2), "Tác động thực nghiệm (điểm DR) theo ngũ phân vị CATE dự báo – Validation")
save_table(gp_df, "cate_quintiles_validation")

# (c) Uplift / Qini: cộng dồn điểm DR khi xếp hạng hộ theo điểm số; so với ngẫu nhiên và chính sách heuristic RFM
def qini_curve(score, gamma, groups, n_pts=21):
    order = np.argsort(-score)
    cum = np.cumsum(gamma[order]) / len(gamma)           # gain cộng dồn (USD/hộ-đợt, chia tổng mẫu)
    ks = np.linspace(0, 1, n_pts)
    pos = np.clip((ks * len(gamma)).astype(int) - 1, 0, len(gamma) - 1)
    return ks, np.where(ks > 0, cum[pos], 0.0)
scores_va = {
    "DR-learner dị biệt (nông)": tau_het_va, "T-learner": tau_T_va,
    "Heuristic RFM (chi tiêu 90 ngày)": VA.spend_90.values,
    "Heuristic nhạy coupon": VA.coupon_share_90.values,
    "Ngẫu nhiên": np.random.default_rng(SEED).random(len(VA)),
}
plt.figure(figsize=(7, 4.2))
auuc = {}
for nm, sc in scores_va.items():
    ks, qv = qini_curve(sc[sv], GAM_va[sv], g_va[sv])
    auuc[nm] = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(qv, ks)
    plt.plot(ks, qv, label=nm, lw=2.2 if "DR" in nm else 1.3, ls="--" if "Ngẫu" in nm else "-")
plt.xlabel("Tỷ lệ hộ được can thiệp (xếp hạng theo điểm số)"); plt.ylabel("Gain tăng thêm cộng dồn (USD/hộ-đợt)")
plt.title("Qini / Uplift trên Validation (điểm DR)"); plt.legend(fontsize=8)
save_fig("fig_4_qini")
auuc_df = pd.Series(auuc, name="AUUC").sort_values(ascending=False).round(3).to_frame()
show_df(auuc_df, "AUUC trên Validation (cao hơn = xếp hạng tốt hơn)")
save_table(auuc_df, "auuc_validation")

# %%
# ---------------------------------------------------------------------------
# 4.6 Giải thích mô hình tác động và chuyển thành hành động (PFI nhóm, PDP, cây thay thế)
# ---------------------------------------------------------------------------
def pfi_tau(model, X, gamma, groups_, n_rep=6, seed=SEED):
    r = np.random.default_rng(seed)
    base = np.mean((gamma - model.predict(X)) ** 2)
    out = {}
    for g, cols in groups_.items():
        d = []
        for _ in range(n_rep):
            Xp = X.copy(); Xp[cols] = X[cols].values[r.permutation(len(X))]
            d.append(np.mean((gamma - model.predict(Xp)) ** 2) - base)
        out[g] = np.mean(d)
    return pd.Series(out).sort_values(ascending=False)
pt = pfi_tau(tau_het, X_va[sv], GAM_va[sv], FEATURE_GROUPS)
show_df(pt.round(3).to_frame("Δ MSE điểm DR"), "Nhóm đặc trưng quyết định dị biệt tác động (PFI trên mô hình CATE)")

from sklearn.tree import DecisionTreeRegressor, export_text
sur = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300, random_state=SEED).fit(X_va[sv], tau_het_va[sv])
print("\nCây thay thế (surrogate, độ sâu 3) mô tả quy tắc của mô hình CATE:")
print(export_text(sur, feature_names=FEATURES, decimals=1))
print(f"R² của cây thay thế so với CATE: {sur.score(X_va[sv], tau_het_va[sv]):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
for a, (feat, grid) in zip(ax, [("spend_90", np.linspace(20, 900, 20)), ("recency", np.linspace(0, 80, 17)), ("coupon_share_90", np.linspace(0, 0.08, 17))]):
    pdp = []
    sub = X_va[sv].sample(min(1500, int(sv.sum())), random_state=SEED)
    for v in grid:
        Xs = sub.copy(); Xs[feat] = v; pdp.append(tau_het.predict(Xs).mean())
    a.plot(grid, pdp, color="#E45756", lw=2); a.axhline(0, color="k", lw=.7); a.set_xlabel(feat); a.set_ylabel("CATE (USD)"); a.set_title(f"PDP của CATE theo {feat}")
save_fig("fig_4_cate_pdp")

# %%
# ---------------------------------------------------------------------------
# 4.7 Hàm giá trị: chi phí coupon KỲ VỌNG lấy từ dữ liệu + chi phí liên hệ (giả định)
# ---------------------------------------------------------------------------
tr_t = (t_tr == 1)
c1 = hgb_reg(**NUIS_REG).fit(X_tr[tr_t], TR.Y_coupon_cost.values[tr_t])           # chiết khấu coupon nếu được nhắm
c0 = hgb_reg(**NUIS_REG).fit(X_tr[~tr_t], TR.Y_coupon_cost.values[~tr_t])         # chiết khấu coupon nền nếu không nhắm (coupon nhà sản xuất...)
def cost_hat(X):
    """Chi phí TĂNG THÊM của một lần gửi = chiết khấu thêm do được nhắm (T-learner) + chi phí liên hệ (giả định)."""
    inc = np.clip(c1.predict(X) - c0.predict(X), 0, None)
    return CONTACT_COST + inc
print(f"Chiết khấu coupon thực tế TB của hộ được nhắm (Train): {TR.Y_coupon_cost[tr_t].mean():.2f} USD; "
      f"của hộ không được nhắm: {TR.Y_coupon_cost[~tr_t].mean():.2f} USD")
print(f"Chi phí tăng thêm mỗi lần gửi (Validation) = {cost_hat(X_va).mean():.2f} USD (gồm {CONTACT_COST:.2f} USD liên hệ giả định)")

def unit_value(tau, X, margin=MARGIN):
    """Giá trị kỳ vọng của việc gửi coupon: lợi nhuận gộp từ chi tiêu tăng thêm − chi phí coupon − chi phí liên hệ."""
    return margin * tau - cost_hat(X)
v_va_hat = unit_value(tau_va, X_va)
print(f"Điểm break-even: cần tác động ≥ {cost_hat(X_va).mean() / MARGIN:.1f} USD/hộ-đợt để một lần gửi có lãi (ATE ước lượng {th_sup:.1f} USD)")
print(f"Giá trị kỳ vọng mỗi lần gửi (Validation, trong miền): trung bình {v_va_hat[sv].mean():.2f} USD | "
      f"{(v_va_hat[sv] > 0).mean():.1%} hộ có giá trị dương")

CAUSAL_ARTIFACTS = {"tau_model": tau_model, "bag_models": bag_models, "cost_models": (c1, c0), "TAU_CFG": TAU_CFG,
                    "nuisance_note": "e, m, mu1, mu0 huấn luyện trên toàn bộ Train"}

# %% [markdown]
# # GIAI ĐOẠN 5. Cấu trúc hóa bài toán quyết định: Predict-Then-Optimize và SPO+
#
# Quy trình hai bước truyền thống huấn luyện mô hình để *giảm sai số dự báo* (MSE), rồi mới đưa dự báo vào bộ tối ưu. Nhưng bộ tối ưu chỉ quan tâm *thứ tự / lựa chọn* – một sai số nhỏ ở sai chỗ có thể làm hỏng quyết định.
# **SPO+ (Smart Predict-then-Optimize+)** huấn luyện mô hình trực tiếp theo *tổn thất quyết định*.
#
# **Bài toán quyết định mỗi đợt (đơn vị nhỏ = một "khối" 100 hộ chia ngẫu nhiên trong đợt, mỗi khối có ngân sách tối đa K lượt gửi):**
#
# max Σᵢ vᵢ·wᵢ  s.t.  Σᵢ wᵢ ≤ K, wᵢ ∈ {0,1},   với  vᵢ = MARGIN·τᵢ − cᵢ  (giá trị ròng của việc gửi coupon cho hộ i).
#
# **Nhãn quyết định:** giá trị thật vᵢ không quan sát được (mỗi hộ chỉ có một trong hai kết cục). Ta dùng *điểm doubly-robust* Γᵢ (không chệch về kỳ vọng, nhưng rất nhiễu) làm nhãn đại diện:
# vᵢ^nhãn = MARGIN·Γᵢ − cᵢ. Mọi so sánh chính sách đều đi kèm khoảng tin cậy bootstrap theo hộ để không diễn giải nhiễu thành tín hiệu.

# %%
# ---------------------------------------------------------------------------
# 5.1 Dữ liệu quyết định, tiện ích khối và đánh giá chính sách
# ---------------------------------------------------------------------------
from sklearn.preprocessing import QuantileTransformer

KSHARE, BLOCK = 0.25, 100
K_BLOCK = int(round(KSHARE * BLOCK))

COST_tr, COST_va = cost_hat(X_tr), cost_hat(X_va)
VLAB_tr = MARGIN * GAM_tr - COST_tr                    # nhãn giá trị (DR) – Train
VLAB_va = MARGIN * GAM_va - COST_va                    # nhãn giá trị (DR) – Validation (chỉ để ĐÁNH GIÁ)

qt = QuantileTransformer(output_distribution="normal", n_quantiles=300, subsample=200_000, random_state=SEED)
def design(X, cost):
    Z = qt.transform(X[FEATURES]) if hasattr(qt, "quantiles_") else None
    return np.column_stack([np.clip(Z, -4, 4), (cost - COST_tr.mean()) / (COST_tr.std() + 1e-9)])
qt.fit(X_tr[FEATURES])
Z_tr, Z_va = design(X_tr, COST_tr), design(X_va, COST_va)
Zb_tr, Zb_va = np.column_stack([np.ones(len(Z_tr)), Z_tr]), np.column_stack([np.ones(len(Z_va)), Z_va])
LAB_SCALE = float(np.std(VLAB_tr[supp_tr]))

def make_blocks(unit_idx, waves, rng, block=BLOCK):
    """Chia ngẫu nhiên các đơn vị trong CÙNG đợt thành các khối kích thước `block` (mỗi khối = 1 bài toán quyết định)."""
    out = []
    for w in np.unique(waves[unit_idx]):
        u = unit_idx[waves[unit_idx] == w]
        u = rng.permutation(u)
        nb = len(u) // block
        if nb:
            out.append(u[: nb * block].reshape(nb, block))
    return np.vstack(out)

def select_rows(S, k=K_BLOCK, positive_only=True):
    """Nghiệm chính xác của bài toán 'chọn tối đa k phần tử có điểm lớn nhất' (ràng buộc cardinality)."""
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    W = np.zeros(S.shape, dtype=bool)
    np.put_along_axis(W, top, True, axis=1)
    if positive_only:
        W &= S > 0
    return W

def eval_policy(score_all, vlab, groups, waves, unit_idx, positive_only=True, k=K_BLOCK, reps=5, seed=SEED, return_sel=False):
    """Giá trị DR trung bình trên mỗi hộ đủ điều kiện (USD/hộ-đợt) + CI bootstrap theo hộ"""
    rng = np.random.default_rng(seed)
    vals, sent, ids = [], [], []
    for _ in range(reps):
        B = make_blocks(unit_idx, waves, rng)
        S, V = score_all[B], vlab[B]
        W = select_rows(S, k, positive_only)
        vals.append((W * V).ravel()); sent.append(W.ravel()); ids.append(groups[B].ravel())
    v, s, g = map(np.concatenate, (vals, sent, ids))
    mean, lo, hi = boot_by_cluster(v, g, B=300, seed=seed)
    res = {"share_gửi": s.mean(), "giá trị TB (USD/hộ)": mean, "CI thấp": lo, "CI cao": hi}
    return (res, (v, s, g)) if return_sel else res

wave_va = VA.wave.values; wave_tr = TR.wave.values
idx_tr_sup = np.where(supp_tr)[0]
idx_va_sup = np.where(supp_va)[0]
print(f"Khối quyết định: {BLOCK} hộ/khối, K={K_BLOCK} lượt/khối ({KSHARE:.0%}); Train có {len(idx_tr_sup):,} hộ đủ điều kiện, Validation {len(idx_va_sup):,}")
print(f"Nhãn giá trị: trung bình {VLAB_va[supp_va].mean():.2f} USD, độ lệch chuẩn {VLAB_va[supp_va].std():.1f} USD (rất nhiễu) | chi phí TB/lượt {COST_va.mean():.2f} USD")

# %%
# ---------------------------------------------------------------------------
# 5.2 Huấn luyện SPO+ (mô hình tuyến tính) bằng đạo hàm dưới + Adam; so với huấn luyện hai bước (Ridge-MSE)
# ---------------------------------------------------------------------------
def spo_plus_loss_grad(beta, Zb, v_scaled, blocks, k=K_BLOCK):
    """Tổn thất SPO+ trung bình trên các khối và đạo hàm dưới theo β.
    ℓ = max_w (2v̂ − v)ᵀw − 2 v̂ᵀw* + vᵀw*  ;  ∇_v̂ ℓ = 2 (w̃ − w*),  w̃ = argmax_w (2v̂ − v)ᵀw,  w* = argmax_w vᵀw."""
    Zbk, V = Zb[blocks], v_scaled[blocks]
    Vh = Zbk @ beta
    Wstar = select_rows(V, k, True)
    Wtil = select_rows(2 * Vh - V, k, True)
    loss = ((2 * Vh - V) * Wtil).sum(1) - 2 * (Vh * Wstar).sum(1) + (V * Wstar).sum(1)
    gv = 2.0 * (Wtil.astype(float) - Wstar.astype(float))
    grad = np.einsum("bn,bnp->p", gv, Zbk) / len(blocks)
    return loss.mean(), grad

def fit_spo_plus(Zb, v_scaled, unit_idx, waves, l2=1e-2, lr=0.02, epochs=80, batch=48, seed=SEED, beta0=None):
    rng = np.random.default_rng(seed)
    p = Zb.shape[1]
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    m, v_ = np.zeros(p), np.zeros(p)
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    hist = []
    for ep in range(epochs):
        B = make_blocks(unit_idx, waves, rng)
        order = rng.permutation(len(B))
        tot = 0.0
        for s in range(0, len(B), batch):
            blk = B[order[s: s + batch]]
            loss, g = spo_plus_loss_grad(beta, Zb, v_scaled, blk)
            g = g + l2 * np.r_[0.0, beta[1:]]
            step += 1
            m = b1 * m + (1 - b1) * g; v_ = b2 * v_ + (1 - b2) * g * g
            beta -= lr * (m / (1 - b1 ** step)) / (np.sqrt(v_ / (1 - b2 ** step)) + eps)
            tot += loss * len(blk)
        hist.append(tot / len(B))
    return beta, hist

vs_tr = VLAB_tr / LAB_SCALE
inner_fit_idx = idx_tr_sup[wave_tr[idx_tr_sup] <= 5]
inner_ev_idx = idx_tr_sup[wave_tr[idx_tr_sup] >= 6]

# Huấn luyện hai bước: Ridge trên nhãn giá trị (MSE) – chọn α bằng holdout nội bộ theo thời gian
two_stage = {}
for alpha in [1e2, 1e3, 1e4, 1e5]:
    rg = Ridge(alpha=alpha).fit(Z_tr[inner_fit_idx], vs_tr[inner_fit_idx])
    sc = np.full(len(Z_tr), -1.0); sc[inner_ev_idx] = rg.predict(Z_tr[inner_ev_idx])
    r = eval_policy(sc, VLAB_tr, g_tr, wave_tr, inner_ev_idx, positive_only=True, reps=3)
    two_stage[alpha] = r["giá trị TB (USD/hộ)"]
ALPHA_2S = max(two_stage, key=two_stage.get)
print("Hai bước – giá trị holdout nội bộ theo α:", {k: round(v, 3) for k, v in two_stage.items()}, "→ chọn α =", ALPHA_2S)

spo_grid, spo_res = {}, {}
for l2 in [1e-3, 1e-2, 1e-1]:
    beta, h = fit_spo_plus(Zb_tr, vs_tr, inner_fit_idx, wave_tr, l2=l2, epochs=50)
    sc = np.full(len(Zb_tr), -1.0); sc[inner_ev_idx] = Zb_tr[inner_ev_idx] @ beta
    spo_res[l2] = eval_policy(sc, VLAB_tr, g_tr, wave_tr, inner_ev_idx, positive_only=True, reps=3)["giá trị TB (USD/hộ)"]
L2_SPO = max(spo_res, key=spo_res.get)
print("SPO+ – giá trị holdout nội bộ theo λ:", {k: round(v, 3) for k, v in spo_res.items()}, "→ chọn λ =", L2_SPO)

# Huấn luyện lại trên toàn Train
rg_final = Ridge(alpha=ALPHA_2S).fit(Z_tr[idx_tr_sup], vs_tr[idx_tr_sup])
beta_spo, spo_hist = fit_spo_plus(Zb_tr, vs_tr, idx_tr_sup, wave_tr, l2=L2_SPO, epochs=80)
plt.figure(figsize=(5.5, 3.4)); plt.plot(spo_hist); plt.xlabel("Epoch"); plt.ylabel("Tổn thất SPO+ trung bình"); plt.title("Hội tụ huấn luyện SPO+")
save_fig("fig_5_spo_training")

# %%
# ---------------------------------------------------------------------------
# 5.3 Đánh giá theo GIÁ TRỊ QUYẾT ĐỊNH trên Validation (không phải theo MSE)
# ---------------------------------------------------------------------------
tau_sel_va = tau_va                                   # CATE được chọn (có thể là hằng số)
scores_pol = {
    "Ngẫu nhiên (lấp đầy ngân sách)": (np.random.default_rng(SEED).random(len(VA)), False),
    "Heuristic RFM: chi tiêu 90 ngày cao nhất": (VA.spend_90.values, False),
    "Heuristic: rủi ro rời bỏ cao nhất": (-CALIBRATOR(clf_final.predict_proba(X_va)[:, 1]), False),
    "Giải tích: MARGIN·CATE − chi phí (hai bước)": (MARGIN * tau_sel_va - COST_va, True),
    "Giải tích UCB: MARGIN·(CATE+κ·sd) − chi phí": (MARGIN * (lambda tm, ts: tm + 1.28 * ts)(*tau_predict(X_va)) - COST_va, True),
    "Hai bước: Ridge-MSE trên nhãn giá trị": (rg_final.predict(Z_va) * LAB_SCALE, True),
    "SPO+ tuyến tính (decision-focused)": ((Zb_va @ beta_spo) * LAB_SCALE, True),
}
pol_rows = {}
for nm, (sc, pos) in scores_pol.items():
    pol_rows[nm] = eval_policy(sc, VLAB_va, g_va, wave_va, idx_va_sup, positive_only=pos)
# chính sách lịch sử (thực tế doanh nghiệp đã làm) và các mốc tham chiếu – không bị chặn ngân sách
hist_v = (VA["T"].values * VLAB_va)[idx_va_sup]
mh, lh, hh_ = boot_by_cluster(hist_v, g_va[idx_va_sup], B=300)
pol_rows["[Tham chiếu] Chính sách lịch sử (T thực tế)"] = {"share_gửi": float(VA["T"].values[idx_va_sup].mean()), "giá trị TB (USD/hộ)": mh, "CI thấp": lh, "CI cao": hh_}
pol_rows["[Tham chiếu] Không gửi cho ai"] = {"share_gửi": 0.0, "giá trị TB (USD/hộ)": 0.0, "CI thấp": 0.0, "CI cao": 0.0}
pol_df = pd.DataFrame(pol_rows).T
pol_df["regret so với chính sách tốt nhất trong nhóm (USD/hộ)"] = pol_df["giá trị TB (USD/hộ)"].max() - pol_df["giá trị TB (USD/hộ)"]
pol_df = pol_df.round(3)
show_df(pol_df, "So sánh chính sách trên Validation: giá trị DR ròng (USD/hộ-đợt) – khoảng tin cậy bootstrap theo hộ")
save_table(pol_df, "policy_comparison_validation")

# MSE vs giá trị quyết định: sai số dự báo nhỏ chưa chắc cho quyết định tốt
mse_2s = float(np.mean((rg_final.predict(Z_va[idx_va_sup]) * LAB_SCALE - VLAB_va[idx_va_sup]) ** 2))
mse_spo = float(np.mean(((Zb_va[idx_va_sup] @ beta_spo) * LAB_SCALE - VLAB_va[idx_va_sup]) ** 2))
print(f"MSE trên nhãn giá trị – hai bước: {mse_2s:,.1f} | SPO+: {mse_spo:,.1f}  (SPO+ không tối ưu MSE; điều quan trọng là giá trị quyết định ở bảng trên)")

# Độ nhạy theo MARGIN (giả định lợi nhuận gộp): chính sách nào còn dương?
sens = []
for mg in [0.15, 0.25, 0.35]:
    vl = mg * GAM_va - COST_va
    for nm in ["Hai bước: Ridge-MSE trên nhãn giá trị", "SPO+ tuyến tính (decision-focused)", "Giải tích: MARGIN·CATE − chi phí (hai bước)"]:
        sc, pos = scores_pol[nm]
        if nm.startswith("Giải tích"):
            sc = mg * tau_sel_va - COST_va
        r = eval_policy(sc, vl, g_va, wave_va, idx_va_sup, positive_only=pos, reps=3)
        sens.append({"MARGIN": mg, "chính sách": nm, "giá trị TB": r["giá trị TB (USD/hộ)"], "share_gửi": r["share_gửi"]})
sens_df = pd.DataFrame(sens).pivot(index="chính sách", columns="MARGIN", values="giá trị TB").round(3)
show_df(sens_df, "Độ nhạy giá trị chính sách theo tỷ suất lợi nhuận gộp giả định")
save_table(sens_df, "policy_margin_sensitivity")

# %%
# ---------------------------------------------------------------------------
# 5.4 Chọn checkpoint và quy tắc đưa vào vận hành (đăng ký mô hình quyết định)
# ---------------------------------------------------------------------------
# Quy tắc chọn checkpoint (khai báo trước): chỉ thay chính sách giải tích mặc định bằng mô hình học (hai bước / SPO+) nếu CẬN DƯỚI CI95 của
# giá trị DR trên Validation > 0 (có bằng chứng thống kê sinh lời). Nếu không, giữ quy tắc giải tích thận trọng (UCB) – ưu tiên đơn giản khi chưa có bằng chứng.
LEARNED = ["Hai bước: Ridge-MSE trên nhãn giá trị", "SPO+ tuyến tính (decision-focused)"]
passed = [n for n in LEARNED if pol_df.loc[n, "CI thấp"] > 0]
CHECKPOINT_NAME = max(passed, key=lambda n: pol_df.loc[n, "giá trị TB (USD/hộ)"]) if passed else "Giải tích: MARGIN·CATE − chi phí (hai bước)"
print("Mô hình học có CI95 thấp > 0:", passed if passed else "không có")
print("Checkpoint quyết định được chọn:", CHECKPOINT_NAME, "" if passed else "(giữ quy tắc giải tích UCB – SPO+/hai bước chưa chứng minh được giá trị vượt 0)")
VALUE_MODELS = {"two_stage_ridge": rg_final, "spo_beta": beta_spo, "qt": qt, "lab_scale": LAB_SCALE, "checkpoint": CHECKPOINT_NAME}
MODEL_REGISTRY["decision_value"] = {"algo": "SPO+ linear / Ridge / analytic", "checkpoint": CHECKPOINT_NAME, "l2_spo": L2_SPO, "alpha_2stage": ALPHA_2S,
                                    "block": BLOCK, "k_share": KSHARE, "margin": MARGIN, "contact_cost": CONTACT_COST}

# %% [markdown]
# # GIAI ĐOẠN 6. Tối ưu hóa phân bổ nguồn lực dưới ràng buộc (MILP)
#
# Với mỗi đợt, quản lý phải chọn tập hộ nhận coupon sao cho tổng *giá trị kỳ vọng* lớn nhất mà vẫn tuân thủ ràng buộc vận hành. Bài toán là quy hoạch nguyên hỗn hợp (ở đây toàn biến nhị phân):
#
# max Σᵢ uᵢxᵢ  s.t.
# (1) ngân sách: Σᵢ cᵢxᵢ ≤ B (B = 3% tổng chi phí nếu gửi cho mọi hộ đủ điều kiện)  (2) số lượt gửi: Σᵢ xᵢ ≤ N  (3) trần theo phân khúc chi tiêu: mỗi tầng ≤ 50% số lượt  (4) ít nhất 10% số lượt gửi dành cho nhóm rủi ro rời bỏ: Σ_{i∈risk} xᵢ ≥ 0,10·Σᵢ xᵢ (không ép phải gửi)  (5) chống mệt mỏi: xᵢ = 0 nếu vừa bị nhắm dày.
#
# **Điểm số uᵢ:** theo nguyên tắc *lạc quan trước bất định* (UCB) – ưu tiên hộ mà giá trị kỳ vọng cộng κ độ lệch chuẩn bất định vẫn dương. Khi dữ liệu chưa đủ bằng chứng (CI của tác động chứa 0),
# phần ngân sách được cấp là **ngân sách học (EXPLORE)** có giới hạn chứ không phải cam kết lãi. Bộ giải: HiGHS (`scipy.optimize.milp`).

# %%
# ---------------------------------------------------------------------------
# 6.1 Điểm giá trị, cận bất định và công thức MILP
# ---------------------------------------------------------------------------
from scipy.optimize import milp, LinearConstraint, Bounds, linprog
KAPPA = 1.28                      # hệ số lạc quan (≈ cận trên 90% một phía)
TOP_CAP, RISK_P = 0.50, 0.80      # trần mỗi tầng chi tiêu; ngưỡng 'rủi ro rời bỏ' (p_active < 0,8)
BUDGET_MULT = 1.0                 # nhân hệ số ngân sách (dùng cho phân tích độ nhạy)
BUDGET_FRAC = 0.03                # ngân sách mỗi đợt = 3% tổng chi phí nếu gửi cho MỌI hộ đủ điều kiện (đủ chặt để ràng buộc ngân sách thực sự có tác dụng)

def value_bounds(X, tau_mean=None, tau_sd=None, margin=MARGIN):
    """Giá trị kỳ vọng và cận dưới/trên của một lần gửi coupon cho từng hộ."""
    if tau_mean is None:
        tau_mean, tau_sd = tau_predict(X)
    cst = cost_hat(X)
    return {"ev": margin * tau_mean - cst, "lcb": margin * (tau_mean - KAPPA * tau_sd) - cst,
            "ucb": margin * (tau_mean + KAPPA * tau_sd) - cst, "cost": cst, "tau": tau_mean, "tau_sd": tau_sd}

def fatigue_block(X):
    """Quy tắc chống mệt mỏi: không gửi nếu đợt trước đã bị nhắm VÀ 90 ngày qua đã bị nhắm ≥ 3 lần."""
    return ((X.targ_prev28 >= 1) & (X.targ_prev90 >= 3)).values

def spend_tier(X):
    q = np.quantile(X_tr.spend_90, [1 / 3, 2 / 3])
    return np.digitize(X.spend_90.values, q)

def solve_allocation(u, c, tier, risk, blocked, n_max, budget, top_cap=TOP_CAP, risk_share=0.0, time_limit=30, gap=1e-4):
    """MILP chọn tập hộ. Trả về (x, thông tin bộ giải)."""
    n = len(u)
    A, lo, hi = [], [], []
    A.append(c); lo.append(-np.inf); hi.append(budget)                                  # (1) ngân sách
    A.append(np.ones(n)); lo.append(-np.inf); hi.append(n_max)                          # (2) số lượt gửi
    for s in np.unique(tier):                                                           # (3) trần theo tầng
        A.append((tier == s).astype(float)); lo.append(-np.inf); hi.append(np.floor(top_cap * n_max))
    if risk_share > 0:
        A.append(risk.astype(float) - risk_share); lo.append(0.0); hi.append(np.inf)    # (4) ≥ risk_share số lượt gửi dành cho nhóm rủi ro (không ép gửi)
    ub = np.where(blocked, 0.0, 1.0)                                                    # (5) chống mệt mỏi
    t0 = time.perf_counter()
    res = milp(c=-u, constraints=LinearConstraint(sparse.csr_matrix(np.vstack(A)), lo, hi),
               integrality=np.ones(n), bounds=Bounds(np.zeros(n), ub),
               options={"time_limit": time_limit, "mip_rel_gap": gap, "presolve": True})
    el = time.perf_counter() - t0
    x = np.round(res.x).astype(int) if res.x is not None else np.zeros(n, int)
    return x, {"status": res.status, "ok": res.success, "obj": float(u @ x), "seconds": el,
               "bound": float(-res.mip_dual_bound) if getattr(res, "mip_dual_bound", None) is not None else np.nan,
               "gap": float(res.mip_gap) if getattr(res, "mip_gap", None) is not None else np.nan}

def greedy_allocation(u, c, blocked, n_max, budget):
    """Tham lam: sắp theo giá trị trên chi phí, bỏ qua ràng buộc tầng/nhóm rủi ro (đối sánh)."""
    order = np.argsort(-(u / np.maximum(c, 1e-6)))
    x = np.zeros(len(u), int); spent, cnt = 0.0, 0
    for i in order:
        if u[i] <= 0 or blocked[i]:
            continue
        if cnt + 1 <= n_max and spent + c[i] <= budget:
            x[i] = 1; spent += c[i]; cnt += 1
    return x

def lp_bound(u, c, blocked, n_max, budget):
    """Cận trên từ LP nới lỏng (chỉ ngân sách + số lượt) – dùng đo khoảng cách tối ưu của tham lam."""
    n = len(u); ub = np.where(blocked, 0.0, 1.0)
    r = linprog(-u, A_ub=np.vstack([c, np.ones(n)]), b_ub=[budget, n_max], bounds=list(zip(np.zeros(n), ub)), method="highs")
    return float(-r.fun)

# %%
# ---------------------------------------------------------------------------
# 6.2 Giải từng đợt Validation: MILP so với tham lam và top-K không ràng buộc; khoảng cách tối ưu
# ---------------------------------------------------------------------------
bnd_va = value_bounds(X_va)
pa_va = CALIBRATOR(clf_final.predict_proba(X_va)[:, 1])
tier_va, risk_va, block_va = spend_tier(X_va), pa_va < RISK_P, fatigue_block(X_va)
print(f"Cận bất định CATE (Validation): τ̂ TB = {bnd_va['tau'][supp_va].mean():.2f}, sd = {bnd_va['tau_sd'][supp_va].mean():.2f} USD | "
      f"EV>0: {(bnd_va['ev'][supp_va] > 0).mean():.1%}, LCB>0: {(bnd_va['lcb'][supp_va] > 0).mean():.1%}, UCB>0: {(bnd_va['ucb'][supp_va] > 0).mean():.1%}")

def run_waves(df, X, bnd, supp, tier, risk, blocked, n_share=0.25, budget_mult=BUDGET_MULT, top_cap=TOP_CAP, use="ucb", risk_share=0.10, time_limit=30):
    out_x = np.zeros(len(df), int); recs = []
    for w in sorted(df.wave.unique()):
        ix = np.where((df.wave.values == w) & supp)[0]
        u, c = bnd[use][ix], bnd["cost"][ix]
        n_max = int(np.floor(n_share * len(ix)))
        budget = budget_mult * BUDGET_FRAC * c.sum()
        x, info = solve_allocation(u, c, tier[ix], risk[ix], blocked[ix], n_max, budget, top_cap=top_cap, risk_share=risk_share, time_limit=time_limit)
        xg = greedy_allocation(u, c, blocked[ix], n_max, budget)
        ub_lp = lp_bound(u, c, blocked[ix], n_max, budget)
        u_k = np.where(blocked[ix], -np.inf, u); topk = np.zeros(len(ix), int); topk[np.argsort(-u_k)[:n_max]] = 1; topk[u_k <= 0] = 0   # loại hộ bị chặn TRƯỚC khi chọn top-K (công bằng với MILP)
        out_x[ix] = x
        recs.append({"đợt": w, "n hộ đủ ĐK": len(ix), "N tối đa": n_max, "MILP: số gửi": int(x.sum()), "MILP obj": info["obj"], "tham lam obj": float(u @ xg),
                     "top-K không ràng buộc obj": float(u @ topk), "cận LP": ub_lp,
                     "gap MILP vs LP (%)": 100 * (ub_lp - info["obj"]) / max(abs(ub_lp), 1e-3), "gap tham lam vs MILP (%)": 100 * (info["obj"] - float(u @ xg)) / max(abs(info["obj"]), 1e-3),
                     "giải (s)": info["seconds"], "optimal": info["status"] == 0, "chi phí dùng": float(c @ x), "ngân sách": budget})
    return out_x, pd.DataFrame(recs).set_index("đợt")

x_milp_va, milp_tab = run_waves(VA, X_va, bnd_va, supp_va, tier_va, risk_va, block_va)
show_df(milp_tab.round(3), "MILP theo đợt – Validation (điểm số = UCB)")
save_table(milp_tab, "milp_waves_validation")
print("Ràng buộc ngân sách có hiệu lực (chi phí dùng ≥ 99% ngân sách) ở", int((milp_tab["chi phí dùng"] >= 0.99 * milp_tab["ngân sách"]).sum()), "/", len(milp_tab), "đợt;",
      "MILP hơn tham lam ở", int((milp_tab["gap tham lam vs MILP (%)"] > 1e-6).sum()), "đợt (gap TB %.3f%%)" % milp_tab["gap tham lam vs MILP (%)"].mean())

# kiểm tra ràng buộc thực sự được thỏa
chk = []
for w in sorted(VA.wave.unique()):
    ix = np.where((VA.wave.values == w) & supp_va)[0]; x = x_milp_va[ix]
    chk.append({"đợt": w, "ngân sách OK": float(bnd_va["cost"][ix] @ x) <= milp_tab.loc[w, "ngân sách"] + 1e-6,
                "N OK": x.sum() <= milp_tab.loc[w, "N tối đa"], "tầng max (≤50%)": max((x[tier_va[ix] == s].sum() for s in np.unique(tier_va[ix])), default=0) / max(x.sum(), 1),
                "vi phạm chống mệt mỏi": int((x[block_va[ix]]).sum())})
show_df(pd.DataFrame(chk).set_index("đợt").round(3), "Kiểm tra thỏa ràng buộc")

# %%
# ---------------------------------------------------------------------------
# 6.3 Giá trị thực hiện (DR) của chính sách MILP và độ ổn định / độ nhạy ràng buộc
# ---------------------------------------------------------------------------
def dr_policy_value(x_sel, vlab, groups, mask):
    v = (x_sel * vlab)[mask]
    m, lo, hi = boot_by_cluster(v, groups[mask], B=400)
    return m, lo, hi

rows = []
for nm, xs in [("MILP (UCB, đủ ràng buộc)", x_milp_va), ("Không gửi", np.zeros(len(VA), int)), ("Chính sách lịch sử (T thực tế)", VA["T"].values * supp_va)]:
    m, lo, hi = dr_policy_value(xs, VLAB_va, g_va, supp_va)
    cost_spent = float((xs * COST_va)[supp_va].mean())
    rows.append({"chính sách": nm, "share gửi": xs[supp_va].mean(), "giá trị ròng TB (USD/hộ)": m, "CI thấp": lo, "CI cao": hi, "chi phí coupon TB (USD/hộ)": cost_spent})
milp_val_df = pd.DataFrame(rows).set_index("chính sách").round(3)
show_df(milp_val_df, "Giá trị DR của chính sách MILP trên Validation")
save_table(milp_val_df, "milp_policy_value_validation")

# độ nhạy theo ngân sách và theo trần phân khúc (cách giá trị / chi phí thay đổi)
sens = []
for bm in [0.5, 1.0, 1.5]:
    for tc in [0.40, 0.50, 0.70]:
        xs, tab = run_waves(VA, X_va, bnd_va, supp_va, tier_va, risk_va, block_va, budget_mult=bm, top_cap=tc, time_limit=15)
        m, lo, hi = dr_policy_value(xs, VLAB_va, g_va, supp_va)
        sens.append({"nhân ngân sách": bm, "trần tầng": tc, "obj UCB": tab["MILP obj"].sum(), "giá trị DR": m, "số gửi": int(xs.sum()), "chi phí": float((xs * COST_va).sum())})
sens_df = pd.DataFrame(sens).round(3)
show_df(sens_df, "Độ nhạy MILP theo ngân sách và trần phân khúc (mục tiêu UCB vs giá trị DR thực hiện)")
save_table(sens_df, "milp_sensitivity")

# độ ổn định chính sách: chồng lấn tập chọn giữa hai hạt giống bootstrap của mô hình CATE (Jaccard)
alt_models = [_fit_tau(boot_rows_support(r_bag), SEED + 900 + b) for b in range(4)]
jac = []
for am in alt_models[:3]:
    tm = am.predict(X_va)
    b2 = value_bounds(X_va, tm, bnd_va["tau_sd"])
    x2, _ = run_waves(VA, X_va, b2, supp_va, tier_va, risk_va, block_va, time_limit=15)
    a, b = set(np.where(x_milp_va)[0]), set(np.where(x2)[0])
    jac.append(len(a & b) / max(len(a | b), 1))
print(f"Độ ổn định chính sách MILP giữa các mô hình bootstrap (Jaccard tập được chọn): {np.round(jac, 3)} – trung bình {np.mean(jac):.3f}")

# %%
# ---------------------------------------------------------------------------
# 6.4 Chi phí tính toán của bộ giải theo quy mô bài toán
# ---------------------------------------------------------------------------
scal = []
r_sc = np.random.default_rng(SEED + 5)
ix_pool = np.where(supp_va)[0]
for n in [500, 1000, 2000, 4000, 8000, 16000]:
    ix = r_sc.choice(ix_pool, n, replace=True)
    u, c = bnd_va["ucb"][ix], bnd_va["cost"][ix]
    n_max, budget = int(0.25 * n), BUDGET_FRAC * c.sum()
    x, info = solve_allocation(u, c, tier_va[ix], risk_va[ix], block_va[ix], n_max, budget, risk_share=0.10, time_limit=60)
    t0 = time.perf_counter(); greedy_allocation(u, c, block_va[ix], n_max, budget); tg = time.perf_counter() - t0
    scal.append({"quy mô n": n, "MILP (s)": info["seconds"], "tham lam (s)": tg, "optimal": info["status"] == 0, "obj": info["obj"]})
scal_df = pd.DataFrame(scal).set_index("quy mô n").round(4)
show_df(scal_df, "Thời gian giải theo quy mô (dữ liệu Validation có lặp mẫu)")
save_table(scal_df, "milp_scaling")
plt.figure(figsize=(5.5, 3.5)); plt.loglog(scal_df.index, scal_df["MILP (s)"], "o-", label="MILP (HiGHS)"); plt.loglog(scal_df.index, scal_df["tham lam (s)"].clip(lower=1e-5), "s--", label="Tham lam")
plt.xlabel("Số hộ trong bài toán"); plt.ylabel("Giây"); plt.legend(); plt.title("Chi phí tính toán của bộ giải")
save_fig("fig_6_solver_scaling")

# %% [markdown]
# # GIAI ĐOẠN 7. Định tuyến hành động, quy tắc nghiệp vụ (Guardrails) và con người tham gia (HITL)
#
# Điểm quyết định (giá trị kỳ vọng, cận bất định, xác suất quay lại) chưa phải *hành động*. Action Router chuyển điểm thành một trong **5 hành động** và ghi lý do:
#
# | Hành động | Điều kiện | Ý nghĩa vận hành |
# |---|---|---|
# | `SEND` | Được MILP chọn (có ràng buộc ngân sách) **và** cận dưới giá trị (LCB) > 0 | Có bằng chứng sinh lời → gửi coupon |
# | `EXPLORE` | Hộ đủ điều kiện, không thuộc SEND; gửi theo **đồng xu độc lập p = 5%** (không phụ thuộc chi phí; ghi lại propensity) | Ngân sách *học*: tạo biến thiên ngẫu nhiên để khắc phục khuyết nhãn chọn lọc |
# | `NO_ACTION` | Không được chọn / ngoài miền positivity / nhóm đối chứng của EXPLORE | Mặc định an toàn |
# | `SUPPRESS` | Quy tắc chống mệt mỏi (vừa bị nhắm dày) | Không liên hệ |
# | `HITL` | Bất thường phân phối (Isolation Forest) hoặc hộ giá trị rất cao | Chuyển người duyệt, có trần công suất |
#
# **Thứ tự ưu tiên rào chắn:** positivity → chống mệt mỏi → bất thường/giá trị cao → tối ưu MILP → phân loại SEND/EXPLORE.

# %%
# ---------------------------------------------------------------------------
# 7.1 Chuẩn bị các thành phần của Router (mô hình propensity, phát hiện bất thường, ngưỡng)
# ---------------------------------------------------------------------------
PROP_MODEL = hgb_clf(**NUIS_CLF).fit(X_tr, t_tr)                         # propensity cuối cùng (khớp với e_f ở Giai đoạn 4)
OOD_FEATURES = ["spend_28", "spend_90", "spend_180", "visits_28", "visits_90", "avg_basket_90", "recency",
                "coupon_share_90", "promo_share_90", "private_share_90", "fresh_share_90"]
ood_model = IsolationForest(n_estimators=200, max_samples=2048, random_state=SEED).fit(X_tr[OOD_FEATURES])
OOD_THR = float(np.quantile(ood_model.score_samples(X_tr[OOD_FEATURES]), 0.01))          # 1% bất thường nhất của Train
HV_THR = float(np.quantile(X_tr.spend_90, 0.99))                                          # hộ giá trị rất cao (top 1%)
EXPLORE_P = 0.05                                                                         # xác suất EXPLORE đồng đều trên hộ đủ điều kiện (phủ đều → positivity cho lần học sau)
HITL_CAP = 0.08                                                                           # công suất duyệt: ≤ 8% số hộ mỗi đợt
ROUTER_CFG = {"ps_clip": PS_CLIP, "kappa": KAPPA, "explore_p": EXPLORE_P, "hitl_cap": HITL_CAP, "ood_thr": OOD_THR, "hv_thr": HV_THR,
              "fatigue_rule": "targ_prev28>=1 & targ_prev90>=3", "n_share": 0.25, "budget_frac": BUDGET_FRAC, "top_cap": TOP_CAP, "risk_share_min": 0.10}
print("Cấu hình Router:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ROUTER_CFG.items()}, ensure_ascii=False))

ACTIONS = ["SEND", "EXPLORE", "NO_ACTION", "SUPPRESS", "HITL"]

def action_router(df, X, ood_on=True, seed=SEED, time_limit=15):
    """Chuyển điểm quyết định thành hành động, kèm lý do. df cần cột 'wave'."""
    rng = np.random.default_rng(seed)
    n = len(X)
    e = PROP_MODEL.predict_proba(X)[:, 1]
    insupp = (e >= PS_CLIP[0]) & (e <= PS_CLIP[1])
    fat = fatigue_block(X)
    ood_score = ood_model.score_samples(X[OOD_FEATURES])
    ood = (ood_score < OOD_THR) if ood_on else np.zeros(n, bool)
    hv = (X.spend_90.values > HV_THR) if ood_on else np.zeros(n, bool)
    pa = CALIBRATOR(clf_final.predict_proba(X)[:, 1])
    bnd = value_bounds(X)
    action = np.array(["NO_ACTION"] * n, dtype=object); reason = np.array(["không được chọn"] * n, dtype=object)
    # 1) positivity
    action[~insupp] = "NO_ACTION"; reason[~insupp] = "ngoài miền positivity"
    # 2) chống mệt mỏi
    m = insupp & fat; action[m] = "SUPPRESS"; reason[m] = "chống mệt mỏi"
    # 3) bất thường / giá trị cao → HITL (có trần công suất theo đợt)
    cand = insupp & ~fat & (ood | hv)
    for w in np.unique(df.wave.values):
        ix = np.where(cand & (df.wave.values == w))[0]
        cap = int(np.floor(HITL_CAP * (df.wave.values == w).sum()))
        severity = np.where(hv[ix], 1e3, 0) + (OOD_THR - ood_score[ix])
        keep = ix[np.argsort(-severity)[:cap]]
        over = np.setdiff1d(ix, keep)
        action[keep] = "HITL"; reason[keep] = np.where(hv[keep], "hộ giá trị rất cao", "bất thường phân phối")
        action[over] = "NO_ACTION"; reason[over] = "HITL quá tải → mặc định không gửi"
    # 4) MILP trên tập đủ điều kiện
    elig = insupp & ~fat & (action == "NO_ACTION") & (reason == "không được chọn")
    tier, risk = spend_tier(X), pa < RISK_P
    x_mip, _ = run_waves(df, X, bnd, elig, tier, risk, np.zeros(n, bool), n_share=ROUTER_CFG["n_share"], top_cap=TOP_CAP, use="ucb", time_limit=time_limit)
    sel = x_mip == 1
    send = sel & (bnd["lcb"] > 0)
    waiting = sel & ~send                                  # UCB > 0 nhưng chưa đủ bằng chứng (LCB ≤ 0)
    pool = elig & ~send                                    # mọi hộ đủ điều kiện còn lại → ngân sách học ε
    coin = rng.random(n) < EXPLORE_P
    expl = pool & coin
    action[send] = "SEND"; reason[send] = "LCB giá trị > 0"
    action[expl] = "EXPLORE"; reason[expl] = "ngân sách học (đồng xu = 1)"
    reason[pool & ~coin & waiting] = "UCB>0 nhưng LCB≤0 và không trúng đồng xu EXPLORE"
    reason[pool & ~coin & ~waiting] = "đối chứng của EXPLORE (đồng xu = 0)"
    out = pd.DataFrame({"action": action, "reason": reason, "p_explore": np.where(pool, EXPLORE_P, np.nan), "hitl_candidate": cand, "ev": bnd["ev"], "lcb": bnd["lcb"],
                        "ucb": bnd["ucb"], "cost": bnd["cost"], "p_active": pa, "ood_score": ood_score, "propensity": e}, index=df.index)
    out["gets_coupon"] = out.action.isin(["SEND", "EXPLORE"]).astype(int)
    return out

# %%
# ---------------------------------------------------------------------------
# 7.2 Chạy Router trên Validation: phân bổ hành động, lý do, tải HITL
# ---------------------------------------------------------------------------
dec_va = action_router(VA, X_va)
mix = (dec_va.groupby([VA.wave, "action"]).size().unstack(fill_value=0).reindex(columns=ACTIONS, fill_value=0))
mix_share = mix.div(mix.sum(1), axis=0).round(3)
show_df(mix_share, "Tỷ trọng hành động theo đợt (Validation)")
save_table(mix_share, "router_action_mix_validation")
show_df(dec_va.groupby(["action", "reason"]).size().to_frame("số hộ"), "Lý do ra quyết định (Validation)")

print(f"Nhu cầu HITL TRƯỚC khi áp trần: {dec_va.hitl_candidate.mean():.2%} số hộ (trần công suất {HITL_CAP:.0%})")
hitl_load = dec_va[dec_va.action == "HITL"].groupby(VA.wave).size()
print("Tải HITL theo đợt (số hộ cần duyệt):", hitl_load.to_dict(), f"| trần công suất {HITL_CAP:.0%} số hộ/đợt")

# giá trị DR và chi phí theo hành động (đối chiếu với nhãn DR – chỉ để đánh giá)
act_val = (pd.DataFrame({"action": dec_va.action.values, "v": VLAB_va, "cost": COST_va, "in_supp": supp_va})
           .query("in_supp").groupby("action").agg(n=("v", "size"), nhãn_giá_trị_TB=("v", "mean"), chi_phí_kỳ_vọng_TB=("cost", "mean")).round(3))
show_df(act_val, "Nhãn giá trị DR theo nhóm hành động (rất nhiễu; chỉ minh họa)")

# Chi phí coupon thực hiện của Router so với chính sách lịch sử (cùng tập hộ đủ điều kiện)
cost_router = float((dec_va.gets_coupon.values * COST_va)[supp_va].sum())
cost_hist = float((VA["T"].values * COST_va)[supp_va].sum())
v_router = dr_policy_value(dec_va.gets_coupon.values, VLAB_va, g_va, supp_va)
v_hist = dr_policy_value(VA["T"].values, VLAB_va, g_va, supp_va)
router_cmp = pd.DataFrame({
    "chi phí coupon kỳ vọng (USD)": [cost_router, cost_hist], "số hộ nhận coupon": [int((dec_va.gets_coupon.values * supp_va).sum()), int((VA["T"].values * supp_va).sum())],
    "giá trị DR TB (USD/hộ)": [v_router[0], v_hist[0]], "CI thấp": [v_router[1], v_hist[1]], "CI cao": [v_router[2], v_hist[2]]},
    index=["Router (SEND+EXPLORE)", "Chính sách lịch sử"]).round(3)
show_df(router_cmp, "Router so với chính sách lịch sử trên Validation (trong miền positivity)")
save_table(router_cmp, "router_vs_historical_validation")

# Độ nhạy theo định nghĩa chi phí: nhà bán lẻ chịu ρ phần chiết khấu coupon (ρ=1: toàn bộ, như giả định gốc; ρ=0: chỉ chi phí liên hệ,
# ví dụ khi coupon do nhà sản xuất hoàn tiền). Cần đối chiếu tài liệu dunnhumby để biết ρ thực tế.
rho_rows = []
for rho in [0.0, 0.5, 1.0]:
    cost_rho = CONTACT_COST + rho * np.clip(c1.predict(X_va) - c0.predict(X_va), 0, None)
    vl = MARGIN * GAM_va - cost_rho
    for nm_, xs_ in [("Router (SEND+EXPLORE)", dec_va.gets_coupon.values), ("Chính sách lịch sử", VA["T"].values)]:
        m_, lo_, hi_ = dr_policy_value(xs_, vl, g_va, supp_va)
        rho_rows.append({"ρ (tỷ lệ chiết khấu nhà bán lẻ chịu)": rho, "chính sách": nm_, "giá trị DR TB": m_, "CI thấp": lo_, "CI cao": hi_})
rho_df = pd.DataFrame(rho_rows).round(3)
show_df(rho_df, "Độ nhạy theo định nghĩa chi phí coupon (ρ)")
save_table(rho_df.set_index(["ρ (tỷ lệ chiết khấu nhà bán lẻ chịu)", "chính sách"]), "policy_cost_share_sensitivity")

# %%
# ---------------------------------------------------------------------------
# 7.3 Hiệu quả của rào chắn khi dữ liệu trôi (stress test): tiêm trôi vào đặc trưng lúc suy luận
# ---------------------------------------------------------------------------
def inject_drift(X, lam, seed=SEED):
    """Mô phỏng lỗi/trôi đầu vào: chi tiêu bị nhân (1+λ), recency lệch, thêm nhiễu – chỉ tại thời điểm suy luận."""
    r = np.random.default_rng(seed)
    Xd = X.copy()
    for c in ["spend_28", "spend_prev28", "spend_90", "spend_180", "avg_basket_90"]:
        Xd[c] = X[c] * (1 + lam) * np.exp(r.normal(0, 0.25 * lam, len(X)))
    Xd["visits_28"] = X.visits_28 * (1 + 0.5 * lam); Xd["visits_90"] = X.visits_90 * (1 + 0.5 * lam)
    Xd["recency"] = np.clip(X.recency + 20 * lam * r.random(len(X)), 0, None)
    return Xd

base_act = dec_va.action.values
drift_rows = []
for lam in [0.0, 0.25, 0.5, 1.0, 2.0]:
    Xd = inject_drift(X_va, lam)
    for on in [False, True]:
        d = action_router(VA, Xd, ood_on=on)
        auto = d.action.isin(["SEND", "EXPLORE"]).values
        flips = (d.action.values != base_act) & auto
        drift_rows.append({"độ trôi λ": lam, "rào chắn bất thường": "BẬT" if on else "TẮT",
                           "tỷ lệ gửi tự động": auto.mean(), "tỷ lệ chuyển HITL": (d.action == "HITL").mean(),
                           "tỷ lệ quyết định tự động bị đổi so với không trôi": flips.mean(),
                           "chi phí coupon (USD)": float((auto * COST_va).sum()),
                           "TVD hành động so với không trôi": tvd(pd.Series(base_act).value_counts(normalize=True).reindex(ACTIONS, fill_value=0).values,
                                                                  d.action.value_counts(normalize=True).reindex(ACTIONS, fill_value=0).values)})
drift_df = pd.DataFrame(drift_rows).round(4)
show_df(drift_df, "Stress test trôi dữ liệu: Router có và không có rào chắn bất thường/giá trị cao")
save_table(drift_df, "router_drift_stress")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for on, mk in [("TẮT", "o--"), ("BẬT", "s-")]:
    s_ = drift_df[drift_df["rào chắn bất thường"] == on]
    ax[0].plot(s_["độ trôi λ"], s_["tỷ lệ quyết định tự động bị đổi so với không trôi"], mk, label=f"Rào chắn {on}")
    ax[1].plot(s_["độ trôi λ"], s_["tỷ lệ chuyển HITL"], mk, label=f"Rào chắn {on}")
ax[0].set_xlabel("Độ trôi λ"); ax[0].set_ylabel("Tỷ lệ quyết định tự động bị đổi"); ax[0].legend(); ax[0].set_title("Rào chắn giảm quyết định sai khi dữ liệu trôi")
ax[1].set_xlabel("Độ trôi λ"); ax[1].set_ylabel("Tỷ lệ chuyển HITL"); ax[1].set_title("Cái giá: tải duyệt thủ công"); ax[1].legend()
save_fig("fig_7_guardrail_drift")

# %%
# ---------------------------------------------------------------------------
# 7.4 Playbook ngoại lệ và HITL
# ---------------------------------------------------------------------------
PLAYBOOK = pd.DataFrame([
    ["PSI đặc trưng ≥ 0,25 trong 2 đợt liên tiếp", "Cảnh báo đỏ", "Chuyển 100% sang EXPLORE/NO_ACTION; kích hoạt tái huấn luyện Challenger", "Data Scientist", "24 giờ"],
    ["Nhu cầu HITL (trước trần) vượt 8% số hộ của đợt", "Quá tải duyệt", "Phần vượt mặc định NO_ACTION; điều tra nguồn dữ liệu đầu vào", "Trưởng nhóm CRM", "8 giờ"],
    ["Brier hoặc ECE xấu đi sau khi nhãn về (đợt t+1)", "Suy giảm hiệu chỉnh", "Hiệu chỉnh lại bằng dữ liệu mới; chặn Challenger nếu không qua Release Gate", "Data Scientist", "1 đợt"],
    ["Độ trễ online P95 vượt ngân sách", "Vi phạm SLA", "Hạ về chính sách luật đơn giản (cache điểm số của đợt trước)", "MLOps / Kỹ sư", "4 giờ"],
    ["Bộ giải MILP không tối ưu trong thời hạn", "Solver timeout", "Dùng nghiệm tham lam (đã có) và ghi nhận gap", "MLOps / Kỹ sư", "tức thì"],
    ["Hộ giá trị rất cao hoặc bất thường phân phối", "Ngoại lệ cá nhân", "Người duyệt quyết định thủ công; ghi kết quả làm nhãn", "Chuyên viên CRM", "2 ngày làm việc"],
    ["Giám đốc yêu cầu dừng chương trình", "Kill-switch", "Chuyển toàn bộ sang NO_ACTION; giữ nhóm EXPLORE nhỏ để theo dõi", "Quản lý chương trình", "tức thì"],
], columns=["Sự cố / tín hiệu", "Mức", "Hành động xử lý", "Chủ trì", "SLA"])
show_df(PLAYBOOK, "Playbook ngoại lệ và HITL")
save_table(PLAYBOOK.set_index("Sự cố / tín hiệu"), "playbook")

# %% [markdown]
# # GIAI ĐOẠN 8. Triển khai, đo lường thực chứng và giám sát hệ thống (MLOps)
#
# Quy trình của giai đoạn này:
# 1. **Đóng băng** toàn bộ thành phần (mô hình, hiệu chỉnh, cấu hình Router, cổng phát hành) và tính SHA-256.
# 2. **Khóa**: ghi biên bản khóa có dấu thời gian + mã băm *trước khi* nhìn Locked Test / OOT. Biên bản chỉ ghi một lần, đặt chế độ chỉ đọc, và các lần chạy sau chỉ *kiểm tra* mã băm.
# 3. **Đánh giá thực chứng một lần** trên Locked Test và Out-of-Time: chất lượng dự báo, hiệu chỉnh, giá trị chính sách, regret, độ trễ đầu cuối (P50/P95/P99), độ trôi (PSI, Brier, ECE, Action TVD), độ trễ nhãn.
# 4. **Current vs Challenger** và **Release Gates** (đã khai báo ở mục 0, không chỉnh sau khi thấy kết quả).
#
# Ở `RUN_STAGE="dev"`, các bước 2–3 chạy *thử* trên Validation để kiểm tra mã; Locked/OOT vẫn bị niêm phong.

# %%
# ---------------------------------------------------------------------------
# 8.1 Đóng băng thành phần và mã băm
# ---------------------------------------------------------------------------
import stat
MU1_FINAL = hgb_reg(**NUIS_REG).fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
MU0_FINAL = hgb_reg(**NUIS_REG).fit(X_tr[t_tr == 0], y_tr[t_tr == 0])

MODEL_BUNDLE = {"clf": clf_final, "reg": reg_final, "iso": iso, "platt": platt, "baseline_lr": lr, "propensity": PROP_MODEL,
                "ood": ood_model, "cost_treated": c1, "cost_control": c0, "tau_het": tau_het, "mu1": MU1_FINAL, "mu0": MU0_FINAL}
tau_state = ([float(m.c) for m in bag_models] if TAU_CFG == "hằng số (ATE)" else "sklearn_models_not_serialised_in_json")
FREEZE_JSON = {
    "features": FEATURES, "cal_name": CAL_NAME, "conformal_q": {int(k): float(v) for k, v in conf_q.items()}, "conformal_cut": [float(x) for x in cut],
    "tau_cfg": TAU_CFG, "tau_const_bag": tau_state, "router_cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ROUTER_CFG.items()},
    "release_gates": {k: list(v) for k, v in RELEASE_GATES.items()}, "params": {"LOOKBACK": LOOKBACK, "WAVE_LEN": WAVE_LEN, "MARGIN": MARGIN,
    "CONTACT_COST": CONTACT_COST, "PS_CLIP": PS_CLIP, "SEED": SEED, "Y_CAP": Y_CAP, "KAPPA": KAPPA}, "split_waves": SPLIT_WAVES, "embargo_waves": EMBARGO_WAVES,
    "registry": {k: {kk: vv for kk, vv in v.items() if kk != "features"} for k, v in MODEL_REGISTRY.items()},
}
def model_fingerprint():
    """Dấu vân tay theo DỰ BÁO của mọi mô hình trên một tập dò cố định (Validation) – bền vững hơn băm file pickle khi đổi máy/Colab."""
    parts_ = [clf_final.predict_proba(X_va)[:, 1], reg_final.predict(X_va), PROP_MODEL.predict_proba(X_va)[:, 1], MU1_FINAL.predict(X_va),
              MU0_FINAL.predict(X_va), tau_het.predict(X_va), ood_model.score_samples(X_va[OOD_FEATURES]), c1.predict(X_va), c0.predict(X_va)]
    parts_ += [m.predict(X_va) for m in bag_models]
    return hashlib.sha256(np.round(np.concatenate(parts_), 5).tobytes()).hexdigest()
FINGERPRINT = model_fingerprint()
_seal_file = OUT_DIR / "locked_seal.json"
if RUN_STAGE == "final" and _seal_file.exists():       # đối chiếu TRƯỚC khi ghi bất kỳ tệp nào → không bao giờ ghi đè mô hình đã khóa khi lệch
    _old = json.loads(_seal_file.read_text(encoding="utf-8"))
    if _old.get("sha256 dự báo mô hình (đối chiếu)") != FINGERPRINT:
        raise RuntimeError("Mô hình huấn luyện lại KHÁC mô hình đã khóa (dấu vân tay dự báo lệch) – dừng trước khi ghi đè tệp.")
mpath = OUT_DIR / "models" / "model_bundle.joblib"
joblib.dump(MODEL_BUNDLE, mpath)
jpath = OUT_DIR / "models" / "freeze_config.json"
jpath.write_text(json.dumps(FREEZE_JSON, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
freeze_hashes = {"model_bundle.joblib": sha256_file(mpath), "freeze_config.json": sha256_file(jpath)}
show_df(pd.Series(freeze_hashes).to_frame("SHA-256"), "Mã băm các thành phần đã đóng băng")

# %%
# ---------------------------------------------------------------------------
# 8.2 Hàm đánh giá thực chứng dùng chung (Validation khi dev; Locked & OOT khi final)
# ---------------------------------------------------------------------------
def dr_for(df, X):
    e = PROP_MODEL.predict_proba(X)[:, 1]
    mu1, mu0 = MU1_FINAL.predict(X), MU0_FINAL.predict(X)
    return e, dr_score(df.Y_w.values, df["T"].values, e, mu1, mu0)

OOF_CAL_PRED = CALIBRATOR(oof_p)                                           # phân phối dự báo tham chiếu (Train, out-of-fold)
REF_MIX = dec_va.action.value_counts(normalize=True).reindex(ACTIONS, fill_value=0).values

def evaluate_split(df, name):
    df = df.reset_index(drop=True); X = df[FEATURES]
    y, g = df.Y_active.values, df.household_key.values
    p_raw = clf_final.predict_proba(X)[:, 1]; p = CALIBRATOR(p_raw); p_b = lr.predict_proba(X)[:, 1]
    yhat = np.clip(reg_final.predict(X), 0, None); lo, hi = conformal_interval(yhat)
    e, Gam = dr_for(df, X); sp = (e >= PS_CLIP[0]) & (e <= PS_CLIP[1])
    vlab = MARGIN * Gam - cost_hat(X)
    dec = action_router(df, X)
    got = dec.gets_coupon.values
    v_r = boot_by_cluster((got * vlab)[sp], g[sp], B=400)
    v_h = boot_by_cluster((df["T"].values * vlab)[sp], g[sp], B=400)
    d_ = boot_by_cluster(((got - df["T"].values) * vlab)[sp], g[sp], B=400)
    c_r, c_h = float((got * cost_hat(X))[sp].sum()), float((df["T"].values * cost_hat(X))[sp].sum())
    rfm = eval_policy(df.spend_90.values, vlab, g, df.wave.values, np.where(sp)[0], positive_only=False, reps=3)["giá trị TB (USD/hộ)"]
    cand = {"Không gửi": 0.0, "Lịch sử": v_h[0], "RFM top-25%": rfm, "Router": v_r[0]}
    # PSI tính THEO TỪNG ĐỢT rồi lấy lớn nhất (PSI gộp nhiều đợt làm loãng một đợt bị trôi)
    psi_feats = {c: max(psi(X_tr[c], X[c][df.wave.values == w]) for w in np.unique(df.wave.values)) for c in OOD_FEATURES}
    mix = dec.action.value_counts(normalize=True).reindex(ACTIONS, fill_value=0)
    ate, ate_se = Gam[sp].mean(), cluster_se_mean(Gam[sp], g[sp])
    m = {"split": name, "n": len(df), "n_trong_miền": int(sp.sum()),
         "AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p), "Brier baseline": brier_score_loss(y, p_b),
         "ΔBrier tương đối vs baseline": (brier_score_loss(y, p) - brier_score_loss(y, p_b)) / brier_score_loss(y, p_b),
         "ECE": expected_calibration_error(y, p), "Độ phủ conformal": float(((df.Y_w.values >= lo) & (df.Y_w.values <= hi)).mean()),
         "MAE chi tiêu": mean_absolute_error(df.Y_w, yhat),
         "ATE (DR, trong miền)": ate, "SE ATE": ate_se,
         "Giá trị Router (USD/hộ)": v_r[0], "Giá trị lịch sử (USD/hộ)": v_h[0], "Chênh lệch Router−lịch sử": d_[0], "Chênh lệch CI thấp": d_[1], "Chênh lệch CI cao": d_[2],
         "Chi phí coupon Router (USD)": c_r, "Chi phí coupon lịch sử (USD)": c_h, "Giảm chi phí": 1 - c_r / max(c_h, 1e-9),
         "Regret của Router (USD/hộ)": max(cand.values()) - v_r[0], "Chính sách tốt nhất sau-thực-tế": max(cand, key=cand.get),
         "Tỷ lệ HITL": float(mix["HITL"]), "Nhu cầu HITL (trước trần)": float(dec.hitl_candidate.mean()), "Tỷ lệ gửi (SEND+EXPLORE)": float(mix["SEND"] + mix["EXPLORE"]),
         "PSI đặc trưng max": max(psi_feats.values()), "PSI điểm dự báo": psi(OOF_CAL_PRED, p), "Action TVD vs Validation": tvd(mix.values, REF_MIX)}
    return m, dec, {"psi_feats": psi_feats, "mix": mix, "p": p}

def gate_table(m, p95_ms):
    vals = {"G1_brier_rel": m["ΔBrier tương đối vs baseline"], "G2_ece": m["ECE"], "G3_coverage": m["Độ phủ conformal"],
            "G4_psi": max(m["PSI đặc trưng max"], m["PSI điểm dự báo"]), "G5_p95_ms": p95_ms, "G6_noninf": m["Chênh lệch CI thấp"],
            "G7_cost_cut": m["Giảm chi phí"], "G8_hitl": m["Nhu cầu HITL (trước trần)"]}
    ops = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b}
    rows = [{"cổng": k, "tiêu chí": RELEASE_GATES[k][0], "ngưỡng": RELEASE_GATES[k][2], "giá trị đo": vals[k],
             "kết quả": "ĐẠT" if ops[RELEASE_GATES[k][1]](vals[k], RELEASE_GATES[k][2]) else "KHÔNG ĐẠT"} for k in RELEASE_GATES]
    return pd.DataFrame(rows).set_index("cổng")

# %%
# ---------------------------------------------------------------------------
# 8.3 Độ trễ quyết định đầu cuối đo THẬT (P50 / P95 / P99), tách theo thành phần
# ---------------------------------------------------------------------------
def scalar_rules(row):
    """Luật nghiệp vụ cho 1 hộ (giống Router, không gồm MILP theo lô)."""
    fat = (row.targ_prev28 >= 1) and (row.targ_prev90 >= 3)
    hv = row.spend_90 > HV_THR
    return "SUPPRESS" if fat else ("HITL" if hv else "PASS")

def latency_benchmark(df, n=600, seed=SEED):
    r = np.random.default_rng(seed)
    pick = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    rec = []
    for i in range(len(pick)):
        hi, d0 = int(pick.hi[i]), int(pick.decision_day[i])
        t0 = time.perf_counter()
        f = make_features(np.array([hi]), np.array([d0]))[FEATURES]                          # (1) kéo đặc trưng
        t1 = time.perf_counter()
        pa_ = CALIBRATOR(clf_final.predict_proba(f)[:, 1])                                    # (2) suy luận: dự báo + conformal + CATE + propensity + chi phí + OOD
        yh = np.clip(reg_final.predict(f), 0, None); conformal_interval(yh)
        tm, ts = tau_predict(f); pr = PROP_MODEL.predict_proba(f)[:, 1]; cst = cost_hat(f); osc = ood_model.score_samples(f[OOD_FEATURES])
        t2 = time.perf_counter()
        scalar_rules(f.iloc[0])                                                               # (3) luật nghiệp vụ / rào chắn
        t3 = time.perf_counter()
        rec.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3))
    lat = pd.DataFrame(rec, columns=["feature_ms", "inference_ms", "rules_ms"])
    lat["online_ms"] = lat.sum(axis=1)
    return lat

lat = latency_benchmark(VA)
# bộ giải MILP chạy theo lô mỗi đợt → chi phí chia đều cho từng hộ
solver_per_unit_ms = float((milp_tab["giải (s)"].sum() / milp_tab["n hộ đủ ĐK"].sum()) * 1e3)
lat["optimizer_amortized_ms"] = solver_per_unit_ms
lat["total_ms"] = lat.online_ms + lat.optimizer_amortized_ms
q = lambda s: pd.Series({"P50": s.quantile(.50), "P95": s.quantile(.95), "P99": s.quantile(.99), "TB": s.mean()})
lat_tab = pd.DataFrame({c: q(lat[c]) for c in ["feature_ms", "inference_ms", "rules_ms", "optimizer_amortized_ms", "online_ms", "total_ms"]}).T.round(2)
budget_ms = {"feature_ms": 60, "inference_ms": 150, "rules_ms": 10, "optimizer_amortized_ms": 20, "online_ms": 250, "total_ms": 270}
lat_tab["Ngân sách P95 (giả định)"] = pd.Series(budget_ms)
lat_tab["Trong ngân sách"] = lat_tab["P95"] <= lat_tab["Ngân sách P95 (giả định)"]
show_df(lat_tab, "Độ trễ đo thật (ms) – chạy tuần tự từng yêu cầu trên máy hiện tại")
print(f"Thời gian giải MILP trung bình mỗi đợt: {milp_tab['giải (s)'].mean():.3f}s (chia đều ≈ {solver_per_unit_ms:.3f} ms/hộ). "
      "Hàng đợi và thực thi gửi coupon nằm ngoài phạm vi đo của bài (không mô phỏng để tránh số liệu giả).")
save_table(lat_tab, "latency_end_to_end")
P95_ONLINE = float(lat_tab.loc["online_ms", "P95"])
fig, ax = plt.subplots(figsize=(7, 3.4))
comp = lat_tab.loc[["feature_ms", "inference_ms", "rules_ms", "optimizer_amortized_ms"], ["P50", "P95", "P99"]]
comp.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518", "#E45756"]); ax.set_ylabel("ms"); ax.set_title("Độ trễ theo thành phần"); ax.set_xticklabels(["đặc trưng", "suy luận", "luật", "bộ giải (chia đều)"], rotation=0)
save_fig("fig_8_latency")

# %%
# ---------------------------------------------------------------------------
# 8.4 Đánh giá thử trên Validation (kiểm tra mã) và cổng phát hành thử
# ---------------------------------------------------------------------------
m_val, dec_val_eval, aux_val = evaluate_split(VA, "Validation")
gates_val = gate_table(m_val, P95_ONLINE)
show_df(pd.Series(m_val).to_frame("Validation").T.T, "Chỉ số thực chứng trên Validation")
show_df(gates_val, "Cổng phát hành – chạy thử trên Validation (chưa phải kết luận cuối)")

# %%
# ---------------------------------------------------------------------------
# 8.5 KHÓA: ghi biên bản khóa (một lần, chỉ đọc), rồi mở niêm phong Locked/OOT
# ---------------------------------------------------------------------------
import getpass
SEAL_PATH, SEAL_LOG = OUT_DIR / "locked_seal.json", OUT_DIR / "seal_log.jsonl"

def _script_sha():
    try:
        return sha256_file(Path(__file__))
    except Exception:
        return "n/a (chạy trong notebook)"

def seal_payload():
    sealed_cols = ["household_key", "wave", "decision_day", "T", "Y", "Y_active"] + FEATURES
    return {
        "thời điểm khóa (UTC)": dt.datetime.now(dt.timezone.utc).isoformat(), "thời điểm khóa (cục bộ)": dt.datetime.now().astimezone().isoformat(),
        "người chạy": getpass.getuser(), "máy": platform.node(), "RUN_STAGE": RUN_STAGE,
        "sha256 mô hình": freeze_hashes, "sha256 dự báo mô hình (đối chiếu)": FINGERPRINT, "sha256 cấu hình + cổng phát hành": sha256_obj({"cfg": FREEZE_JSON}),
        "sha256 dữ liệu đầu vào": dict(zip(data_contract_in.file, data_contract_in.sha256)),
        "sha256 hàng Locked+OOT (đặc trưng + nhãn)": hashlib.sha256(pd.util.hash_pandas_object(_sealed_panel[sealed_cols], index=False).values.tobytes()).hexdigest(),
        "số hàng Locked/OOT": {"locked": int((_sealed_panel.split == "locked").sum()), "oot": int((_sealed_panel.split == "oot").sum())},
        "sha256 mã nguồn": _script_sha(), "phiên bản": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                                       "sklearn": sklearn.__version__, "scipy": scipy.__version__},
        "cam kết": "Locked Test và OOT chưa được dùng để huấn luyện, chọn siêu tham số, chọn ngưỡng hay chỉnh cổng phát hành trước thời điểm này.",
    }

def write_or_verify_seal():
    new = seal_payload()
    if SEAL_PATH.exists():
        old = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
        checks = {"mô hình (dấu vân tay dự báo)": old["sha256 dự báo mô hình (đối chiếu)"] == new["sha256 dự báo mô hình (đối chiếu)"],
                  "cấu hình + cổng": old["sha256 cấu hình + cổng phát hành"] == new["sha256 cấu hình + cổng phát hành"],
                  "dữ liệu đầu vào": old["sha256 dữ liệu đầu vào"] == new["sha256 dữ liệu đầu vào"],
                  "hàng Locked+OOT": old["sha256 hàng Locked+OOT (đặc trưng + nhãn)"] == new["sha256 hàng Locked+OOT (đặc trưng + nhãn)"]}
        print("Biên bản khóa ĐÃ TỒN TẠI (không ghi đè). Đối chiếu mã băm với lần chạy hiện tại:", checks)
        if not all(checks.values()):
            raise RuntimeError("Thành phần đã bị thay đổi sau khi khóa – kết quả Locked/OOT không còn hợp lệ.")
        return old
    new["seal_hash"] = sha256_obj(new)
    SEAL_PATH.write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
    os.chmod(SEAL_PATH, stat.S_IREAD)                                           # chỉ đọc
    with open(SEAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"seal_hash": new["seal_hash"], "utc": new["thời điểm khóa (UTC)"]}, ensure_ascii=False) + "\n")
    print("ĐÃ GHI biên bản khóa:", SEAL_PATH, "| seal_hash =", new["seal_hash"][:16], "…")
    return new

if RUN_STAGE == "final":
    SEAL = write_or_verify_seal()
    LK = _sealed_panel[_sealed_panel.split == "locked"].reset_index(drop=True)
    OO = _sealed_panel[_sealed_panel.split == "oot"].reset_index(drop=True)
    print(f"Mở niêm phong: Locked {len(LK):,} hàng (đợt {sorted(LK.wave.unique())}), OOT {len(OO):,} hàng (đợt {sorted(OO.wave.unique())})")
else:
    SEAL = None
    preview = seal_payload()
    print("[DEV] CHƯA KHÓA. Nếu đặt RUN_STAGE='final', biên bản sẽ gồm các mục:", list(preview.keys()))
    print("[DEV] Locked/OOT vẫn bị niêm phong:", {k: int(v) for k, v in preview["số hàng Locked/OOT"].items()}, "hàng chưa được dùng.")

# %%
# ---------------------------------------------------------------------------
# 8.6 Đánh giá thực chứng MỘT LẦN trên Locked & OOT + cổng phát hành
# ---------------------------------------------------------------------------
FINAL_RESULTS = {}
if RUN_STAGE == "final":
    for nm, dfx in [("Locked Test", LK), ("Out-of-Time", OO)]:
        m_, dec_, aux_ = evaluate_split(dfx, nm)
        FINAL_RESULTS[nm] = (m_, dec_, aux_, gate_table(m_, P95_ONLINE))
    res_tab = pd.DataFrame({nm: pd.Series(v[0]) for nm, v in FINAL_RESULTS.items()})
    res_tab["Validation (tham chiếu)"] = pd.Series(m_val)
    show_df(res_tab, "Kết quả thực chứng: Validation vs Locked Test vs Out-of-Time")
    save_table(res_tab, "final_results")
    for nm, v in FINAL_RESULTS.items():
        show_df(v[3], f"Release Gates – {nm}")
        save_table(v[3], f"release_gates_{nm.replace(' ', '_')}")
else:
    print("[DEV] Bỏ qua đánh giá Locked/OOT. Bảng cổng phát hành thử trên Validation ở mục 8.4.")

# %%
# ---------------------------------------------------------------------------
# 8.7 Giám sát trôi dạt theo đợt: PSI, Action TVD (không cần nhãn) và Brier, ECE (cần nhãn → trễ 28 ngày)
# ---------------------------------------------------------------------------
mon_frames = [VA] + ([LK, OO] if RUN_STAGE == "final" else [])
MON = pd.concat(mon_frames, ignore_index=True)
dec_mon = action_router(MON, MON[FEATURES])
p_mon = CALIBRATOR(clf_final.predict_proba(MON[FEATURES])[:, 1])
mon_rows = []
for w in sorted(MON.wave.unique()):
    mk = (MON.wave == w).values
    mix = dec_mon[mk].action.value_counts(normalize=True).reindex(ACTIONS, fill_value=0).values
    d0 = int(MON.decision_day[mk].iloc[0])
    mon_rows.append({"đợt": w, "tập": MON.split[mk].iloc[0], "ngày quyết định": d0, "ngày có nhãn": d0 + WAVE_LEN,
                     "PSI spend_90": psi(X_tr.spend_90, MON.spend_90[mk]), "PSI recency": psi(X_tr.recency, MON.recency[mk]),
                     "PSI điểm dự báo": psi(OOF_CAL_PRED, p_mon[mk]), "Action TVD": tvd(mix, REF_MIX),
                     "Brier (sau khi có nhãn)": brier_score_loss(MON.Y_active[mk], p_mon[mk]), "ECE (sau khi có nhãn)": expected_calibration_error(MON.Y_active[mk], p_mon[mk]),
                     "AUC (sau khi có nhãn)": roc_auc_score(MON.Y_active[mk], p_mon[mk])})
mon_df = pd.DataFrame(mon_rows).set_index("đợt").round(4)
show_df(mon_df, "Giám sát theo đợt – chỉ báo không cần nhãn có ngay tại ngày quyết định; chỉ báo cần nhãn chỉ có sau 28 ngày")
save_table(mon_df, "monitoring_by_wave")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(mon_df.index, mon_df["PSI spend_90"], "o-", label="PSI spend_90"); ax[0].plot(mon_df.index, mon_df["PSI recency"], "s-", label="PSI recency")
ax[0].plot(mon_df.index, mon_df["PSI điểm dự báo"], "^-", label="PSI điểm dự báo"); ax[0].axhline(0.25, color="r", ls=":"); ax[0].set_xlabel("Đợt"); ax[0].legend(fontsize=8); ax[0].set_title("Trôi dữ liệu / điểm dự báo (ngưỡng 0,25)")
ax[1].plot(mon_df.index, mon_df["Brier (sau khi có nhãn)"], "o-", label="Brier"); ax[1].plot(mon_df.index, mon_df["ECE (sau khi có nhãn)"], "s-", label="ECE"); ax[1].plot(mon_df.index, mon_df["Action TVD"], "^-", label="Action TVD")
ax[1].set_xlabel("Đợt"); ax[1].legend(fontsize=8); ax[1].set_title("Chất lượng (trễ nhãn 28 ngày) và dịch chuyển hành động")
save_fig("fig_8_monitoring")

# %%
# ---------------------------------------------------------------------------
# 8.8 Regret và quan hệ PHI TUYẾN giữa sai số dự báo và regret (mô phỏng bán tổng hợp có sự thật biết trước)
# ---------------------------------------------------------------------------
# Thế giới bán tổng hợp: lấy τ*(x) = CATE dị biệt đã học (chuẩn hóa độ lệch chuẩn = 10 USD, trung bình = 2 USD) làm SỰ THẬT;
# dự báo τ̂ = τ* + nhiễu N(0, σ²); quyết định = chọn top-K theo MARGIN·τ̂ − c; regret = giá trị tối ưu (biết τ*) − giá trị thực hiện.
ev_idx = idx_va_sup
tau_true = tau_het.predict(X_va)[ev_idx]
tau_true = 2.0 + 10.0 * (tau_true - tau_true.mean()) / (tau_true.std() + 1e-9)
c_true = COST_va[ev_idx]; waves_sub = wave_va[ev_idx]
sig_grid = np.array([0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]) * 10.0
reg_rows = []
rr = np.random.default_rng(SEED + 11)
for sg in sig_grid:
    regs = []
    for rep in range(15):
        pred = tau_true + rr.normal(0, sg, len(tau_true))
        Bk = make_blocks(np.arange(len(ev_idx)), waves_sub, rr)
        V_true = (MARGIN * tau_true - c_true)[Bk]
        V_pred = (MARGIN * pred - c_true)[Bk]
        W_opt = select_rows(V_true, K_BLOCK, True); W_pol = select_rows(V_pred, K_BLOCK, True)
        regs.append(((W_opt * V_true).sum() - (W_pol * V_true).sum()) / V_true.size)
    reg_rows.append({"σ nhiễu dự báo (USD)": sg, "RMSE dự báo (USD)": sg, "regret TB (USD/hộ)": float(np.mean(regs)), "sd": float(np.std(regs))})
reg_df = pd.DataFrame(reg_rows).round(4)
show_df(reg_df, "Sai số dự báo CATE và regret quyết định (bán tổng hợp)")
save_table(reg_df, "regret_vs_forecast_error")
pos = reg_df[(reg_df["σ nhiễu dự báo (USD)"] > 0) & (reg_df["regret TB (USD/hộ)"] > 0)]
def _slope(m):
    s_ = pos[m]
    return np.polyfit(np.log(s_["σ nhiễu dự báo (USD)"]), np.log(s_["regret TB (USD/hộ)"]), 1)[0]
sl_low, sl_high = _slope(pos["σ nhiễu dự báo (USD)"] <= 10), _slope(pos["σ nhiễu dự báo (USD)"] >= 15)
print(f"Độ dốc log–log của regret theo sai số: vùng sai số nhỏ (σ ≤ 10 USD) = {sl_low:.2f}; vùng sai số lớn (σ ≥ 15 USD) = {sl_high:.2f}. " +
      ("Dốc > 1 ở vùng nhỏ (regret tăng nhanh hơn sai số) rồi thoải dần ở vùng lớn ⇒ quan hệ phi tuyến, không phải tỷ lệ thuận." if sl_low > sl_high + 0.3
       else "Hai độ dốc gần nhau ⇒ chưa thấy phi tuyến rõ rệt trong mô phỏng này."))
plt.figure(figsize=(5.6, 3.6)); plt.errorbar(reg_df["RMSE dự báo (USD)"], reg_df["regret TB (USD/hộ)"], yerr=reg_df["sd"], fmt="o-")
plt.xlabel("Sai số dự báo CATE – RMSE (USD)"); plt.ylabel("Regret (USD/hộ-đợt)"); plt.title("Regret phi tuyến theo sai số dự báo (bán tổng hợp)")
save_fig("fig_8_regret_curve")

# %%
# ---------------------------------------------------------------------------
# 8.9 Current vs Challenger (tái huấn luyện có trách nhiệm)
# ---------------------------------------------------------------------------
def train_bundle(df_tr):
    Xb, yb, tb, gb = df_tr[FEATURES], df_tr.Y_w.values, df_tr["T"].values, df_tr.household_key.values
    ya = df_tr.Y_active
    clf_b = hgb_clf(**BEST_KW).fit(Xb, ya)
    oof = np.zeros(len(Xb))
    for tri, tei in GroupKFold(n_splits=3).split(Xb, ya, gb):
        oof[tei] = hgb_clf(**BEST_KW).fit(Xb.iloc[tri], ya.iloc[tri]).predict_proba(Xb.iloc[tei])[:, 1]
    iso_b = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, ya)
    cf = crossfit(Xb, yb, tb, gb, n_splits=4)
    sp_ = (cf["e"] >= PS_CLIP[0]) & (cf["e"] <= PS_CLIP[1])
    G_ = dr_score(yb, tb, cf["e"], cf["mu1"], cf["mu0"])
    prop_b = hgb_clf(**NUIS_CLF).fit(Xb, tb)
    mu1_b = hgb_reg(**NUIS_REG).fit(Xb[tb == 1], yb[tb == 1]); mu0_b = hgb_reg(**NUIS_REG).fit(Xb[tb == 0], yb[tb == 0])
    return {"clf": clf_b, "iso": iso_b, "ate": float(G_[sp_].mean()), "ate_se": cluster_se_mean(G_[sp_], gb[sp_]), "prop": prop_b, "mu1": mu1_b, "mu0": mu0_b, "n": len(df_tr)}

def eval_bundle(b, df):
    X = df[FEATURES]; y = df.Y_active.values
    p = b["iso"].predict(b["clf"].predict_proba(X)[:, 1])
    e = b["prop"].predict_proba(X)[:, 1]; sp = (e >= PS_CLIP[0]) & (e <= PS_CLIP[1])
    G = dr_score(df.Y_w.values, df["T"].values, e, b["mu1"].predict(X), b["mu0"].predict(X))
    return {"AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p), "ECE": expected_calibration_error(y, p),
            "ATE hiệu quả coupon (DR)": float(G[sp].mean()), "SE (gom cụm)": cluster_se_mean(G[sp], df.household_key.values[sp])}

if RUN_STAGE == "final":
    cur_train, chal_train, eval_sets = TR, pd.concat([TR, VA], ignore_index=True), {"Locked Test": LK, "Out-of-Time": OO}
else:   # dev: chạy thử cơ chế trên Validation (Current = Train đợt 0–5, Challenger = Train đợt 0–7)
    cur_train, chal_train, eval_sets = TR[TR.wave <= 5].reset_index(drop=True), TR, {"Validation (chạy thử)": VA}
t0 = time.time()
B_cur, B_chal = train_bundle(cur_train), train_bundle(chal_train)
print(f"Huấn luyện Current (n={B_cur['n']:,}) và Challenger (n={B_chal['n']:,}) xong sau {time.time() - t0:.0f}s")
cc_rows = []
for nm, dfx in eval_sets.items():
    for who, b in [("Current", B_cur), ("Challenger", B_chal)]:
        r_ = eval_bundle(b, dfx); r_.update({"tập đánh giá": nm, "mô hình": who, "ATE huấn luyện": b["ate"]}); cc_rows.append(r_)
cc_df = pd.DataFrame(cc_rows).set_index(["tập đánh giá", "mô hình"]).round(4)
show_df(cc_df, "Current vs Challenger")
save_table(cc_df, "current_vs_challenger")

def promote_decision(cur, chal):
    ok_brier = chal["Brier"] <= cur["Brier"] * (1 + RELEASE_GATES["G1_brier_rel"][2])
    ok_ece = chal["ECE"] <= RELEASE_GATES["G2_ece"][2]
    better = chal["Brier"] < cur["Brier"]
    return ok_brier and ok_ece, better
dec_rows = []
for nm in eval_sets:
    cur, chal = cc_df.loc[(nm, "Current")], cc_df.loc[(nm, "Challenger")]
    ok, better = promote_decision(cur, chal)
    dec_rows.append({"tập": nm, "qua cổng (Brier không tệ hơn & ECE đạt)": ok, "Brier tốt hơn": better,
                     "quyết định": "THĂNG CẤP Challenger" if (ok and better) else "GIỮ Current"})
show_df(pd.DataFrame(dec_rows).set_index("tập"), "Quyết định thăng cấp theo cổng phát hành")

# %%
# ---------------------------------------------------------------------------
# 8.10 Vòng phản hồi, hợp đồng dữ liệu đầu ra, ma trận phiên bản và manifest tái lập
# ---------------------------------------------------------------------------
FEEDBACK = pd.DataFrame([
    ["Nhật ký quyết định", f"Mỗi quyết định ghi action, lý do, propensity của EXPLORE ({EXPLORE_P:.0%}), phiên bản mô hình", "Dùng IPW khi tái huấn luyện, khắc phục khuyết nhãn chọn lọc"],
    ["Nhãn về trễ", "Chi tiêu 28 ngày chỉ biết sau ngày quyết định + 28", "Chỉ số không cần nhãn (PSI, Action TVD) cảnh báo sớm; Brier/ECE xác nhận sau"],
    ["Kích hoạt tái huấn luyện", "PSI ≥ 0,25 hai đợt liên tiếp hoặc ECE > 0,05", "Huấn luyện Challenger → Release Gates → thăng cấp hoặc giữ Current"],
    ["Nhóm đối chứng EXPLORE", f"Đồng xu p = {EXPLORE_P:.0%} trên mọi hộ đủ điều kiện không thuộc SEND", "Nguồn dữ liệu nhân quả sạch cho chu kỳ sau; chi phí học bị chặn bởi ngân sách"],
], columns=["Thành phần", "Cách làm", "Mục đích"])
show_df(FEEDBACK, "Thiết kế vòng phản hồi")

data_contract_out = pd.DataFrame([
    ["household_key", "int", "Mã hộ", "không null"], ["wave", "int", "Đợt quyết định 0–17", "0 ≤ wave ≤ 17"], ["action", "str", "SEND/EXPLORE/NO_ACTION/SUPPRESS/HITL", "thuộc 5 giá trị"],
    ["reason", "str", "Lý do định tuyến", "không null"], ["p_explore", "float", "Propensity của EXPLORE (= EXPLORE_P) hoặc NaN", "0 < p < 1 nếu có"],
    ["ev, lcb, ucb", "float", "Giá trị kỳ vọng, cận dưới, cận trên (USD)", "lcb ≤ ev ≤ ucb"], ["cost", "float", "Chi phí tăng thêm kỳ vọng của một lần gửi (USD)", "≥ 0,10"],
    ["p_active", "float", "Xác suất quay lại 28 ngày (đã hiệu chỉnh)", "0 ≤ p ≤ 1"], ["propensity", "float", "P(được nhắm | x)", "0 ≤ e ≤ 1"],
], columns=["trường", "kiểu", "ý nghĩa", "ràng buộc"])
show_df(data_contract_out, "Hợp đồng dữ liệu đầu ra của Router")
save_table(data_contract_out.set_index("trường"), "data_contract_output")
# kiểm tra hợp đồng đầu ra trên kết quả thực tế
chk = dec_val_eval
assert chk.action.isin(ACTIONS).all() and chk.p_active.between(0, 1).all() and (chk.lcb <= chk.ev + 1e-9).all() and (chk.ev <= chk.ucb + 1e-9).all() and (chk.cost >= CONTACT_COST - 1e-9).all()
print("Hợp đồng dữ liệu đầu ra: đạt kiểm tra trên toàn bộ quyết định Validation.")

version_matrix = pd.DataFrame([
    ["Python", platform.python_version(), ">= 3.10"], ["numpy", np.__version__, ">= 1.26"], ["pandas", pd.__version__, ">= 2.0"],
    ["scikit-learn", sklearn.__version__, ">= 1.4"], ["scipy", scipy.__version__, ">= 1.11 (scipy.optimize.milp)"],
    ["statsmodels", sm.__version__, ">= 0.14"], ["matplotlib", matplotlib.__version__, ">= 3.8"], ["seaborn", sns.__version__, ">= 0.13"], ["joblib", joblib.__version__, ">= 1.3"],
], columns=["thư viện", "phiên bản đã chạy", "tối thiểu tương thích"]).set_index("thư viện")
show_df(version_matrix, "Ma trận tương thích phiên bản")
save_table(version_matrix, "version_matrix")

out_hashes = {}
for f in sorted((OUT_DIR / "tables").glob("*.csv")) + sorted((OUT_DIR / "figures").glob("*.png")) + sorted((OUT_DIR / "models").glob("*")):
    out_hashes[str(f.relative_to(OUT_DIR))] = sha256_file(f)
manifest = {"run_stage": RUN_STAGE, "chạy lúc": dt.datetime.now().astimezone().isoformat(), "seed": SEED, "input": dict(zip(data_contract_in.file, data_contract_in.sha256)),
            "output_sha256": out_hashes, "seal_hash": (SEAL or {}).get("seal_hash")}
(OUT_DIR / "manifest_sha256.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Đã ghi manifest SHA-256 cho {len(out_hashes)} tệp đầu ra → {OUT_DIR / 'manifest_sha256.json'}")
