# -*- coding: utf-8 -*-
"""
阶段 8.1｜图表生成
配色使用经 validate_palette.js 校验通过的参考调色板：
  slot1 #2a78d6 (blue)  slot2 #eb6834 (orange)
所有图：单一纵轴、实线细网格、克制标注、直接标签
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
_c = [ROOT / "data"] + sorted(ROOT.glob("**/data"))
DATA = next(p for p in _c if (p / "users.csv").exists())

# ---- 调色板 ----
SURF   = "#fcfcfb"
INK    = "#0b0b0b"
INK2   = "#52514e"
MUTED  = "#898781"
GRID   = "#e1e0d9"
BASE   = "#c3c2b7"
S1     = "#2a78d6"
S2     = "#eb6834"
NEUTRAL= "#d8d7d0"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

def style(ax, ylab=None, xlab=None, grid_axis="y"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    if ylab: ax.set_ylabel(ylab, color=INK2, fontsize=10)
    if xlab: ax.set_xlabel(xlab, color=INK2, fontsize=10)

def save(fig, name):
    fig.patch.set_facecolor(SURF)
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("saved", name)

# ============ 数据 ============
orders = pd.read_csv(DATA / "orders.csv")
orders["created_at"] = pd.to_datetime(orders["created_at"], format="ISO8601", utc=True, errors="coerce")
orders["year"] = orders["created_at"].dt.year
items = pd.read_csv(DATA / "order_items.csv")
items["created_at"] = pd.to_datetime(items["created_at"], format="ISO8601", utc=True, errors="coerce")
items["year"] = items["created_at"].dt.year
products = pd.read_csv(DATA / "products.csv")
items = items.merge(products[["id", "category", "retail_price"]], left_on="product_id",
                    right_on="id", how="left", suffixes=("", "_p"))

OL = items.groupby("order_id").agg(n_cat=("category", "nunique"),
                                   revenue=("sale_price", "sum")).reset_index()
OL = OL.merge(orders[["order_id", "year"]], on="order_id")

yr = OL.groupby("year").agg(o=("order_id", "nunique"), r=("revenue", "sum"))
yr = yr.loc[yr.index <= 2023]
yr["aov"] = yr["r"] / yr["o"]

# ---------- 图 1：增长归因（指数化，单轴） ----------
fig, ax = plt.subplots(figsize=(8.4, 4.6))
idx_o = yr["o"] / yr["o"].iloc[0] * 100
idx_a = yr["aov"] / yr["aov"].iloc[0] * 100
ax.plot(yr.index, idx_o, color=S1, linewidth=2, marker="o", markersize=7,
        markerfacecolor=S1, markeredgecolor=SURF, markeredgewidth=2, label="订单数")
ax.plot(yr.index, idx_a, color=S2, linewidth=2, marker="o", markersize=7,
        markerfacecolor=S2, markeredgecolor=SURF, markeredgewidth=2, label="客单价")
ax.text(2023, idx_o.iloc[-1] * 1.05, f"{idx_o.iloc[-1]:.0f}", color=S1,
        fontsize=11, ha="center", va="bottom", fontweight="bold")
ax.text(2023, idx_a.iloc[-1] + 40, f"{idx_a.iloc[-1]:.0f}", color=S2,
        fontsize=11, ha="center", va="bottom", fontweight="bold")
ax.axhline(100, color=BASE, linewidth=0.8, linestyle="-")
style(ax, "指数（2019 = 100）")
ax.set_xticks(yr.index)
ax.set_ylim(0, 2500)
ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK2)
ax.set_title("五年间订单数涨了 23 倍，客单价始终原地不动", color=INK,
             fontsize=13, loc="left", pad=14, fontweight="bold")
save(fig, "fig1_growth.png")

# ---------- 图 2：篮筐宽度 → 客单价 ----------
aov_b = OL.groupby("n_cat")["revenue"].agg(["mean", "size"])
cnt_b = aov_b["size"]
fig, ax = plt.subplots(figsize=(8.4, 4.6))
x = np.arange(len(aov_b))
bars = ax.bar(x, aov_b["mean"], width=0.55, color=S1,
              edgecolor=SURF, linewidth=2)
for i, (v, n) in enumerate(zip(aov_b["mean"], cnt_b)):
    ax.text(i, v + 6, f"${v:.2f}", ha="center", va="bottom",
            color=INK, fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{k} 个品类\n{n:,} 单 · {n/cnt_b.sum()*100:.1f}%"
                    for k, n in zip(aov_b.index, cnt_b)],
                   color=INK2, fontsize=9.5)
ax.set_ylim(0, 290)
ax.tick_params(axis="x", length=0, pad=8)
style(ax, "平均客单价（美元）")
ax.annotate("", xy=(0.86, 112), xytext=(0.14, 72),
            arrowprops=dict(arrowstyle="->", color=S2, lw=1.6))
ax.text(0.5, 138, "+$61.03", color=S2, fontsize=10.5, ha="center", fontweight="bold")
ax.set_title("每多买一个品类，客单价几乎正好翻倍", color=INK,
             fontsize=13, loc="left", pad=14, fontweight="bold")
save(fig, "fig2_basket.png")

# ---------- 图 3：价格带 × 单品类占比 ----------
first = items.sort_values("sale_price").groupby("order_id").tail(1)[["order_id", "retail_price"]]
tmp = OL.merge(first, on="order_id", how="left")
tmp["band"] = pd.cut(tmp["retail_price"], [0, 20, 40, 60, 100, 200, 1e9], right=False,
                     labels=["$0-20", "$20-40", "$40-60", "$60-100", "$100-200", "$200+"])
r = tmp.groupby("band", observed=True).apply(
    lambda d: (d.n_cat == 1).mean() * 100, include_groups=False)

fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.plot(range(len(r)), r.values, color=S1, linewidth=2, marker="o", markersize=8,
        markerfacecolor=S1, markeredgecolor=SURF, markeredgewidth=2)
ax.text(0, r.iloc[0] + 3.5, f"{r.iloc[0]:.1f}%", color=INK, fontsize=11,
        ha="left", va="bottom", fontweight="bold")
ax.text(len(r) - 1, r.iloc[-1] - 3.5, f"{r.iloc[-1]:.1f}%", color=INK, fontsize=11,
        ha="right", va="top", fontweight="bold")
ax.set_xticks(range(len(r)))
ax.set_xticklabels(r.index, color=INK2, fontsize=9.5)
ax.set_ylim(40, 102)
style(ax, "单品类订单占比（%）", "订单中最高单价商品所在的价格带")
ax.set_title("商品越便宜，越容易被单独买走", color=INK,
             fontsize=13, loc="left", pad=14, fontweight="bold")
save(fig, "fig3_priceband.png")

# ---------- 图 4：单品类订单分层（强调目标池 / 排除池） ----------
single = OL[OL.n_cat == 1].copy()
single["tier"] = pd.cut(single["revenue"], [0, 30, 50, 80, 120, 1e9],
                        labels=["$0-30", "$30-50", "$50-80", "$80-120", "$120+"], right=False)
tc = single.groupby("tier", observed=True)["revenue"].agg(["size", "mean"])
colors = [S1, S1, NEUTRAL, NEUTRAL, S2]

fig, ax = plt.subplots(figsize=(8.4, 4.4))
y = np.arange(len(tc))[::-1]
ax.barh(y, tc["size"], height=0.62, color=colors, edgecolor=SURF, linewidth=2)
for i, (yy, (n, m)) in enumerate(zip(y, zip(tc["size"], tc["mean"]))):
    ax.text(n + 900, yy, f"{n:,} 单 · 均价 ${m:.2f}", va="center",
            color=INK2, fontsize=10)
ax.set_yticks(y)
ax.set_yticklabels(tc.index, color=INK2, fontsize=10)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 62000)
ax.set_xticks([])
for s in ("left", "right", "top", "bottom"):
    ax.spines[s].set_visible(False)
ax.set_facecolor(SURF)
ax.grid(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=S1, label="目标池（< $50，53,943 单）"),
                   Patch(facecolor=S2, label="排除池（≥ $120，10,271 单）"),
                   Patch(facecolor=NEUTRAL, label="其余档位")],
          frameon=False, fontsize=9.5, labelcolor=INK2, loc="lower right")
ax.set_title("单品类订单里，真正该干预的只有中间这 53,943 单", color=INK,
             fontsize=13, loc="left", pad=14, fontweight="bold")
save(fig, "fig4_pools.png")

# ---------- 图 5：关联规则 lift ----------
rules = pd.DataFrame([
    ("Intimates + Socks & Hosiery", 1.27), ("Socks + Underwear", 1.22),
    ("Dresses + Intimates", 1.21), ("Intimates + Plus", 1.20),
    ("Pants + Underwear", 1.20), ("Intimates + Maternity", 1.20),
    ("Intimates + Leggings", 1.19), ("Pants + Socks", 1.18),
    ("Intimates + Pants & Capris", 1.15),
], columns=["rule", "lift"]).sort_values("lift")

fig, ax = plt.subplots(figsize=(8.4, 4.8))
y = np.arange(len(rules))
ax.barh(y, rules["lift"] - 1, left=1, height=0.6, color=S1,
        edgecolor=SURF, linewidth=2)
ax.axvline(1.0, color=BASE, linewidth=1)
for yy, v in zip(y, rules["lift"]):
    ax.text(v + 0.006, yy, f"{v:.2f}", va="center", color=INK2, fontsize=9.5)
ax.set_yticks(y)
ax.set_yticklabels(rules["rule"], color=INK2, fontsize=9.5)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0.98, 1.33)
ax.set_ylim(-1.9, len(rules) - 0.35)
ax.set_xticks([])
for s in ("left", "right", "top", "bottom"):
    ax.spines[s].set_visible(False)
ax.set_facecolor(SURF); ax.grid(False)
ax.text(0.982, -1.45, "竖线 = 随机水平 1.0　|　其余 76 条规则 lift 全部低于 1.0，品类之间没有可靠互补",
        color=INK2, fontsize=9.5, ha="left")
ax.set_title("85 条规则里，只有 9 条高于随机，且全部围绕内衣—袜品", color=INK,
             fontsize=13, loc="left", pad=14, fontweight="bold")
save(fig, "fig5_rules.png")

# ---------- 图 6：敏感性热力图 ----------
conv = [0.05, 0.10, 0.15, 0.20]
addv = [15, 20, 25, 30, 40]
pool = 53943
M = np.array([[pool * c * a for a in addv] for c in conv])
cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

fig, ax = plt.subplots(figsize=(7.6, 4.4))
im = ax.imshow(M, cmap=cmap, aspect="auto")
ax.set_xticks(range(len(addv)))
ax.set_xticklabels([f"${a}" for a in addv], color=INK2, fontsize=10)
ax.set_yticks(range(len(conv)))
ax.set_yticklabels([f"{int(c*100)}%" for c in conv], color=INK2, fontsize=10)
ax.set_xlabel("每单平均增量（美元）", color=INK2, fontsize=10)
ax.set_ylabel("单品类→双品类 转化率", color=INK2, fontsize=10)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        dark = v > M.max() * 0.55
        ax.text(j, i, f"{v/1000:,.0f}K", ha="center", va="center",
                color=("#ffffff" if dark else "#0b0b0b"), fontsize=10.5,
                fontweight="bold")
ax.tick_params(length=0, colors=MUTED)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("新增收入敏感性（美元）：基线情形约 +$100K", color=INK,
             fontsize=13, loc="left", pad=14, fontweight="bold")
save(fig, "fig6_sensitivity.png")

print("\n全部图表已生成 →", OUT)
