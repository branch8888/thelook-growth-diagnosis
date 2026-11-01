# -*- coding: utf-8 -*-
"""
阶段 1｜L1 增长归因
阶段 2｜L2 篮筐诊断
阶段 3｜L3 目标池细分
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
_c = [ROOT / "data"] + sorted(ROOT.glob("**/data"))
DATA = next(p for p in _c if (p / "users.csv").exists())

lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

# ---------------- 读数据 ----------------
orders = pd.read_csv(DATA / "orders.csv")
orders["created_at"] = pd.to_datetime(orders["created_at"], format="ISO8601", utc=True, errors="coerce")
orders["year"] = orders["created_at"].dt.year

items = pd.read_csv(DATA / "order_items.csv")
items["created_at"] = pd.to_datetime(items["created_at"], format="ISO8601", utc=True, errors="coerce")
items["year"] = items["created_at"].dt.year

products = pd.read_csv(DATA / "products.csv")
items = items.merge(products[["id", "category", "brand", "retail_price", "cost"]],
                    left_on="product_id", right_on="id", how="left", suffixes=("", "_prod"))

# 订单级宽表
OL = items.groupby("order_id").agg(
    n_cat=("category", "nunique"),
    n_item=("product_id", "size"),
    revenue=("sale_price", "sum"),
    status=("status", "first"),
).reset_index()
OL = OL.merge(orders[["order_id", "year"]], on="order_id", how="left")

P("=" * 78); P("L1  增长归因：收入 = 订单数 × 客单价"); P("=" * 78)
yr = OL.groupby("year").agg(orders=("order_id", "nunique"),
                            revenue=("revenue", "sum"),
                            items=("n_item", "sum")).reset_index()
yr["aov"] = yr["revenue"] / yr["orders"]
yr["items_per_order"] = yr["items"] / yr["orders"]

P(f"{'年份':<7}{'订单数':>9}{'销售额':>13}{'客单价':>9}{'件/单':>8}")
for _, r in yr.iterrows():
    P(f"{int(r['year']):<7}{int(r['orders']):>9,}{r['revenue']:>13,.0f}{r['aov']:>9.2f}{r['items_per_order']:>8.2f}")

a = yr[yr.year == 2019].iloc[0]; b = yr[yr.year == 2023].iloc[0]
g_ord = b["orders"] / a["orders"]; g_aov = b["aov"] / a["aov"]; g_rev = b["revenue"] / a["revenue"]
P(f"\n2019 → 2023：")
P(f"  订单数  {int(a['orders']):>8,} → {int(b['orders']):>8,}   增长 {g_ord:>6.2f}×")
P(f"  客单价  {a['aov']:>8.2f} → {b['aov']:>8.2f}   变化 {g_aov:>6.3f}×")
P(f"  销售额  {a['revenue']:>8,.0f} → {b['revenue']:>8,.0f}   增长 {g_rev:>6.2f}×")
P(f"  校验    {g_ord:.3f} × {g_aov:.3f} = {g_ord*g_aov:.3f}   （实际 {g_rev:.3f}）")
# 用对数分解计算贡献占比：ln(g_rev) = ln(g_ord) + ln(g_aov)
ln_rev, ln_ord, ln_aov = np.log(g_rev), np.log(g_ord), np.log(g_aov)
P(f"\n  对数分解    ln({g_rev:.3f}) = ln({g_ord:.3f}) + ln({g_aov:.4f})")
P(f"              {ln_rev:.4f} = {ln_ord:.4f} + ({ln_aov:.4f})")
P(f"  贡献占比：订单数 {ln_ord/ln_rev*100:>6.1f}%    客单价 {ln_aov/ln_rev*100:>6.1f}%")
P(f"  即：增长几乎完全由订单数堆砌，客单价不仅没有贡献，还略微拉低了整体增速。")

# 稳健性：客单价的年度波动
P(f"\n  客单价 5 年均值 {yr['aov'].mean():.2f}，标准差 {yr['aov'].std(ddof=1):.2f}，CV = {yr['aov'].std(ddof=1)/yr['aov'].mean()*100:.2f}%")
P(f"  件/单 5 年均值 {yr['items_per_order'].mean():.2f}，CV = {yr['items_per_order'].std(ddof=1)/yr['items_per_order'].mean()*100:.2f}%")

P("\n" + "=" * 78); P("L2  篮筐诊断：客单价 × 订单内品类数"); P("=" * 78)
P(f"订单总数 {len(OL):,}")
P(f"\n{'品类数':<8}{'订单数':>9}{'占比':>8}{'客单价':>10}{'中位数':>10}{'P25':>9}{'P75':>9}")
for k in sorted(OL.n_cat.unique()):
    s = OL.loc[OL.n_cat == k, "revenue"]
    P(f"{k:<8}{len(s):>9,}{len(s)/len(OL)*100:>7.1f}%{s.mean():>10.2f}{s.median():>10.2f}{s.quantile(.25):>9.2f}{s.quantile(.75):>9.2f}")

groups = [OL.loc[OL.n_cat == k, "revenue"].values for k in sorted(OL.n_cat.unique())]
H, p = stats.kruskal(*groups)
n_all = len(OL); k_all = len(groups)
eps2 = (H - k_all + 1) / (n_all - k_all)
P(f"\nKruskal-Wallis 检验（H0：各篮筐宽度的客单价分布相同）")
P(f"  H = {H:,.2f}, p = {p:.3e}, epsilon² = {eps2:.4f}")
P(f"  → {'拒绝' if p < 0.001 else '不能拒绝'} H0，篮筐宽度对客单价有{'显著' if p < 0.001 else '不显著'}影响")
P(f"  效应量 epsilon² = {eps2:.3f}（>0.14 为大效应）")

# 相邻档位差
P(f"\n  相邻档位客单价差：")
aov_by = OL.groupby("n_cat")["revenue"].mean()
prev = None
for k, v in aov_by.items():
    if prev is not None:
        P(f"    {k-1} → {k} 品类：{aov_by[k-1]:>7.2f} → {v:>7.2f}   +{v-aov_by[k-1]:>7.2f}")
    prev = k

P("\n  按件数看（排除品类口径的干扰）：")
P(f"{'件数':<8}{'订单数':>9}{'占比':>8}{'客单价':>10}")
for k in sorted(OL.n_item.unique())[:6]:
    s = OL.loc[OL.n_item == k, "revenue"]
    P(f"{k:<8}{len(s):>9,}{len(s)/len(OL)*100:>7.1f}%{s.mean():>10.2f}")

P("\n  卡方检验：篮筐宽度是否与「首件商品价格带」独立")
first_item = items.sort_values("sale_price").groupby("order_id").tail(1)[["order_id", "retail_price"]]
tmp = OL.merge(first_item, on="order_id", how="left")
tmp["band"] = pd.cut(tmp["retail_price"], [0, 20, 40, 60, 100, 200, 1e9], right=False,
                     labels=["0-20", "20-40", "40-60", "60-100", "100-200", "200+"])
ct = pd.crosstab(tmp["band"], tmp["n_cat"])
chi2, p2, dof, _ = stats.chi2_contingency(ct)
V = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
P(f"    chi2 = {chi2:,.1f}, dof = {dof}, p = {p2:.3e}, Cramér's V = {V:.3f}")
P(f"    → 高价位商品所在订单的篮筐宽度分布与低价位{'不同' if p2 < 0.001 else '无差异'}")
P(f"\n    各价格带订单的篮筐宽度构成（行百分比）：")
P(f"    {'价格带':<10}{'订单数':>9}{'1品类':>8}{'2品类':>8}{'3品类':>8}{'4品类':>8}{'单品类占比':>11}")
rowpct = ct.div(ct.sum(axis=1), axis=0) * 100
for idx in ct.index:
    tot = ct.loc[idx].sum()
    P(f"    {str(idx):<10}{tot:>9,}" + "".join(f"{rowpct.loc[idx, c]:>7.1f}%" for c in ct.columns)
      + f"{rowpct.loc[idx, 1]:>10.1f}%")

P("\n" + "=" * 78); P("L3  目标池细分"); P("=" * 78)
single = OL[OL.n_cat == 1].copy()
P(f"单品类订单: {len(single):,}  占全部订单 {len(single)/len(OL)*100:.1f}%")
P(f"单品类订单贡献收入: {single['revenue'].sum():,.0f}  占全部 {single['revenue'].sum()/OL['revenue'].sum()*100:.1f}%")

bins = [0, 30, 50, 80, 120, 1e9]
labels = ["$0-30", "$30-50", "$50-80", "$80-120", "$120+"]
single["tier"] = pd.cut(single["revenue"], bins, labels=labels, right=False)
P(f"\n{'档位':<10}{'订单数':>9}{'占单品类':>10}{'客单价':>10}{'收入':>13}{'占总收入':>10}")
for t in labels:
    s = single[single.tier == t]
    if len(s) == 0: continue
    P(f"{t:<10}{len(s):>9,}{len(s)/len(single)*100:>9.1f}%{s['revenue'].mean():>10.2f}"
      f"{s['revenue'].sum():>13,.0f}{s['revenue'].sum()/OL['revenue'].sum()*100:>9.1f}%")

target = single[single.revenue < 50]; excl = single[single.revenue >= 120]
P(f"\n  【目标池】客单价 < $50 的单品类订单: {len(target):,} 单 ({len(target)/len(single)*100:.1f}% of 单品类)")
P(f"    贡献收入 {target['revenue'].sum():,.0f}（占总收入 {target['revenue'].sum()/OL['revenue'].sum()*100:.1f}%）")
P(f"  【排除池】客单价 ≥ $120 的单品类订单: {len(excl):,} 单 ({len(excl)/len(single)*100:.1f}% of 单品类)")
P(f"    贡献收入 {excl['revenue'].sum():,.0f}（占总收入 {excl['revenue'].sum()/OL['revenue'].sum()*100:.1f}%）")

P("\n  目标池的主力品类：")
t2 = items[items.order_id.isin(target.order_id)]
for c, n in t2["category"].value_counts().head(8).items():
    sub = target[target.order_id.isin(items.loc[items.category == c, "order_id"])]
    P(f"    {c:<34}{n:>7,} 单   客单价 {sub['revenue'].mean():>6.2f}")

P("\n  排除池的主力品类：")
e2 = items[items.order_id.isin(excl.order_id)]
for c, n in e2["category"].value_counts().head(6).items():
    sub = excl[excl.order_id.isin(items.loc[items.category == c, "order_id"])]
    P(f"    {c:<34}{n:>7,} 单   客单价 {sub['revenue'].mean():>6.2f}")

P("\n  目标池的用户属性（与全站对比）：")
users = pd.read_csv(DATA / "users.csv")
tg = users[users.id.isin(items[items.order_id.isin(target.order_id)].user_id)]
allu = users
P(f"    {'指标':<16}{'目标池':>12}{'全站':>12}")
P(f"    {'平均年龄':<16}{tg['age'].mean():>12.1f}{allu['age'].mean():>12.1f}")
for g in ["F", "M"]:
    P(f"    {'性别 ' + g:<16}{(tg.gender == g).mean()*100:>11.1f}%{(allu.gender == g).mean()*100:>11.1f}%")
for s in ["Search", "Organic", "Facebook", "Email", "Display"]:
    P(f"    {'渠道 ' + s:<18}{(tg.traffic_source == s).mean()*100:>10.1f}%{(allu.traffic_source == s).mean()*100:>11.1f}%")

with open(OUT / "02_growth_basket.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n>>> 已保存到 outputs/02_growth_basket.txt")
