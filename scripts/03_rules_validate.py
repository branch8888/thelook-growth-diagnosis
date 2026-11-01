# -*- coding: utf-8 -*-
"""
阶段 4｜L4 关联规则
阶段 5｜稳健性验证（年度稳定性 + 样本外验证 + 门槛敏感性）
阶段 6｜决策测算（情景 + 敏感性）
"""
import pandas as pd
import numpy as np
import itertools
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

orders = pd.read_csv(DATA / "orders.csv")
orders["created_at"] = pd.to_datetime(orders["created_at"], format="ISO8601", utc=True, errors="coerce")
orders["year"] = orders["created_at"].dt.year

items = pd.read_csv(DATA / "order_items.csv")
items["created_at"] = pd.to_datetime(items["created_at"], format="ISO8601", utc=True, errors="coerce")
items["year"] = items["created_at"].dt.year

products = pd.read_csv(DATA / "products.csv")
items = items.merge(products[["id", "category"]], left_on="product_id", right_on="id",
                    how="left", suffixes=("", "_p"))

# 订单 -> 品类集合 / 金额 / 年份
g = items.groupby("order_id")
ord_cats = g["category"].apply(lambda s: frozenset(s.dropna()))
ord_rev = g["sale_price"].sum()
ord_year = items.groupby("order_id")["year"].first()
O = pd.DataFrame({"cats": ord_cats, "revenue": ord_rev, "year": ord_year}).dropna(subset=["year"])
O["year"] = O["year"].astype(int)
O["n_cat"] = O["cats"].apply(len)

P("=" * 78); P("L4  关联规则（全量订单）"); P("=" * 78)
N = len(O)
P(f"订单总数 {N:,}   多品类订单 {(O.n_cat>=2).sum():,} ({(O.n_cat>=2).mean()*100:.1f}%)")

def build_rules(df, min_sup=0.003):
    cc = {}
    pre = {}
    for cats in df["cats"]:
        for c in cats:
            cc[c] = cc.get(c, 0) + 1
        if len(cats) >= 2:
            for a, b in itertools.combinations(sorted(cats), 2):
                pre[(a, b)] = pre.get((a, b), 0) + 1
    n = len(df); out = []
    for (a, b), k in pre.items():
        if k / n < min_sup: continue
        out.append({
            "a": a, "b": b, "n": k, "sup": k / n,
            "conf_ab": k / cc[a], "conf_ba": k / cc[b],
            "lift": (k / n) / ((cc[a] / n) * (cc[b] / n)),
        })
    return pd.DataFrame(out)

R = build_rules(O)
P(f"\n满足最小支持度 0.3% 的规则: {len(R)} 条")
P(f"\n正向关联 Top 12（lift > 1）：")
P(f"{'A':<28}{'B':<28}{'共现单':>8}{'Lift':>7}{'P(B|A)':>9}")
for _, r in R[R.lift > 1].sort_values("lift", ascending=False).head(12).iterrows():
    P(f"{r['a'][:27]:<28}{r['b'][:27]:<28}{int(r['n']):>8,}{r['lift']:>7.2f}{r['conf_ab']*100:>8.1f}%")

P("\n负向关联 Bottom 6（lift < 1，即直觉上该搭却搭不起来的组合）：")
for _, r in R.sort_values("lift").head(6).iterrows():
    P(f"{r['a'][:27]:<28}{r['b'][:27]:<28}{int(r['n']):>8,}{r['lift']:>7.2f}{r['conf_ab']*100:>8.1f}%")

P("\n" + "=" * 78); P("阶段 5-1  结构指标的年度稳定性"); P("=" * 78)
P(f"{'年份':<7}{'订单数':>9}{'单品类占比':>11}{'1品类AOV':>11}{'2品类AOV':>11}{'2-1 差值':>11}")
for y in sorted(O.year.unique()):
    d = O[O.year == y]
    s1 = d.loc[d.n_cat == 1, "revenue"]; s2 = d.loc[d.n_cat == 2, "revenue"]
    P(f"{y:<7}{len(d):>9,}{(d.n_cat==1).mean()*100:>10.1f}%{s1.mean():>11.2f}{s2.mean():>11.2f}{s2.mean()-s1.mean():>11.2f}")

P("\n" + "=" * 78); P("阶段 5-2  样本外验证：2019-2022 训练 → 2023 检验"); P("=" * 78)
tr = O[O.year <= 2022]; te = O[O.year == 2023]
Rtr = build_rules(tr, 40 / len(tr)); Rte = build_rules(te, 0.003)
P(f"训练集 {len(tr):,} 单   检验集 {len(te):,} 单")
m = Rtr.merge(Rte, on=["a", "b"], suffixes=("_tr", "_te"))
P(f"两边同时满足支持度的规则: {len(m)} 条")
top = m.sort_values("lift_tr", ascending=False).head(10)
P(f"\n{'规则':<50}{'训练lift':>9}{'2023lift':>10}{'判定':>7}")
keep = 0
for _, r in top.iterrows():
    ok = abs(r.lift_tr - r.lift_te) < 0.25
    keep += ok
    P(f"{(r['a'][:23] + ' + ' + r['b'][:23]):<50}{r.lift_tr:>9.2f}{r.lift_te:>10.2f}{'稳定' if ok else '漂移':>7}")
P(f"\n训练集 Top10 规则在 2023 检验集的保留率: {keep}/10")
P(f"全部规则 lift 相关系数: {m['lift_tr'].corr(m['lift_te']):.3f}")

P("\n" + "=" * 78); P("阶段 5-3  门槛敏感性"); P("=" * 78)
te_single = te[te.n_cat == 1]["revenue"]
P(f"2023 年单品类订单: {len(te_single):,}")
P(f"{'门槛':<10}{'触达订单':>10}{'占单品类':>10}")
for th in (20, 30, 40, 50, 60, 80):
    k = (te_single < th).sum()
    P(f"< ${th:<7}{k:>10,}{k/len(te_single)*100:>9.1f}%")

P("\n" + "=" * 78); P("阶段 6  决策测算"); P("=" * 78)
# 品类均价（用于估算"加一件"的增量）
cat_price = items.groupby("category")["sale_price"].agg(["mean", "median", "size"])
cat_price = cat_price[cat_price["size"] >= 1000]

single_all = O[O.n_cat == 1]
target = single_all[single_all.revenue < 50].copy()
P(f"目标池（客单价 < $50 的单品类订单）: {len(target):,} 单")
P(f"目标池客单价均值: ${target.revenue.mean():.2f}")

# 目标池中每单的品类
tcat = items[items.order_id.isin(target.index)].groupby("order_id")["category"].first()
target = target.join(tcat.rename("category"))
P(f"\n目标池品类分布 Top8 及推荐搭档：")
P(f"{'顾客买的品类':<30}{'订单数':>8}{'最佳搭档':<26}{'Lift':>6}{'搭档均价':>9}")
pair_map = {}
for c, cnt in target["category"].value_counts().head(8).items():
    cand = R[(R.a == c) | (R.b == c)].copy()
    if cand.empty: continue
    cand["partner"] = np.where(cand.a == c, cand.b, cand.a)
    cand = cand[cand["lift"] > 1.05]          # 只保留高于随机的搭档
    if cand.empty:
        P(f"{c[:29]:<30}{cnt:>8,}{'无正向搭档，改为按价格带凑单':<26}{'-':>6}{'-':>9}")
        continue
    best = cand.sort_values("lift", ascending=False).iloc[0]
    pv = cat_price.loc[best["partner"], "mean"] if best["partner"] in cat_price.index else np.nan
    pair_map[c] = (best["partner"], best["lift"], pv, cnt)
    P(f"{c[:29]:<30}{cnt:>8,}{best['partner'][:25]:<26}{best['lift']:>6.2f}{pv:>9.2f}")

# 情景测算：按"加一件搭档商品的均价"计算增量（比套用 $122 更保守也更可辩护）
P("\n情景测算：两种口径")
avg_add = np.mean([v[2] for v in pair_map.values() if not np.isnan(v[2])])
aov2 = O.loc[O.n_cat == 2, "revenue"].mean()
P(f"  口径A（朴素外推，存在选择偏差）：假定升级后客单价 = 观测到的双品类均值 ${aov2:.2f}")
P(f"        增量 = {aov2:.2f} − {target.revenue.mean():.2f} = ${aov2 - target.revenue.mean():.2f}/单")
P(f"        ⚠ 双品类订单本身单价就更高，直接外推会高估，仅作上界参考")
P(f"  口径B（严谨，推荐）：增量 = 搭档品类平均件价 = ${avg_add:.2f}/单")
P(f"        理由：给已有订单加一件商品，增加的就是那一件的价格")
P(f"\n{'转化率':<10}{'口径A 新增':>14}{'口径B 新增':>14}{'口径B 占总收入':>14}")
total_rev = O["revenue"].sum()
for pct in (0.05, 0.10, 0.15, 0.20):
    n_conv = int(len(target) * pct)
    gA = n_conv * (aov2 - target.revenue.mean()); gB = n_conv * avg_add
    P(f"{pct*100:>5.0f}%{'':<5}{gA:>14,.0f}{gB:>14,.0f}{gB/total_rev*100:>13.2f}%")

P(f"\n敏感性：转化率 × 平均增量（美元/单）")
P(f"{'转化率':<8}" + "".join(f"{f'${a:.0f}':>12}" for a in (15, 20, 25, 30, 40)))
for pct in (0.05, 0.10, 0.15, 0.20):
    row = f"{pct*100:>5.0f}%  "
    for a in (15, 20, 25, 30, 40):
        row += f"{int(len(target)*pct*a):>12,}"
    P(row)

with open(OUT / "03_rules_validate.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n>>> 已保存到 outputs/03_rules_validate.txt")
