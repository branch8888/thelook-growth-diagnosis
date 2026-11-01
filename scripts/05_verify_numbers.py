# -*- coding: utf-8 -*-
"""
复核脚本：复现正文第一、二、七章引用的全部数字
运行：python scripts/05_verify_numbers.py
"""
import pandas as pd, numpy as np, itertools, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
_c = [ROOT / "data"] + sorted(ROOT.glob("**/data"))
DATA = next(p for p in _c if (p / "users.csv").exists())

print("=" * 70); print("第一章 研究背景"); print("=" * 70)
P = pd.read_csv(DATA / "products.csv")
U = pd.read_csv(DATA / "users.csv")
O = pd.read_csv(DATA / "orders.csv")
O["created_at"] = pd.to_datetime(O["created_at"], format="ISO8601", utc=True, errors="coerce")
O["year"] = O["created_at"].dt.year
I = pd.read_csv(DATA / "order_items.csv")
I = I.merge(P[["id", "category", "brand", "retail_price", "cost"]],
            left_on="product_id", right_on="id", how="left", suffixes=("", "_p"))
DC = pd.read_csv(DATA / "distribution_centers.csv")

print(f"SKU 数              {len(P):,}")
print(f"品类数              {P['category'].nunique()}")
print(f"品牌数(nunique)     {P['brand'].nunique():,}")
print(f"实际售出过的品牌数   {I['brand'].nunique():,}")
print(f"部门                {dict(P['department'].value_counts())}")
print("\n品牌收入 Top10：")
for b, v in I.groupby("brand")["sale_price"].sum().nlargest(10).items():
    print(f"  {b:<22}{v:>12,.0f}")

print("\n价格带 SKU 分布：")
P["band"] = pd.cut(P["retail_price"], [0, 20, 40, 60, 100, 200, 1e9], right=False,
                   labels=["$0-20", "$20-40", "$40-60", "$60-100", "$100-200", "$200+"])
print(P["band"].value_counts().sort_index().to_string())

print("\n国家分布 Top5：")
for k, v in U["country"].value_counts().head(5).items():
    print(f"  {k:<16}{v:>8,}  {v/len(U)*100:>6.2f}%")
print(f"非美国用户占比      {(U['country'] != 'United States').mean()*100:.1f}%")
print("\n渠道分布：")
for k, v in U["traffic_source"].value_counts().items():
    print(f"  {k:<12}{v:>8,}  {v/len(U)*100:>6.2f}%")
print(f"配送中心数          {len(DC)}")

print("\n" + "=" * 70); print("第二章 问题界定"); print("=" * 70)
OL = I.groupby("order_id").agg(n_cat=("category", "nunique"),
                               n_item=("product_id", "size"),
                               revenue=("sale_price", "sum")).reset_index()
OL = OL.merge(O[["order_id", "year"]], on="order_id")

yr = OL.groupby("year").agg(orders=("order_id", "nunique"),
                            rev=("revenue", "sum"), items=("n_item", "sum"))
yr["aov"] = yr["rev"] / yr["orders"]
yr["ipo"] = yr["items"] / yr["orders"]
print(yr.round(2).to_string())
print(f"\n客单价年度变异系数  {yr.loc[yr.index<=2023,'aov'].std(ddof=1)/yr.loc[yr.index<=2023,'aov'].mean()*100:.2f}%")

a, b = yr.loc[2019], yr.loc[2023]
print(f"\n对数分解  ln({b.rev/a.rev:.3f}) = ln({b.orders/a.orders:.3f}) + ln({b.aov/a.aov:.4f})")
print(f"订单数贡献 {(np.log(b.orders/a.orders)/np.log(b.rev/a.rev))*100:.1f}%   "
      f"客单价贡献 {(np.log(b.aov/a.aov)/np.log(b.rev/a.rev))*100:.1f}%")

print(f"\n订单总数            {len(OL):,}")
print(f"单品类订单          {(OL.n_cat==1).sum():,}  ({(OL.n_cat==1).mean()*100:.1f}%)")
print(f"1/2 品类客单价      {OL[OL.n_cat==1].revenue.mean():.2f} / {OL[OL.n_cat==2].revenue.mean():.2f}")

first = I.sort_values("sale_price").groupby("order_id").tail(1)[["order_id", "retail_price"]]
tmp = OL.merge(first, on="order_id", how="left")
tmp["band"] = pd.cut(tmp["retail_price"], [0, 20, 40, 60, 100, 200, 1e9], right=False,
                     labels=["$0-20", "$20-40", "$40-60", "$60-100", "$100-200", "$200+"])
print("\n各价格带单品类率：")
print((tmp.groupby("band", observed=True)
         .apply(lambda d: (d.n_cat == 1).mean() * 100, include_groups=False)
         .round(1)).to_string())

print("\n" + "=" * 70); print("第七章 业务建议"); print("=" * 70)
single = OL[OL.n_cat == 1]
target = single[single.revenue < 50]
excl = single[single.revenue >= 120]
print(f"目标池(<$50 单品类)   {len(target):,} 单   AOV {target.revenue.mean():.2f}   "
      f"占收入 {target.revenue.sum()/OL.revenue.sum()*100:.1f}%")
print(f"排除池(>=$120 单品类) {len(excl):,} 单   AOV {excl.revenue.mean():.2f}   "
      f"占收入 {excl.revenue.sum()/OL.revenue.sum()*100:.1f}%")

y23 = OL[OL.year == 2023]
s23 = y23[y23.n_cat == 1]
print(f"2023 单品类订单       {len(s23):,}   其中 <$50: {(s23.revenue<50).sum():,} ({(s23.revenue<50).mean()*100:.1f}%)")

# 目标池品类分布与搭档均价
tcat = I[I.order_id.isin(target.order_id)].groupby("order_id")["category"].first()
target = target.set_index("order_id").join(tcat.rename("category"))
cat_price = I.groupby("category")["sale_price"].agg(["mean", "size"])
cat_price = cat_price[cat_price["size"] >= 1000]
print(f"\n目标池主力品类：")
for c, n in target["category"].value_counts().head(8).items():
    print(f"  {c:<34}{n:>7,}   单买客单价 {target[target.category==c].revenue.mean():>6.2f}")

print(f"\n搭档品类平均件价：")
for c in ["Socks & Hosiery", "Socks", "Underwear"]:
    print(f"  {c:<20}{cat_price.loc[c,'mean']:>8.2f}")
avg_add = (cat_price.loc["Socks & Hosiery", "mean"] + cat_price.loc["Underwear", "mean"]) / 2
print(f"  两搭档均值(R2口径)  {avg_add:>8.2f}")

print(f"\n内衣三品类合计：{5425+3478+3162:,} 单")

# 低价带
pb = I.copy()
pb["band"] = pd.cut(pb["retail_price"], [0,20,40,60,100,200,1e9], right=False)
low = pb[pb["band"] == pd.Interval(0, 20, closed="left")]
n_sku = int((P["retail_price"] < 20).sum())
rev, cost = low["sale_price"].sum(), low["cost"].sum()
print(f"$0-20 价格带：SKU {n_sku:,} 个 ({n_sku/len(P)*100:.1f}%)  "
      f"收入 {rev:,.0f} ({rev/pb['sale_price'].sum()*100:.1f}%)  "
      f"毛利率 {(rev-cost)/rev*100:.1f}%")
