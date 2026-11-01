# -*- coding: utf-8 -*-
"""把文档正文里引用的代码原样跑一遍，核对注释中的数字"""
import pandas as pd, numpy as np, sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
_c = [ROOT / "data"] + sorted(ROOT.glob("**/data"))
DATA = next(p for p in _c if (p / "users.csv").exists())


def hdr(t):
    print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)


# ---------- 1.1 ----------
hdr("1.1 研究对象")
P = pd.read_csv(DATA / "products.csv")
print(len(P), P["category"].nunique(), P["brand"].nunique())
print(P["department"].value_counts().to_dict())
print(pd.cut(P["retail_price"], [0, 20, 40, 60, 100, 200, 1e9],
             right=False).value_counts().sort_index().to_string())
I = pd.read_csv(DATA / "order_items.csv").merge(
    P[["id", "brand", "retail_price", "cost"]],
    left_on="product_id", right_on="id", how="left", suffixes=("", "_p"))
print(I.groupby("brand")["sale_price"].sum().nlargest(5).to_string())

# ---------- 1.3 ----------
hdr("1.3 业务模式")
U = pd.read_csv(DATA / "users.csv")
DC = pd.read_csv(DATA / "distribution_centers.csv")
print(U["country"].nunique())
print((U["country"].value_counts() / len(U) * 100).head(3).to_string())
print(round((U["country"] != "United States").mean() * 100, 1))
print((U["traffic_source"].value_counts(normalize=True) * 100).to_string())
print(len(DC))

# ---------- 2.1 ----------
hdr("2.1 增长现状")
O = pd.read_csv(DATA / "orders.csv")
O["created_at"] = pd.to_datetime(O["created_at"], format="ISO8601", utc=True, errors="coerce")
O["year"] = O["created_at"].dt.year

I = pd.read_csv(DATA / "order_items.csv")
I = I.merge(P[["id", "category", "brand", "retail_price", "cost"]],
            left_on="product_id", right_on="id", how="left", suffixes=("", "_p"))
OL = I.groupby("order_id").agg(n_cat=("category", "nunique"),
                               n_item=("product_id", "size"),
                               revenue=("sale_price", "sum")).reset_index()
OL = OL.merge(O[["order_id", "year"]], on="order_id", how="left")
yr = OL.groupby("year").agg(orders=("order_id", "nunique"),
                            rev=("revenue", "sum"),
                            items=("n_item", "sum"))
print(yr.round(2).to_string())

# ---------- 2.2 ----------
hdr("2.2 增长结构诊断")
yr["aov"] = yr["rev"] / yr["orders"]
yr["ipo"] = yr["items"] / yr["orders"]
y = yr.loc[yr.index <= 2023]
print(y[["orders", "rev", "aov", "ipo"]].round(2).to_string())
print("CV = %.2f%%" % (y["aov"].std(ddof=1) / y["aov"].mean() * 100))
print("均值 %.2f 标准差 %.2f" % (y["aov"].mean(), y["aov"].std(ddof=1)))

# ---------- 2.3 ----------
hdr("2.3 核心问题定义")
a, b = yr.loc[2019], yr.loc[2023]
ln_rev, ln_ord, ln_aov = np.log(b.rev/a.rev), np.log(b.orders/a.orders), np.log(b.aov/a.aov)
print(f"订单数贡献 {ln_ord/ln_rev*100:.1f}%   客单价贡献 {ln_aov/ln_rev*100:.1f}%")
print((OL["n_cat"].value_counts(normalize=True).sort_index() * 100).round(1).to_string())
print(OL.groupby("n_cat")["revenue"].mean().round(2).to_string())

first = I.sort_values("sale_price").groupby("order_id").tail(1)[["order_id", "retail_price"]]
tmp = OL.merge(first, on="order_id", how="left")
tmp["band"] = pd.cut(tmp["retail_price"], [0, 20, 40, 60, 100, 200, 1e9], right=False)
print((tmp.groupby("band", observed=True)
         .apply(lambda d: (d.n_cat == 1).mean() * 100, include_groups=False)
         .round(1)).to_string())

# ---------- 7.1 ----------
hdr("7.1 建议总览")
single = OL[OL.n_cat == 1]
total_rev = OL["revenue"].sum()
tcat = I.groupby("order_id")["category"].first()
single = single.set_index("order_id").join(tcat.rename("category"))
target = single[single.revenue < 50]
excl = single[single.revenue >= 120]
print(len(target), round(target.revenue.mean(), 2), round(target.revenue.sum()/total_rev*100, 1))
print(len(excl), round(excl.revenue.mean(), 2), round(excl.revenue.sum()/total_rev*100, 1))

cat_price = I.groupby("category")["sale_price"].agg(["mean", "size"])
cat_price = cat_price[cat_price["size"] >= 1000]
avg_add_R1 = (cat_price.loc["Socks & Hosiery", "mean"] + cat_price.loc["Socks", "mean"]) / 2
avg_add_R2 = (cat_price.loc["Socks & Hosiery", "mean"] + cat_price.loc["Underwear", "mean"]) / 2
print("R1 增量 %.2f  R2 增量 %.2f" % (avg_add_R1, avg_add_R2))
print(round(int(len(target) * 0.10) * avg_add_R1))
n3 = target["category"].isin(["Intimates", "Underwear", "Socks"]).sum()
print(n3, round(n3 * 0.15) * avg_add_R2)

# ---------- 7.3 ----------
hdr("7.3 内衣三品类")
for c in ["Intimates", "Underwear", "Socks"]:
    sub = target[target["category"] == c]
    print(f"{c:<12}{len(sub):>7,}   {sub.revenue.mean():>7.2f}")

# ---------- 7.5 ----------
hdr("7.5 低价 SKU")
low = I[I["retail_price"] < 20]
n_sku = (P["retail_price"] < 20).sum()
rev, cost = low["sale_price"].sum(), low["cost"].sum()
print(n_sku, round(n_sku / len(P) * 100, 1))
print(round(rev), round(rev / I["sale_price"].sum() * 100, 1))
print(round((rev - cost) / rev * 100, 1))

inv = pd.read_csv(DATA / "inventory_items.csv")
sku_rev = I.groupby("product_id")["sale_price"].sum()
sku_cost = inv.groupby("product_id")["cost"].sum()
skus = sorted(P["id"])
for pct in (0.2, 0.5):
    cut = sorted(skus, key=lambda s: sku_rev.get(s, 0))[:int(len(skus) * pct)]
    print(pct,
          round(sku_rev.reindex(cut).sum() / sku_rev.sum() * 100, 1),
          round(sku_cost.reindex(cut).sum() / sku_cost.sum() * 100, 1))
