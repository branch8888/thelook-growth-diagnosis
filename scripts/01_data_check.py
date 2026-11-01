# -*- coding: utf-8 -*-
"""
阶段 0｜数据准备与体检
0.1 数据获取校验  0.2 完整性校验  0.3 字段字典  0.4 质量审计
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# 自动定位 data 目录（兼容仓库嵌套结构）
_cands = [ROOT / "data"] + sorted(ROOT.glob("**/data"))
DATA = next((p for p in _cands if (p / "users.csv").exists()), None)
assert DATA is not None, "找不到包含 users.csv 的 data 目录"

lines = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)

P("=" * 78)
P("0.1  文件清单与完整性")
P("=" * 78)

TABLES = ["users", "orders", "order_items", "products",
          "inventory_items", "events", "distribution_centers"]

rows = {}
for t in TABLES:
    f = DATA / f"{t}.csv"
    if not f.exists():
        P(f"  ✗ 缺失: {t}.csv")
        continue
    with open(f, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    n = sum(1 for _ in open(f, "r", encoding="utf-8", errors="ignore")) - 1
    rows[t] = n
    P(f"  ✓ {t:<22} 行数={n:>9,}  列数={len(header):>2}  大小={f.stat().st_size/1024/1024:>7.1f} MB")

P("\n" + "=" * 78)
P("0.2  主键唯一性与引用完整性")
P("=" * 78)

users = pd.read_csv(DATA / "users.csv")
orders = pd.read_csv(DATA / "orders.csv")
order_items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")
dc = pd.read_csv(DATA / "distribution_centers.csv")
inventory = pd.read_csv(DATA / "inventory_items.csv")

P(f"  users.id            唯一值 {users['id'].nunique():>9,} / 行数 {len(users):>9,}  {'✓' if users['id'].is_unique else '✗ 有重复'}")
P(f"  orders.order_id     唯一值 {orders['order_id'].nunique():>9,} / 行数 {len(orders):>9,}  {'✓' if orders['order_id'].is_unique else '✗ 有重复'}")
P(f"  order_items.id      唯一值 {order_items['id'].nunique():>9,} / 行数 {len(order_items):>9,}  {'✓' if order_items['id'].is_unique else '✗ 有重复'}")
P(f"  products.id         唯一值 {products['id'].nunique():>9,} / 行数 {len(products):>9,}  {'✓' if products['id'].is_unique else '✗ 有重复'}")
P(f"  inventory_items.id  唯一值 {inventory['id'].nunique():>9,} / 行数 {len(inventory):>9,}  {'✓' if inventory['id'].is_unique else '✗ 有重复'}")

P("\n  引用完整性检查：")
checks = [
    ("orders.user_id → users.id",        orders["user_id"],       users["id"]),
    ("order_items.order_id → orders",    order_items["order_id"], orders["order_id"]),
    ("order_items.user_id → users.id",   order_items["user_id"],  users["id"]),
    ("order_items.product_id → products",order_items["product_id"], products["id"]),
    ("products.distribution_center_id → dc.id", products["distribution_center_id"], dc["id"]),
]
for name, child, parent in checks:
    bad = (~child.isin(parent)).sum()
    P(f"    {name:<45} 孤儿记录 {bad:>6,}  {'✓' if bad == 0 else '✗'}")

P("\n" + "=" * 78)
P("0.3  字段字典（基数与取值）")
P("=" * 78)

def profile(df, name, cols=None):
    P(f"\n--- {name}  ({len(df):,} 行 × {df.shape[1]} 列) ---")
    P(f"  {'字段':<22}{'非空':>9}{'缺失':>7}{'缺失率':>8}{'唯一值':>10}")
    for c in (cols or df.columns):
        miss = df[c].isna().sum()
        P(f"  {c:<22}{df[c].notna().sum():>9,}{miss:>7,}{miss/len(df)*100:>7.2f}%{df[c].nunique():>10,}")

profile(users, "users")
profile(orders, "orders")
profile(order_items, "order_items")
profile(products, "products")
profile(dc, "distribution_centers")

P("\n--- 关键分类字段取值分布 ---")
P("\n  users.country:")
for k, v in users["country"].value_counts().items():
    P(f"    {k:<18}{v:>8,}  {v/len(users)*100:>6.2f}%")
P("\n  users.traffic_source:")
for k, v in users["traffic_source"].value_counts().items():
    P(f"    {k:<18}{v:>8,}  {v/len(users)*100:>6.2f}%")
P("\n  users.gender:")
for k, v in users["gender"].value_counts().items():
    P(f"    {k:<18}{v:>8,}  {v/len(users)*100:>6.2f}%")
P("\n  orders.status:")
for k, v in orders["status"].value_counts().items():
    P(f"    {k:<18}{v:>8,}  {v/len(orders)*100:>6.2f}%")
P("\n  order_items.status:")
for k, v in order_items["status"].value_counts().items():
    P(f"    {k:<18}{v:>8,}  {v/len(order_items)*100:>6.2f}%")

P("\n" + "=" * 78)
P("0.4  质量审计")
P("=" * 78)

P("\n[审计 1] users.city 字面量 \"null\"")
P(f"  city == 'null' 的行数: {(users['city'] == 'null').sum():,}")
P(f"  pandas 读入后视为缺失: {users['city'].isna().sum():,}")

P("\n[审计 2] 同一(国家,城市)对应多个 state")
tmp = users.dropna(subset=["city"]).groupby(["country", "city"])["state"].nunique()
P(f"  (国家,城市) 组合数: {len(tmp):,}   对应多州的组合数: {(tmp > 1).sum():,}")

P("\n[审计 3] email 唯一性")
P(f"  email 唯一值: {users['email'].nunique():,} / {len(users):,}")
dup = users["email"].value_counts()
P(f"  重复邮箱数: {(dup > 1).sum():,}，涉及记录: {dup[dup > 1].sum():,}")
P(f"  最大重复次数: {dup.max()}")

P("\n[审计 4] 用户注册时间分布（月度）")
users["created_at"] = pd.to_datetime(users["created_at"], format="ISO8601", utc=True)
m = users["created_at"].dt.to_period("M").value_counts().sort_index()
P(f"  月份数: {len(m)}   月度注册量 min={m.min():,}  max={m.max():,}  均值={m.mean():.0f}")
P(f"  变异系数 CV = {m.std()/m.mean()*100:.1f}%")
P(f"  前3月: {[(str(k), int(v)) for k, v in m.head(3).items()]}")
P(f"  末3月: {[(str(k), int(v)) for k, v in m.tail(3).items()]}")

P("\n[审计 5] 注册时间只覆盖的时段（小时）")
hr = users["created_at"].dt.hour.value_counts().sort_index()
P(f"  00:00-18:59 每小时约 {hr.iloc[:19].mean():,.0f} 条")
P(f"  19 点: {hr.get(19,0):,} 条   20 点: {hr.get(20,0):,} 条   21-23 点: {hr.reindex(range(21,24), fill_value=0).sum():,} 条")

P("\n[审计 6] 时间戳精度混杂")
mic = users["created_at"].astype(str).str.contains(r"\.\d+")
P(f"  带微秒的行数: {mic.sum():,} ({mic.mean()*100:.2f}%)")
P(f"  这些行的时间范围: {users.loc[mic,'created_at'].min()} → {users.loc[mic,'created_at'].max()}")

P("\n[审计 7] 坐标粒度")
P(f"  唯一坐标点: {users[['latitude','longitude']].drop_duplicates().shape[0]:,} / {len(users):,} 人")
P(f"  平均每个坐标点对应 {len(users)/users[['latitude','longitude']].drop_duplicates().shape[0]:.1f} 人")

P("\n[审计 8] 邮箱域名")
domains = users["email"].str.split("@").str[1].value_counts()
P(f"  域名种类: {len(domains)}")
for k, v in domains.head(5).items():
    P(f"    @{k:<20}{v:>8,}")

P("\n[审计 9] 配送中心分布")
P(f"  数量: {len(dc)}   全部在美国城市: {'是' if dc['name'].str.contains(' TN| IL| TX| CA| LA| NY/NJ| PA| AL| SC| GA').all() else '否'}")

P("\n[审计 10] orders 关键时间字段缺失（缺失本身有业务含义）")
orders["created_at"] = pd.to_datetime(orders["created_at"], format="ISO8601", utc=True, errors="coerce")
for c in ["shipped_at", "delivered_at", "returned_at"]:
    P(f"  {c:<14} 缺失 {orders[c].isna().sum():>8,} ({orders[c].isna().mean()*100:>5.2f}%)")

with open(OUT / "01_data_check.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n>>> 已保存到 outputs/01_data_check.txt")
