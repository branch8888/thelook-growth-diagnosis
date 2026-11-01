# -*- coding: utf-8 -*-
"""
把 TheLook_业务场景分析.md 转成可执行的 Jupyter Notebook
用法：python scripts/07_md_to_ipynb.py [--run]
"""
import re, sys
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "TheLook_业务场景分析.md"
OUT = ROOT / "TheLook_业务场景分析.ipynb"

BOOTSTRAP = '''# 环境准备：导入依赖、定位数据目录并载入 7 张表
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

ROOT = Path.cwd()
_cands = [ROOT / "data"] + sorted(ROOT.glob("**/data"))
DATA = next(p for p in _cands if (p / "users.csv").exists())

users                = pd.read_csv(DATA / "users.csv")
orders               = pd.read_csv(DATA / "orders.csv")
order_items          = pd.read_csv(DATA / "order_items.csv")
products             = pd.read_csv(DATA / "products.csv")
inventory_items      = pd.read_csv(DATA / "inventory_items.csv")
distribution_centers = pd.read_csv(DATA / "distribution_centers.csv")

# 别名，供正文代码块直接引用
dc = distribution_centers
inventory = inventory_items

# 订单明细连接商品维度：后续分析几乎都要 category / retail_price / cost
items = order_items.merge(
    products[["id", "category", "brand", "retail_price", "cost"]],
    left_on="product_id", right_on="id", how="left", suffixes=("", "_p"))

print(f"数据目录：{DATA}")
print(f"users {users.shape}  orders {orders.shape}  "
      f"order_items {order_items.shape}  products {products.shape}")'''


def split_md(chunk: str):
    """按一级标题切分 markdown，让每一章成为独立单元（代码围栏内不切）"""
    out, buf, in_fence = [], [], False
    for line in chunk.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        if not in_fence and line.startswith("# ") and any(b.strip() for b in buf):
            out.append("\n".join(buf).strip())
            buf = [line]
        else:
            buf.append(line)
    if any(b.strip() for b in buf):
        out.append("\n".join(buf).strip())
    return [c for c in out if c]


def md_to_notebook(md_text: str):
    # 与 md 严格一一对应：不额外插入任何单元
    cells = []

    fence = re.compile(r"```python\n(.*?)```", re.S)
    pos = 0
    for mt in fence.finditer(md_text):
        before = md_text[pos:mt.start()]
        if before.strip():
            cells.extend(new_markdown_cell(c) for c in split_md(before))
        cells.append(new_code_cell(mt.group(1).rstrip()))
        pos = mt.end()
    tail = md_text[pos:]
    if tail.strip():
        cells.extend(new_markdown_cell(c) for c in split_md(tail))
    return new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })


nb = md_to_notebook(MD.read_text(encoding="utf-8"))
nbf.write(nb, OUT)
print(f"已生成 {OUT.name}：{len(nb.cells)} 个单元 "
      f"（markdown {sum(c.cell_type=='markdown' for c in nb.cells)}，"
      f"code {sum(c.cell_type=='code' for c in nb.cells)}）")

if "--run" in sys.argv:
    import nbclient
    print("开始执行…")
    nb = nbf.read(OUT, as_version=4)
    nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                            allow_errors=True).execute()
    nbf.write(nb, OUT)
    errs = [(i, out) for i, c in enumerate(nb.cells) if c.cell_type == "code"
            for out in c.get("outputs", []) if out.get("output_type") == "error"]
    print(f"执行完成，错误单元数：{len(errs)}")
    for i, e in errs:
        print(f"  [cell {i}] {e.get('ename')}: {e.get('evalue')}")
