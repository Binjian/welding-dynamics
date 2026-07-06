"""将焊接工艺参数数据库 xlsx 导入 MongoDB: welding_dynamics.welding_parameters

文档模型 (doc_type 区分):
- procedure     一条焊接工艺记录, passes 数组内嵌各焊道 (原始值 + 解析后数值)
- weave_pattern 摆动库路点波形
- source_meta   数据来源与录入规则 (备注表)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING

DATA = Path("/d/devel/sim/welding-dynamics/project_data/data/Welding Process Parameter Database 2022_rev.2022.03.24.xlsx")
MONGO = "mongodb://localhost:27017"
DB, COLL = "welding_dynamics", "welding_parameters"

# ---------- 解析 (与 notebooks/welding_parameter_database_exploration.ipynb 第2节一致) ----------
NUM = re.compile(r"\d+(?:\.\d+)?")

def parse_current(v):
    if pd.isna(v):
        return None
    s = str(v).strip().replace("（", "(").replace("）", ")")
    if not s or s == "*":
        return None
    m = re.search(r"\(([\d.]+)\s*A?\s*\)", s)      # 括号内 = 实际电流 (林肯: 主值为送丝设定)
    if m and float(m.group(1)) >= 50:
        return float(m.group(1))
    m = NUM.search(s.split("/")[0])                 # 双丝取主丝
    if not m:
        return None
    x = float(m.group(0))
    m2 = re.search(r"-\s*([\d.]+)", s.split("(")[0])  # 区间取中值
    if m2 and float(m2.group(1)) > x:
        x = (x + float(m2.group(1))) / 2
    return x

def parse_num(v):
    if pd.isna(v):
        return None
    m = NUM.search(str(v).strip().replace("（", "(").split("/")[0])
    return float(m.group(0)) if m else None

def clean(v):
    """xlsx 单元格 → BSON 兼容标量; 空/'*' → None"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer, np.floating)):
        v = v.item()
    if isinstance(v, str):
        v = v.strip()
        if v in ("", "*"):
            return None
    return v

def manufacturer(s):
    for key, name in [("林肯", "Lincoln"), ("奥太", "Aotai"), ("麦格米特", "Megmeet"),
                      ("福尼斯", "Fronius"), ("伏能士", "Fronius"), ("乐驰", "LeChi"),
                      ("乐弛", "LeChi"), ("松下", "Panasonic"), ("米加尼克", "Migatronic")]:
        if key in str(s):
            return name
    return "other"

# ---------- 读取 ----------
xls = pd.ExcelFile(DATA)
raw = xls.parse("焊接工艺参数", header=None)
df = raw.iloc[4:].reset_index(drop=True)
df["rec_id"] = df[0].notna().cumsum()
REC_COLS = list(range(0, 14)) + list(range(45, 54))
df[REC_COLS] = df[REC_COLS].ffill()

docs = []
for rid, g in df.groupby("rec_id"):
    r = g.iloc[0]
    passes = []
    for _, p in g.iterrows():
        I, U, v = parse_current(p[28]), parse_num(p[29]), parse_num(p[44])
        if U is not None and "一元化" in str(p[29]):
            U = None
        heat = round(U * I / v, 3) if all(x for x in (I, U, v)) else None  # J/mm, 未乘电弧效率
        passes.append({
            "pass_no": clean(p[14]),                       # 备注12: 单道=1; 多道根道=0, 盖面从1起
            "current_raw": clean(p[28]), "current_A": I,
            "voltage_raw": clean(p[29]), "voltage_V": U,
            "voltage_synergic": "一元化" in str(p[29]),
            "travel_speed_mm_s": v, "heat_input_J_mm": heat,
            "weave": {"basis": clean(p[40]), "frequency_raw": clean(p[41]),
                      "frequency_Hz": parse_num(p[41]), "amplitude_raw": clean(p[42]),
                      "amplitude_mm": parse_num(p[42]), "file_no": clean(p[43])},
            "arc_start": {"current": clean(p[24]), "voltage": clean(p[25]), "file_no": clean(p[26])},
            "arc_end": {"current": clean(p[32]), "voltage": clean(p[33]), "file_no": clean(p[34])},
            "burnback": {"mode": clean(p[36]), "workpoint": clean(p[37]),
                         "time": clean(p[38]), "file_no": clean(p[39])},
            "multipass_template": {"name": clean(p[13]), "offset_y": clean(p[17]),
                                   "offset_z": clean(p[18]), "start_offset": clean(p[21]),
                                   "end_offset": clean(p[22])},
        })
    docs.append({
        "doc_type": "procedure",
        "record_id": int(rid),
        "machine": {"model_raw": clean(r[0]), "manufacturer": manufacturer(r[0])},
        "base_metal": clean(r[1]),
        "wire": {"diameter_mm": clean(pd.to_numeric(r[2], errors="coerce")), "type": clean(r[3])},
        "shielding_gas": clean(r[4]),
        "joint": {"seam_type": clean(r[5]), "groove_size": clean(r[7]), "backing": clean(r[8]),
                  "position": clean(r[9]), "leg_size": clean(r[10]),
                  "rx_tilt_deg": clean(pd.to_numeric(r[11], errors="coerce")),
                  "ry_travel_deg": clean(pd.to_numeric(r[12], errors="coerce"))},
        "n_passes": len(passes),
        "passes": passes,
        "source": {"customer": clean(r[45]), "project_no": clean(r[46]),
                   "stickout_mm": clean(r[47]), "torch": clean(r[48]),
                   "workpiece": clean(r[49]), "submitter": clean(r[50]),
                   "reviewer": clean(r[51]), "entry_date": clean(r[52]), "note": clean(r[53])},
    })

# 摆动库
wl = xls.parse("摆动库参数", header=None).iloc[1:, :10]
wl.columns = ["pattern_id", "point", "time_pct", "x_pct", "y_pct", "z_pct",
              "angle_deg", "current_pct", "voltage_pct", "update"]
wl["pattern_id"] = wl["pattern_id"].ffill()
wl = wl.dropna(subset=["point"])
for pid, g in wl.groupby("pattern_id", sort=False):
    docs.append({
        "doc_type": "weave_pattern",
        "pattern_id": int(pid),
        "waypoints": [{k: clean(p[k]) for k in
                       ["point", "time_pct", "x_pct", "y_pct", "z_pct",
                        "angle_deg", "current_pct", "voltage_pct"]}
                      for _, p in g.iterrows()],
    })

# 备注 (录入规则)
notes = [str(x).strip() for x in xls.parse("备注", header=None).iloc[:, 0].dropna()]
docs.append({"doc_type": "source_meta", "source_file": DATA.name,
             "revision": "2022.03.24", "entry_notes": notes})

# ---------- 写入 ----------
client = MongoClient(MONGO, serverSelectionTimeoutMS=5000)
coll = client[DB][COLL]
coll.drop()  # 重复运行时幂等
coll.insert_many(docs)
coll.create_index([("doc_type", ASCENDING)])
coll.create_index([("base_metal", ASCENDING), ("machine.manufacturer", ASCENDING)])
coll.create_index([("passes.current_A", ASCENDING)])

n = {t: coll.count_documents({"doc_type": t}) for t in ["procedure", "weave_pattern", "source_meta"]}
total_passes = sum(d["n_passes"] for d in docs if d["doc_type"] == "procedure")
print(f"已写入 {DB}.{COLL}: {coll.count_documents({})} 文档 "
      f"(procedure={n['procedure']}, 共 {total_passes} 焊道; "
      f"weave_pattern={n['weave_pattern']}; source_meta={n['source_meta']})")
