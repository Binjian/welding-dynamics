"""将 Hydra 配置树 src/welding_dynamics/conf/ 导入 MongoDB: welding_dynamics.welding_config

    uv run python project_data/ingest_config_mongo.py            # 幂等重建集合
    uv run python project_data/ingest_config_mongo.py --dry-run  # 只打印, 不写库

文档模型 (doc_type 区分):
- config_root      根配置 (sim / sim_vi / sim_3d): 原文、解析后的 dict、defaults 列表、入口点
- config_group     分组选项 (process/db_median, material/carbon_steel, model/gmaw, ...)
- config_composed  经 Hydra 组合 + 插值求值后的**最终配置** (仿真真正拿到的那份)
- config_meta      来源目录、git 提交、hydra 版本、各文件 sha256

前三类保留 `raw` 原文, 便于精确复现; `config_composed` 是 `${material.k}` /
`${wd.alpha:...}` 全部求值后的快照 —— 查询"某次参数研究到底用了什么数"看它即可。
"""
import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import hydra
import yaml
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from pymongo import MongoClient, ASCENDING

import welding_dynamics
import welding_dynamics.config  # noqa: F401  导入即注册 wd.* 解析器

CONF = Path(welding_dynamics.__file__).parent / "conf"
CONF_MODULE = "welding_dynamics.conf"
MONGO = "mongodb://localhost:27017"
DB, COLL = "welding_dynamics", "welding_config"

# 根配置 -> CLI 入口点
ENTRY_POINTS = {"sim": "welding-sim", "sim_vi": "welding-sim-vi",
                "sim_3d": "welding-sim-3d"}

# 要落库的组合: 带 process 分组的根配置按各工况预设各存一份 (参数研究的常用切面),
# sim_vi 无 process 分组, 只存默认组合。
# 注意 output=per_run 含 ${hydra:runtime.output_dir}, 脱离 hydra 运行期无法求值, 故不组合。
PROCESS_OPTIONS = ["code_default", "db_p10", "db_median", "db_p90"]
COMPOSITIONS = {
    "sim": [[f"process={p}"] for p in PROCESS_OPTIONS],
    "sim_vi": [[]],
    "sim_3d": [[f"process={p}"] for p in PROCESS_OPTIONS],
}


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_info():
    def run(*args):
        try:
            return subprocess.run(args, cwd=CONF, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--", str(CONF)))}


def check_keys(obj, path=""):
    """BSON 字段名不得含 '.' 或以 '$' 开头 —— 提前失败, 别写进半个集合。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if "." in k or k.startswith("$"):
                raise ValueError(f"非法 BSON 字段名 {k!r} (位于 {path or '/'})")
            check_keys(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_keys(v, f"{path}[{i}]")


def default_groups(defaults):
    """从根配置的 defaults 列表提取 {分组: 默认选项}, 忽略 _self_ 与 model@x 重定向。"""
    out = {}
    for item in defaults:
        if not isinstance(item, dict):
            continue
        for group, option in item.items():
            if "@" not in group:
                out[group] = option
    return out


def build_docs():
    docs = []
    files = {}                                   # 相对路径 -> sha256
    root_defaults = {}                           # 根配置名 -> {分组: 默认选项}

    # ---------- 根配置与分组选项 (原文 + 解析后的 dict) ----------
    for path in sorted(CONF.rglob("*.yaml")):
        rel = path.relative_to(CONF).as_posix()
        raw = path.read_text(encoding="utf-8")
        files[rel] = sha256(raw)
        parsed = yaml.safe_load(raw) or {}

        if path.parent == CONF:                  # sim.yaml / sim_vi.yaml / sim_3d.yaml
            name = path.stem
            root_defaults[name] = default_groups(parsed.get("defaults", []))
            docs.append({
                "doc_type": "config_root",
                "name": name,
                "entry_point": ENTRY_POINTS.get(name),
                "path": rel,
                "defaults": parsed.get("defaults", []),
                "config": {k: v for k, v in parsed.items() if k != "defaults"},
                "raw": raw,
                "sha256": files[rel],
            })
        else:                                    # <group>/<option>.yaml
            docs.append({
                "doc_type": "config_group",
                "group": path.parent.relative_to(CONF).as_posix(),
                "option": path.stem,
                "path": rel,
                # model/* 是 _target_ 节点; process/material/solver 带 name/label
                "target": parsed.get("_target_"),
                "label": parsed.get("label"),
                "config": parsed,
                "raw": raw,
                "sha256": files[rel],
            })

    # ---------- 组合 + 求值后的最终配置 ----------
    for root, override_sets in COMPOSITIONS.items():
        for overrides in override_sets:
            # 每次 compose 都要重新 initialize (GlobalHydra 是单例)
            with initialize_config_module(config_module=CONF_MODULE,
                                          version_base="1.3"):
                cfg = compose(config_name=root, overrides=list(overrides))
                resolved = OmegaConf.to_container(cfg, resolve=True)
            groups = {g: resolved[g]["name"]
                      for g in ("process", "material", "solver")
                      if isinstance(resolved.get(g), dict) and "name" in resolved[g]}
            base = root_defaults.get(root, {})
            docs.append({
                "doc_type": "config_composed",
                "root": root,
                "entry_point": ENTRY_POINTS.get(root),
                "overrides": list(overrides),
                "groups": groups,          # 便于按 process/material/solver 直接查询
                # 是否为该入口不加任何 override 时的组合 (即 README "典型结果" 那一组)
                "is_default": all(base.get(g) == opt for g, opt in groups.items()),
                "resolved": resolved,      # ${...} 已全部求值
            })

    # ---------- 元信息 ----------
    docs.append({
        "doc_type": "config_meta",
        "source_dir": str(CONF),
        "config_module": CONF_MODULE,
        "package_version": welding_dynamics.__version__,
        "hydra_version": hydra.__version__,
        "git": git_info(),
        "files": [{"path": p, "sha256": h} for p, h in sorted(files.items())],
        "n_files": len(files),
        "ingested_at": datetime.now(timezone.utc),
    })
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mongo", default=MONGO)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--collection", default=COLL)
    ap.add_argument("--dry-run", action="store_true", help="只打印统计, 不写库")
    args = ap.parse_args()

    docs = build_docs()
    for d in docs:
        check_keys(d)

    n = {}
    for d in docs:
        n[d["doc_type"]] = n.get(d["doc_type"], 0) + 1
    summary = (f"{len(docs)} 文档 (config_root={n.get('config_root', 0)}, "
               f"config_group={n.get('config_group', 0)}, "
               f"config_composed={n.get('config_composed', 0)}, "
               f"config_meta={n.get('config_meta', 0)})")

    if args.dry_run:
        print(f"[dry-run] 将写入 {args.db}.{args.collection}: {summary}")
        for d in docs:
            if d["doc_type"] == "config_group":
                print(f"  {d['group']}/{d['option']}")
            elif d["doc_type"] == "config_composed":
                print(f"  composed {d['root']} {d['overrides'] or '(默认)'} -> {d['groups']}")
        return

    coll = MongoClient(args.mongo, serverSelectionTimeoutMS=5000)[args.db][args.collection]
    coll.drop()                                   # 重复运行时幂等
    coll.insert_many(docs)
    coll.create_index([("doc_type", ASCENDING)])
    coll.create_index([("group", ASCENDING), ("option", ASCENDING)])
    coll.create_index([("root", ASCENDING), ("is_default", ASCENDING)])

    print(f"已写入 {args.db}.{args.collection}: {summary}")


if __name__ == "__main__":
    main()
