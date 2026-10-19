# make_splits.py — 按"源视频/编舞"分组生成 train / val / test 切分 (杜绝数据泄漏)
#
# 分组键 = (genre, dance_id): 同一支编舞的全部机位/音乐/片段 (如 gBR_d04 的所有 ch/m 组合)
# 必须整体属于且只属于 train/val/test 之一, 绝不跨集 —— 从根上杜绝
# "同舞者 / 同服装 / 同灯光 / 同录制日 / 连续动作" 从切片层面泄漏。
#
# 旧 crossmodal_*.txt 按音乐(m)切分, 同一支编舞 d04 同时出现在 train 和 val, 属于泄漏, 已弃用。
# 输出: splits/grouped_train.txt, grouped_val.txt, grouped_test.txt (每行一个 seq 名, 不带 .mp4)
import os
import random
import re
from collections import defaultdict

SEARCH_DIR = "./dance_dataset_categorized"
OUT_DIR = "./splits"
# 按组三等分 (不是 70/15/15): 每个舞种每集的样本支持数对齐
# (舞种 3 舞者组 -> 1/1/1; gND 15 源组 -> 5/5/5, 各 ~45)
SEED = 42
DANCE_RE = re.compile(r"^d\d+$")  # 文件名形如 gBR_sBM_c01_d04_mBR1_ch03, d\d+ = 编舞/源表演 ID


def group_key(seq: str):
    """(genre, dance_id)。同一支编舞的所有片段共享此键。"""
    parts = seq.split("_")
    dance = next((p for p in parts if DANCE_RE.match(p)), None)
    return (parts[0], dance)


def main():
    seqs = []
    for root, _, files in os.walk(SEARCH_DIR):
        for f in sorted(files):
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                seqs.append(os.path.splitext(f)[0])

    groups = defaultdict(list)
    for s in seqs:
        genre, dance = group_key(s)
        if dance is None:
            raise SystemExit(f"[!] {s} 无法解析 dNN 组键, 拒绝切分 (会引入泄漏)")
        groups[(genre, dance)].append(s)

    # 按舞种分层, 组为单位分配
    by_genre = defaultdict(list)
    for g in groups:
        by_genre[g[0]].append(g)

    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    for genre in sorted(by_genre):
        glist = sorted(by_genre[genre])
        rng.shuffle(glist)

        # 按来源域分层: sBM = AIST++ 录影棚 / sYT = YouTube 真实舞者。
        # 每个域内按组体积贪心均衡分到三集 -> 每集同时含两类域的舞者,
        # 训练/评测域分布一致, 指标才可比 (曾出现 val 全影棚 / test 全 YouTube 的错位)。
        def domain(g):
            return "yt" if "_sYT_" in groups[g][0] else "aist"

        counts = {"train": 0, "val": 0, "test": 0}
        for dom in ("aist", "yt"):
            dom_groups = sorted([g for g in glist if domain(g) == dom],
                                key=lambda g: (-len(groups[g]), g))
            for g in dom_groups:
                target = min(("train", "val", "test"), key=lambda s: (counts[s], s))
                splits[target].extend(groups[g])
                counts[target] += len(groups[g])
        plan_str = "  ".join(f"{s}={counts[s]}片" for s in ("train", "val", "test"))
        print(f"[*] {genre}: {len(glist)} 组 -> {plan_str}")

    # 防泄漏断言: 任何组都不允许跨集
    seen = {}
    for split, seqs_ in splits.items():
        for s in seqs_:
            g = group_key(s)
            if g in seen and seen[g] != split:
                raise SystemExit(f"[!] 泄漏: 组 {g} 同时出现在 {seen[g]} 和 {split}")
            seen[g] = split

    os.makedirs(OUT_DIR, exist_ok=True)
    for split, seqs_ in splits.items():
        path = os.path.join(OUT_DIR, f"grouped_{split}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(seqs_)) + "\n")
        print(f"[*] {path}: {len(seqs_)} 行")

    # 各集合类别计数
    for split, seqs_ in splits.items():
        counts = defaultdict(int)
        for s in seqs_:
            counts[s.split("_")[0]] += 1
        print(f"    {split:<5} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("[*] 切分完成, 组级防泄漏断言通过")


if __name__ == "__main__":
    main()
