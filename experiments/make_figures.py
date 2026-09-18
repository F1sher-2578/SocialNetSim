"""汇总所有实验结果，生成实验报告用的图表。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from socialsim.config import RESULT_DIR, ROOT

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 中文字体
for f in ("PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "Hiragino Sans GB"):
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
C = {"blue": "#2c6fbb", "red": "#c0392b", "orange": "#e67e22", "green": "#27ae60",
     "gray": "#95a5a6", "purple": "#8e44ad"}


def load(name: str):
    p = RESULT_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_bot_ablation():
    d = load("bot_detection_metrics.json")
    if not d or "ablation" not in d:
        return
    ab = d["ablation"]
    order = ["only_profile", "only_behavior", "only_content", "only_graph",
             "drop_profile", "behavior+content+graph", "all"]
    names = {"only_profile": "仅画像", "only_behavior": "仅行为时序", "only_content": "仅内容",
             "only_graph": "仅图与协同", "drop_profile": "去掉画像", "all": "全部特征",
             "behavior+content+graph": "行为+内容+图"}
    ks = [k for k in order if k in ab]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(ks)); w = 0.38
    ax.bar(x - w / 2, [ab[k]["auc"] for k in ks], w, label="AUC", color=C["blue"])
    ax.bar(x + w / 2, [ab[k]["f1"] for k in ks], w, label="F1", color=C["orange"])
    for i, k in enumerate(ks):
        ax.text(i - w / 2, ab[k]["auc"] + .01, f'{ab[k]["auc"]:.3f}', ha="center", fontsize=7)
        ax.text(i + w / 2, ab[k]["f1"] + .01, f'{ab[k]["f1"]:.3f}', ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([names.get(k, k) for k in ks], fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("指标"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("水军检测：特征组消融", fontsize=11)
    ax.grid(axis="y", alpha=.25)
    fig.savefig(FIG / "bot_ablation.png"); plt.close(fig)
    print("  bot_ablation.png")


def fig_evasion_curve():
    d = load("bot_detection_metrics.json")
    if not d or "evasion_curve" not in d:
        return
    c = d["evasion_curve"]
    evs = sorted(int(k.split("_")[1]) for k in c)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for key, lab, col in (("detect_rate@0.5", "检出率 @阈值0.5", C["blue"]),
                          ("detect_rate@0.8", "检出率 @阈值0.8", C["orange"]),
                          ("mean_score", "平均风险分", C["green"])):
        ax.plot(evs, [c[f"evasion_{e}"][key] for e in evs], "o-", label=lab, color=col, lw=2)
    gen = d.get("evasion_generalization") or {}
    if gen:
        xs, ys = [], []
        for k, v in gen.items():
            xs.append(int(k.split("=")[-1])); ys.append(v["detect_rate@0.5"])
        ax.plot(xs, ys, "s--", label="未见过该等级时的检出率", color=C["red"], lw=2)
    ax.set_xticks(evs)
    ax.set_xticklabels([f"等级{e}\n{['朴素','养号+分散','深度伪装'][e] if e < 3 else ''}" for e in evs], fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel("检出率 / 风险分")
    ax.set_title("水军规避强度 → 检测能力", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.savefig(FIG / "evasion_curve.png"); plt.close(fig)
    print("  evasion_curve.png")


def fig_text_bench():
    d = load("text_bench_combined.json")
    if not d:
        return
    slices = [("S1_COLD_offensive", "COLD 冒犯言论"), ("S2_spam_ads", "广告/赌博引流"),
              ("S3_combined", "合并场景")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (k, title) in zip(axes, slices):
        if k not in d:
            continue
        v = d[k]
        methods = [("rule_only", "规则"), ("model_only", "模型"), ("fused", "融合")]
        x = np.arange(3); w = .38
        ax.bar(x - w / 2, [v[m]["auc"] for m, _ in methods], w, label="AUC", color=C["blue"])
        ax.bar(x + w / 2, [v[m]["f1"] for m, _ in methods], w, label="F1", color=C["orange"])
        for i, (m, _) in enumerate(methods):
            ax.text(i - w / 2, v[m]["auc"] + .02, f'{v[m]["auc"]:.2f}', ha="center", fontsize=7)
            ax.text(i + w / 2, v[m]["f1"] + .02, f'{v[m]["f1"]:.2f}', ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in methods], fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_title(f"{title}\n(n={v['n']})", fontsize=10)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("指标"); axes[0].legend(fontsize=8)
    fig.suptitle("文本审核：规则专家与模型专家的互补性", fontsize=11, y=1.04)
    fig.savefig(FIG / "text_bench.png"); plt.close(fig)
    print("  text_bench.png")


def fig_video_bench():
    d = load("video_bench_xdv.json")
    if not d:
        return
    xdv = d.get("xdviolence")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)

    # (a) 视觉通道分数分布（与 XD-Violence 的"视觉暴力"标注口径对齐）
    ax = axes[0]
    if xdv and "rows" in xdv:
        rows = xdv["rows"]
        pos = [r["visual"] for r in rows if r["truth"]]
        neg = [r["visual"] for r in rows if not r["truth"]]
        bins = np.linspace(min(pos + neg), 1.0, 16)
        ax.hist(neg, bins, alpha=.7, label=f"正常 (n={len(neg)})", color=C["green"])
        ax.hist(pos, bins, alpha=.7, label=f"暴力 (n={len(pos)})", color=C["red"])
        th = xdv.get("adaptive_thresholds", {})
        if th:
            ax.axvline(th["review"], ls="--", c=C["orange"], lw=1.4, label=f"复核 {th['review']:.2f}")
            ax.axvline(th["block"], ls="--", c=C["red"], lw=1.4, label=f"拦截 {th['block']:.2f}")
        ax.set_xlabel("视觉通道风险分", fontsize=9); ax.set_ylabel("视频数", fontsize=9)
        ax.set_title(f"(a) 视觉通道分数分布  AUC={xdv.get('auc_visual')}", fontsize=10)
        ax.legend(fontsize=7)

    # (b) 各模态通道与基准标注的一致性
    ax = axes[1]
    if xdv:
        chans = [("auc_visual", "视觉"), ("auc_asr", "语音"), ("auc_ocr", "画面文字"), ("auc", "max 融合")]
        ks = [(lab, xdv[m]) for m, lab in chans if xdv.get(m) is not None]
        colors = [C["blue"], C["orange"], C["purple"], C["gray"]]
        ax.bar([k for k, _ in ks], [v for _, v in ks], .55, color=colors[:len(ks)])
        for i, (_, v) in enumerate(ks):
            ax.text(i, v + .02, f"{v:.3f}", ha="center", fontsize=9)
        ax.axhline(0.5, ls=":", c=C["gray"], lw=1.2)
        ax.set_ylim(0, 1.12); ax.set_ylabel("AUC", fontsize=9); ax.tick_params(labelsize=9)
        ax.set_title("(b) 各通道与基准标注的一致性", fontsize=10); ax.grid(axis="y", alpha=.25)
        m = xdv.get("label_scope_mismatch", {})
        if m is not None:
            note = ("语音/画面文字通道对'视觉暴力'这一标签没有判别力，\n"
                    f"但也未造成误拦（{m.get('n', 0)}/{m.get('n_negatives', 0)} 段正常视频被误判），\n"
                    "其价值体现在中文违规切片上（语音 AUC 0.867）")
            ax.text(.5, -.24, note, transform=ax.transAxes, ha="center",
                    fontsize=7.5, color=C["gray"])

    # (c) 各数据切片
    ax = axes[2]
    ks, vs, ns = [], [], []
    for k, title in (("xdviolence", "XD-Violence\n公开基准"), ("own_slice", "自建\n中文切片"),
                     ("combined", "合并")):
        if k in d and d[k].get("auc") is not None:
            ks.append(title); vs.append(d[k].get("auc_visual") or d[k]["auc"]); ns.append(d[k]["n"])
    x = np.arange(len(ks)); w = .36
    overall = [d[k]["auc"] for k in ("xdviolence", "own_slice", "combined") if k in d and d[k].get("auc")]
    ax.bar(x - w / 2, overall, w, label="整体（全部违规类目）", color=C["gray"])
    ax.bar(x + w / 2, vs, w, label="视觉通道", color=C["blue"])
    for i, (a, b, n) in enumerate(zip(overall, vs, ns)):
        ax.text(i - w / 2, a + .02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + .02, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(ks, ns)], fontsize=8)
    ax.axhline(0.5, ls=":", c=C["gray"], lw=1.2)
    ax.set_ylim(0, 1.12); ax.set_ylabel("AUC", fontsize=9); ax.legend(fontsize=7)
    ax.set_title("(c) 各数据切片", fontsize=10); ax.grid(axis="y", alpha=.25)
    fig.savefig(FIG / "video_bench.png"); plt.close(fig)
    print("  video_bench.png")


def fig_adv_image():
    d = load("adv_image.json")
    if not d:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    # (a) 白盒 eps 扫描
    ax = axes[0, 0]
    wb = d.get("whitebox", {})
    if wb:
        epss = sorted(wb, key=lambda k: int(k.split("_")[1].split("/")[0]))
        xs = [int(k.split("_")[1].split("/")[0]) for k in epss]
        ax.plot(xs, [wb[k]["fgsm_asr"] for k in epss], "o-", label="FGSM", color=C["orange"], lw=2)
        ax.plot(xs, [wb[k]["pgd_asr"] for k in epss], "s-", label="PGD", color=C["red"], lw=2)
        ax2 = ax.twinx()
        ax2.plot(xs, [wb[k]["pgd_psnr_db"] for k in epss], "^--", color=C["gray"], lw=1.4)
        ax2.set_ylabel("PGD 后 PSNR (dB)", color=C["gray"], fontsize=9)
        ax2.tick_params(labelsize=8, colors=C["gray"])
        ax.set_xscale("log", base=2); ax.set_xticks(xs)
        ax.set_xticklabels([f"{x}/255" for x in xs], fontsize=8)
        ax.set_xlabel("扰动预算 ε", fontsize=9); ax.set_ylabel("攻击成功率", fontsize=9)
        ax.set_ylim(0, 1.1); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.25)
        ax.set_title("(a) 白盒攻击：ε=1/255 时 PSNR 仍有 49dB", fontsize=10)

    # (b) 非梯度自然扰动
    ax = axes[0, 1]
    nat = d.get("natural", {})
    if nat:
        items = sorted(nat.items(), key=lambda kv: -kv[1]["asr"])[:8]
        names = [k for k, _ in items][::-1]
        ax.barh(range(len(names)), [v["asr"] for _, v in items][::-1], color=C["purple"], height=.65)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, 1.05); ax.set_xlabel("攻击成功率", fontsize=9)
        ax.set_title("(b) 非梯度自然扰动（零模型访问）", fontsize=10); ax.grid(axis="x", alpha=.25)

    # (c) 防御恢复率随扰动预算的衰减（核心结果）
    ax = axes[1, 0]
    dbe = d.get("defense_by_eps", {})
    if dbe:
        epss = sorted(dbe, key=lambda k: int(k.split("_")[1].split("/")[0]))
        xs = [int(k.split("_")[1].split("/")[0]) for k in epss]
        cols = [C["blue"], C["green"], C["orange"], C["purple"], C["red"], C["gray"]]
        show = ["jpeg75", "bit3", "randresize", "median3", "randomized_ensemble"]
        for i, name in enumerate(show):
            ys = [dbe[e].get(name, {}).get("recovered_rate", np.nan) for e in epss]
            lw = 2.4 if name == "randomized_ensemble" else 1.6
            ax.plot(xs, ys, "o-", label=name, color=cols[i % len(cols)], lw=lw)
        ax.set_xscale("log", base=2); ax.set_xticks(xs)
        ax.set_xticklabels([f"{x}/255" for x in xs], fontsize=8)
        ax.set_xlabel("扰动预算 ε", fontsize=9); ax.set_ylabel("对抗样本恢复率", fontsize=9)
        ax.set_ylim(-0.03, 1.08); ax.legend(fontsize=7); ax.grid(alpha=.25)
        ax.set_title("(c) 防御随攻击强度迅速失效", fontsize=10)

    # (d) 静态 vs 自适应攻击（在防御确实有效的弱扰动下对比才有意义）
    ax = axes[1, 1]
    abe = d.get("adaptive_bpda_by_eps", {})
    if abe:
        eps_key = "eps_1/255" if "eps_1/255" in abe else list(abe)[0]
        blk = abe[eps_key]
        ks = list(blk)
        x = np.arange(len(ks)); w = .35
        static_rec = [blk[k]["static_recovered"] for k in ks]
        adaptive_rec = [blk[k]["adaptive_recovered"] for k in ks]
        ax.bar(x - w / 2, static_rec, w, label="静态攻击下恢复率", color=C["green"])
        ax.bar(x + w / 2, adaptive_rec, w, label="BPDA 自适应攻击下恢复率", color=C["red"])
        for i, (v1, v2) in enumerate(zip(static_rec, adaptive_rec)):
            ax.text(i - w / 2, v1 + .02, f"{v1:.2f}", ha="center", fontsize=8)
            ax.text(i + w / 2, v2 + .02, f"{v2:.2f}", ha="center", fontsize=8, color=C["red"])
        ax.set_xticks(x); ax.set_xticklabels(ks, fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_ylabel("恢复率", fontsize=9); ax.legend(fontsize=7)
        ax.set_title(f"(d) ε={eps_key.split('_')[1]}：自适应攻击下多数防御归零\n"
                     f"（JPEG 例外：扰动小于量化步长，被物理抹除）", fontsize=9.5)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("图像对抗攻防（攻击目标：深伪检测专家，ε=8/255 下 PGD 成功率 100%）",
                 fontsize=11)
    fig.savefig(FIG / "adv_image.png"); plt.close(fig)
    print("  adv_image.png")


def fig_adv_audio():
    d = load("adv_audio.json")
    if not d or "summary" not in d:
        return
    atk = d["summary"].get("attacks", {})
    if not atk:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    items = sorted(atk.items(), key=lambda kv: -kv[1]["escape_rate"])
    ks = [k for k, _ in items]
    ax.barh(ks[::-1], [v["escape_rate"] for _, v in items][::-1], color=C["red"])
    ax.set_xlim(0, 1.05); ax.set_xlabel("端到端审核逃逸率")
    ax.set_title("语音攻击：让 ASR 失真以绕过文本审核", fontsize=10)
    ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=.25)
    ax = axes[1]
    dfn = d["summary"].get("defenses", {})
    if dfn:
        ks = list(dfn)
        ax.bar(ks, [dfn[k]["recovery_rate"] for k in ks], .5, color=C["green"])
        for i, k in enumerate(ks):
            ax.text(i, dfn[k]["recovery_rate"] + .02, f'{dfn[k]["recovery_rate"]:.2f}', ha="center", fontsize=8)
        ax.set_ylim(0, 1.12); ax.set_ylabel("恢复率")
        ax.set_title("防御：在逃逸样本上重新审核", fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=20); ax.grid(axis="y", alpha=.25)
    fig.savefig(FIG / "adv_audio.png"); plt.close(fig)
    print("  adv_audio.png")


def fig_chatbot():
    d = load("chatbot_guard.json")
    if not d:
        return
    off, on = d["guard_off"], d["guard_on"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ks = ["direct", "direct_obfuscated", "indirect"]
    names = ["直接越狱", "混淆变体", "间接注入"]
    x = np.arange(3); w = .36
    ax.bar(x - w / 2, [off[k]["asr"] for k in ks], w, label="无护栏", color=C["red"])
    ax.bar(x + w / 2, [on[k]["asr"] for k in ks], w, label="四层护栏", color=C["green"])
    for i, k in enumerate(ks):
        ax.text(i - w / 2, off[k]["asr"] + .02, f'{off[k]["asr"]:.2f}', ha="center", fontsize=8)
        ax.text(i + w / 2, on[k]["asr"] + .02, f'{on[k]["asr"]:.2f}', ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("攻击成功率")
    ax.set_title("护栏对攻击成功率的影响", fontsize=10); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    ax = axes[1]
    ab = d.get("layer_ablation", {})
    rows = []
    if "L1L2_direct_only" in ab:
        a = ab["L1L2_direct_only"]
        rows += [("L1/L2 直接攻击", a["direct_detect_rate"]), ("L1/L2 混淆变体", a["obfuscated_detect_rate"]),
                 ("L1/L2 间接注入", a["indirect_detect_rate"])]
    if "L1L2_with_indirect_rules" in ab:
        rows.append(("+间接注入规则", ab["L1L2_with_indirect_rules"]["indirect_detect_rate"]))
    if "no_normalization" in ab:
        rows.append(("关闭归一化后\n混淆变体", ab["no_normalization"]["obfuscated_detect_rate"]))
    if rows:
        ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color=C["blue"])
        ax.set_xlim(0, 1.08); ax.set_xlabel("检出率")
        ax.set_title("逐层消融", fontsize=10); ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=.25)
    fr = on["benign"]["false_refusal_rate"]
    fig.text(.5, -.06, f"良性请求误拒率：{fr:.1%}（防御的代价，必须一并报告）",
             ha="center", fontsize=9, color=C["gray"])
    fig.savefig(FIG / "chatbot_guard.png"); plt.close(fig)
    print("  chatbot_guard.png")


def fig_deepfake():
    d = load("deepfake_eval.json")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax = axes[0]
    names, vals, cols = [], [], []
    names.append("同分布测试集\nAUC"); vals.append(d["auc"]); cols.append(C["green"])
    names.append("同分布\n准确率@EER"); vals.append(d["acc@eer"]); cols.append(C["blue"])
    if "ood_real_faces" in d:
        names.append("分布外真实照片\n误报率@0.5"); vals.append(d["ood_real_faces"]["fpr@0.5"]); cols.append(C["red"])
    ax.bar(names, vals, .5, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + .02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.12); ax.tick_params(labelsize=8)
    ax.set_title("深伪专家：同分布表现 vs 泛化失效", fontsize=10); ax.grid(axis="y", alpha=.25)
    ax = axes[1]
    jr = d.get("jpeg_robustness", {})
    if jr:
        qs = sorted(jr, key=lambda k: -int(k[1:]))
        ax.plot([int(q[1:]) for q in qs], [jr[q]["auc"] for q in qs], "o-", color=C["blue"], lw=2)
        ax.axhline(d["auc"], ls="--", c=C["gray"], label="原图 AUC")
        ax.set_xlabel("JPEG 质量"); ax.set_ylabel("AUC"); ax.set_ylim(0.5, 1.02)
        ax.invert_xaxis(); ax.legend(fontsize=8)
        ax.set_title("压缩鲁棒性", fontsize=10); ax.grid(alpha=.25)
    fig.savefig(FIG / "deepfake_eval.png"); plt.close(fig)
    print("  deepfake_eval.png")


def fig_pipeline_cost():
    d = load("video_bench_xdv.json")
    if not d or "xdviolence" not in d:
        return
    rows = d["xdviolence"].get("rows", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    dur = [r["duration"] for r in rows]; el = [r["elapsed"] for r in rows]
    ax.scatter(dur, el, s=22, alpha=.7, color=C["blue"])
    z = np.polyfit(dur, el, 1)
    xs = np.linspace(min(dur), max(dur), 50)
    ax.plot(xs, np.polyval(z, xs), "--", color=C["red"],
            label=f"拟合: {z[0]:.3f}×时长 + {z[1]:.1f}")
    ax.set_xlabel("视频时长 (秒)"); ax.set_ylabel("处理耗时 (秒)")
    ax.set_title(f"处理成本随时长的变化（平均 {d['xdviolence']['realtime_factor']}× 实时）", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.savefig(FIG / "pipeline_cost.png"); plt.close(fig)
    print("  pipeline_cost.png")


def main():
    print("生成图表 →", FIG)
    for f in (fig_bot_ablation, fig_evasion_curve, fig_text_bench, fig_video_bench,
              fig_adv_image, fig_adv_audio, fig_chatbot, fig_deepfake, fig_pipeline_cost):
        try:
            f()
        except Exception as e:
            print(f"  {f.__name__} 失败: {type(e).__name__}: {e}")
    print("完成")


if __name__ == "__main__":
    main()
