#!/bin/bash
# 复现全部实验。总耗时约 60–90 分钟（Mac M4）。
# 注意：多个实验同时使用 MPS 会互相阻塞，因此这里串行执行，
# 只有图像对抗实验走 CPU（它不依赖 MPS 加速，且能与其他实验并行）。
set -e
cd "$(dirname "$0")"
PY=".venv/bin/python"
mkdir -p logs results docs/figures

run() {
    printf "\n\033[1;34m==> %s\033[0m\n" "$1"; shift
    "$@" 2>&1 | grep -viE "^Using a slow|FutureWarning|UserWarning|^\s+warnings\.warn|it/s\]|rtf_avg" || true
}

run "1/8 水军检测（重建沙箱 + 消融 + 泛化 + 规避曲线）" \
    $PY -u experiments/exp_bot_detection.py
run "2/8 阈值校准" \
    $PY -u scripts_calibrate.py
run "3/8 文本审核基准（COLD + 合并场景）" \
    $PY -u experiments/exp_text_bench2.py --cold 1200 --spam 800
run "4/8 提示集与聚合方式网格搜索" \
    $PY -u experiments/exp_prompt_tuning.py --budget 24
run "5/8 视频审核基准（XD-Violence + 自建切片）" \
    $PY -u experiments/exp_video_bench.py --budget 24
run "6/8 深伪专家评估（同分布 / 分布外 / 压缩鲁棒性）" \
    $PY -u experiments/exp_deepfake_eval.py -n 250
run "7/8 对抗攻防（图像 / 语音 / 视频 / 聊天机器人）" \
    env SOCIALSIM_DEVICE=cpu $PY -u experiments/exp_adv_image.py -n 20 --steps 12
run "7b" $PY -u experiments/exp_adv_audio.py
run "7c" $PY -u experiments/exp_adv_video.py -n 3 --budget 20 --skip-pgd
run "7d" $PY -u experiments/exp_chatbot_guard.py
run "8/8 生成图表" \
    $PY -u experiments/make_figures.py

printf "\n\033[1;32m全部完成\033[0m\n"
echo "结果：results/*.json    图表：docs/figures/*.png"
