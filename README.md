# SocialNetSim · 社交网络拟真沙箱

多模态信息处理综合实验 —— **社交网络拟真开发**

一个完全本地运行的仿真社交平台，实现水军账号模拟与检测、(长/短)视频内容解析与不良内容监测，
并在此基础上完成文本 / 图像 / 视频 / 语音四个模态的 AI 安全对抗攻防，以及基于 AI 安全的智能聊天机器人。

> 所有功能仅在本机 SQLite 沙箱内运行，不与任何真实社交平台交互，不使用任何真实违规素材。

![信息流](docs/screenshots/feed.png)

<table>
<tr>
<td width="50%"><img src="docs/screenshots/detect.png" alt="水军检测"><br><sub>水军检测：特征消融与跨团伙泛化</sub></td>
<td width="50%"><img src="docs/screenshots/chat.png" alt="安全助手"><br><sub>安全助手：提示注入攻防演示</sub></td>
</tr>
</table>

**三个最关键的结论**（详见[实验报告](docs/实验报告.md)）：

1. 水军检测 AUC 1.000 看着完美，但**换一种没见过的规避手法，检出率直接掉到 0**。
2. 深伪检测模型在自己的测试集上 AUC 0.999，**换一批真实照片就有 72% 被误判成伪造**。
3. 图像防御在攻击者知情的情况下**全部失效**，唯一例外是扰动小到被 JPEG 压缩物理抹掉时。

---

## 一、功能对照实验要求

| 实验要求 | 本项目实现 | 代码位置 |
|---|---|---|
| **基本功能**：实现基本水军账号的功能 | 5 种水军策略（控评刷量 / 抹黑 / 话题带节奏 / 互粉抱团 / 广告引流）× 3 级规避强度，生成带真值标签的协同行为 | `socialsim/simulation/bot_engine.py` |
| **基本功能**：(长/短)视频内容解析，不良内容监测 | 快慢双通道关键帧选择 + 帧级多专家 + ASR 语音 + OCR 画面文字 + 时序聚合 + 可视化报告 | `socialsim/moderation/video_pipeline.py` |
| **扩展**：结合 AI Security 的对抗实例 | 文本混淆攻击、图像 FGSM/PGD/SPSA/BPDA、视频时序稀疏攻击、语音端到端逃逸攻击 | `socialsim/security/attacks/` |
| **扩展**：对文本、图片、视频等的攻击及防御 | 归一化防御、输入变换、随机化集成、特征压缩检测、自适应攻击评估 | `socialsim/security/defenses/` |
| **扩展**：对声音的攻击及防御 | 攻击审核流水线的语音识别环节（13 种攻击 × 5 种防御，按端到端审核逃逸计成功率） | `socialsim/security/attacks/audio_attack.py` |
| **扩展**：基于 AI 安全的智能聊天机器人 | 四层纵深护栏（输入检测 / 注入识别 / Spotlighting 隔离 / 输出守卫），含间接注入防御 | `socialsim/security/guard.py`、`chatbot.py` |

---

## 二、在你的电脑上跑起来

### 2.1 先装好这三样

| 依赖 | 作用 | macOS | Windows | Ubuntu |
|---|---|---|---|---|
| **Python 3.10–3.12** | 运行环境 | `brew install python@3.11` | [官网下载](https://www.python.org/downloads/)，安装时勾选 *Add to PATH* | `sudo apt install python3.11 python3.11-venv` |
| **ffmpeg** | 视频解码、抽音轨（必需） | `brew install ffmpeg` | `winget install ffmpeg` | `sudo apt install ffmpeg` |
| **git** | 拉代码 | 自带 | [git-scm.com](https://git-scm.com/) | `sudo apt install git` |

验证一下，三条命令都要有输出：

```bash
python3 --version
ffmpeg -version
git --version
```

还需要：**约 4 GB 磁盘**（依赖 1.2G + 模型 1.5G + 数据 1.3G）和**能联网**（首次要下模型和数据）。

> **Windows 用户注意**：`setup.sh` 是 bash 脚本，请在 **Git Bash**（装 git 时自带）或 **WSL** 里运行，
> 不要用 CMD/PowerShell。如果不想用 bash，见下面 2.4 的手动步骤。

---

### 2.2 三步跑起来

```bash
git clone https://github.com/F1sher-2578/SocialNetSim.git
cd SocialNetSim
bash setup.sh
```

`setup.sh` 会自动做完 7 件事，**全程约 20–40 分钟**（主要时间花在下模型和数据上）：

| 步骤 | 内容 | 大约耗时 |
|---|---|---|
| 0 | 检查 python/ffmpeg/磁盘空间，缺什么会直接告诉你怎么装 | 数秒 |
| 1–2 | 建虚拟环境 `.venv/`，装依赖 | 2–5 分钟 |
| 3 | 下载 8 个预训练模型（走 hf-mirror 镜像，国内可直连） | 5–15 分钟 |
| 4 | 下载数据集：COLD 中文语料、真假人脸、CIFAR-10、XD-Violence 视频子集 | 10–20 分钟 |
| 5 | 生成仿真沙箱：80 个正常用户 + 11 个水军团伙的 30 天行为 | 约 1 分钟 |
| 6 | 阈值校准：在正常素材上估计各专家的分数分布 | 3–5 分钟 |
| 7 | 自检：逐项验证文本审核、语音、护栏、信息流是否正常 | 数秒 |

看到最后打印 **`全部通过`** 就说明装好了。

然后启动界面：

```bash
./run_server.sh
```

浏览器打开 **http://127.0.0.1:8000** 。

> 数据集**没有**放进仓库（1.3 GB，且 XD-Violence 是有版权的电影片段不能转发），
> 所以必须跑一次 `setup.sh` 才能有数据。代码、实验结果、图表都在仓库里，可以直接看。

---

### 2.3 界面上可以点什么

六个页面，建议按这个顺序看：

| 页面 | 能做什么 | 推荐动作 |
|---|---|---|
| **信息流** | 看仿真出来的社交平台，水军账号带红色标记 | 在发布框输入 `加 微 信 xk8899 领取内部优惠`，看审核如何拦截 |
| **水军控制台** | 发起新的水军任务，选策略、账号数、**规避等级** | 用规避等级 0 和 2 各发一次，对比检测分数差异 |
| **水军检测** | 跑检测模型，看消融、泛化、特征重要性 | 点「重新提取特征并评估」，看跨规避等级泛化那一栏 |
| **内容审核** | 上传图片/视频做多模态审核 | 上传 `data/videos/bench/gambling_short.mp4`，会生成可视化报告 |
| **安全助手** | 提示注入攻防演示 | **关掉**护栏开关，点左侧攻击样例发送，再打开护栏重发一次对比 |
| **攻防实验** | 汇总展示所有实验数据 | 直接看，不用跑 |

---

### 2.4 不想用 setup.sh 的话（手动步骤）

```bash
# 1. 虚拟环境
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. 依赖（国内建议加清华源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyarrow funasr -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 模型（逐个下载，HF_ENDPOINT 指向镜像）
export HF_ENDPOINT=https://hf-mirror.com     # Windows CMD: set HF_ENDPOINT=https://hf-mirror.com
python scripts_download_models.py Falconsai/nsfw_image_detection
python scripts_download_models.py openai/clip-vit-base-patch32
python scripts_download_models.py thu-coai/roberta-base-cold
python scripts_download_models.py dima806/deepfake_vs_real_image_detection
python scripts_download_models.py openai/whisper-base

# 4. 数据
python scripts_fetch_xdviolence.py --violent 20 --normal 20   # 想省流量可以调小
python scripts_make_testset.py

# 5. 沙箱 + 校准
python experiments/exp_bot_detection.py
python scripts_calibrate.py

# 6. 启动
python -m uvicorn socialsim.app.main:app --host 127.0.0.1 --port 8000
```

---

### 2.5 常见问题

**装依赖很慢或超时**
国内网络请加清华源：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

**模型下载失败 / 连不上 huggingface.co**
本项目默认走镜像 `https://hf-mirror.com`，已在 `socialsim/__init__.py` 里设置。
如果仍失败，手动导出后重试：`export HF_ENDPOINT=https://hf-mirror.com`。

**提示 `No module named 'socialsim'`**
在项目根目录运行，或用 `.venv/bin/python` 而不是系统 python。

**提示 `ffmpeg not found`**
ffmpeg 没装或没进 PATH。Windows 装完要重开终端。

**Web 界面打开是空的 / 信息流没内容**
沙箱数据库没生成。跑一次 `python experiments/exp_bot_detection.py` 重建。

**非 macOS，合成测试视频没有声音**
语音合成在 macOS 上用系统自带的 `say`。其他系统装一个即可：
`pip install edge-tts`（联网，中文效果好）或 `pip install pyttsx3`（离线）。

**端口 8000 被占用**
`./run_server.sh --port 8001`，然后访问 8001。

**没有 GPU 能跑吗**
能。全部模型都可在 CPU 上推理，本项目就是在 Mac M4 的 CPU/MPS 上完成的。
Windows/Linux 上会自动回退到 CPU，视频审核会慢一些但不影响结果。
多个实验同时跑会争抢显存，可用 `SOCIALSIM_DEVICE=cpu` 强制其中一个走 CPU。

---

## 三、目录结构

```
SocialNetSim/
├── socialsim/
│   ├── config.py                    全局配置（路径、模型 id、阈值、设备）
│   ├── db.py / models.py            SQLite ORM：用户/帖子/评论/点赞/关注/行为日志/审核记录
│   ├── simulation/
│   │   ├── personas.py              人设生成（正常用户 / 水军 / 伪装水军）
│   │   ├── content_gen.py           内容生成（正常模板 / 水军模板 / 变体改写）
│   │   ├── seed.py                  30 天基线行为生成（昼夜节律 + 热度偏好）
│   │   ├── bot_engine.py            水军任务引擎（5 策略 × 3 规避等级）
│   │   ├── bot_detector.py          水军检测（41 维 / 五特征组 / 消融 / 泛化 / 无监督协同）
│   │   └── feed.py                  信息流排序与曝光影响量化
│   ├── moderation/
│   │   ├── text_filter.py           文本审核（规则词表 + COLD 模型 + 混淆归一化）
│   │   ├── image_experts.py         图像专家（NSFW / CLIP 零样本 / 人脸深伪 / 画面域判别）
│   │   ├── asr.py                   语音理解（SenseVoice：ASR + 情绪 + 音频事件，回退 Whisper）
│   │   ├── video_pipeline.py        视频流水线（关键帧 / 多专家 / ASR / OCR / 聚合 / 报告）
│   │   └── pipeline.py              统一审核调度与落库
│   ├── security/
│   │   ├── guard.py                 LLM 四层安全护栏
│   │   ├── chatbot.py               内容安全助手（可开关护栏做 A/B 对照）
│   │   ├── attacks/                 图像 / 视频 / 文本 / 语音 对抗攻击
│   │   └── defenses/                对应防御与对抗样本检测
│   └── app/                         FastAPI 服务与前端页面
├── experiments/                     所有可复现实验脚本
├── scripts_*.py                     数据与模型准备脚本
├── docs/                            前沿调研报告 + 实验报告
└── results/                         实验结果 JSON、图表、审核报告
```

---

## 四、数据集

| 用途 | 数据集 | 规模 | 获取方式 |
|---|---|---|---|
| 视频不良内容（主基准） | **XD-Violence** 官方测试集子集 | 82 段，含 41 段帧级标注 | `scripts_fetch_xdviolence.py`（HF 镜像分层抽样） |
| 中文冒犯性文本 | **COLD** (EMNLP 2022) | 测试集 5,323 条 | `setup.sh` 自动下载 |
| 真假人脸 | Deepfake-and-real-images | 测试集 10,905 张 | `setup.sh` 自动下载 |
| 对抗攻击载体 | CIFAR-10 | 测试集 10,000 张 | `setup.sh` 自动下载 |
| 风险类目视觉素材 | Wikimedia Commons 知识共享图片 | 约 40 张（武器/火灾/人群/吸烟/肖像/正常） | `scripts_make_testset.py` |
| 自建中文违规切片 | 合成视频（语音由 macOS TTS 生成） | 8 段带真值标签 | `scripts_make_testset.py` |

自建切片的作用是覆盖 XD-Violence 不包含的违规类型（中文语音广告引流、赌博推广、画面烧录联系方式），
不替代公开基准。所有结论以公开基准为准，自建切片仅作补充说明。

---

## 五、模型

全部为开源预训练模型，本地推理，无需 API：

| 角色 | 模型 | 说明 |
|---|---|---|
| 中文冒犯语言 | `thu-coai/roberta-base-cold` | COLD 官方模型 |
| NSFW 图像 | `Falconsai/nsfw_image_detection` | ViT-base |
| 零样本风险类目 | `openai/clip-vit-base-patch32` | 提示集成 + 画面域判别 |
| 人脸深伪 | `dima806/deepfake_vs_real_image_detection` | ViT-base（**泛化性存在严重问题，见实验报告**） |
| 人脸检测 | OpenCV YuNet | ONNX，CPU 实时 |
| 语音理解 | `FunAudioLLM/SenseVoiceSmall` | ASR + 情绪 + 音频事件；中文优于 Whisper，CPU 约 10× 实时 |
| 语音识别（回退） | `openai/whisper-base` | 含能量门限与防幻觉后处理 |
| 画面文字 | RapidOCR (ONNX) | 中英文 |
| 提示注入检测 | `protectai/deberta-v3-base-prompt-injection-v2` | 护栏 L2 层 |

国内网络通过 `HF_ENDPOINT=https://hf-mirror.com` 镜像下载，已在 `socialsim/__init__.py` 中设置。

---

## 六、复现实验

一条命令跑完全部（约 60–90 分钟）：

```bash
./run_all_experiments.sh
```

或单独运行：

```bash
# 水军检测：重建沙箱 + 特征消融 + 跨团伙/跨规避等级泛化 + 无监督协同
.venv/bin/python experiments/exp_bot_detection.py

# 阈值校准（更换素材域后必须重跑）
.venv/bin/python scripts_calibrate.py

# 文本审核：COLD 基准 + 合并基准 + 混淆攻击
.venv/bin/python experiments/exp_text_bench2.py

# 提示集与聚合方式网格搜索（缓存 CLIP 特征后可快速迭代）
.venv/bin/python experiments/exp_prompt_tuning.py

# 视频审核：XD-Violence 主基准 + 自建切片
.venv/bin/python experiments/exp_video_bench.py

# 深伪专家评估：同分布 vs 分布外 + JPEG 鲁棒性
.venv/bin/python experiments/exp_deepfake_eval.py

# 对抗攻防（图像建议走 CPU，避免与其他实验争抢 MPS）
SOCIALSIM_DEVICE=cpu .venv/bin/python experiments/exp_adv_image.py
.venv/bin/python experiments/exp_adv_audio.py
.venv/bin/python experiments/exp_adv_video.py
.venv/bin/python experiments/exp_chatbot_guard.py

# 生成全部图表
.venv/bin/python experiments/make_figures.py
```

结果写入 `results/*.json`，图表写入 `docs/figures/`。

---

## 七、主要实验结果

| 模块 | 数据 | 结果 |
|---|---|---|
| 水军检测 | 194 账号 / 11 团伙 | AUC 1.000，Recall@FPR=1% 为 1.000，团伙级召回 11/11 |
| 水军检测泛化 | 跨规避等级 | 训练时未见过养号型水军时，检出率 **0.000** |
| 无监督协同发现 | 无标签 | 5 个社区，纯度 100%，覆盖 17.5% 水军 |
| 视频审核 | XD-Violence 62 段 | 整体 AUC 0.892，视觉通道 0.912，11 倍实时 |
| 视频审核 | 自建中文切片 8 段 | AUC 1.000 |
| 文本审核 | COLD + 广告切片 1712 条 | 融合 AUC 0.914（纯模型 0.856，纯规则 0.597）|
| 深伪专家 | 500 张同分布 / 25 张分布外 | AUC 0.999 vs 分布外误报率 **72%** |
| 图像对抗 | 20 张伪造样本 | PGD 在 ε=1/255（PSNR 49dB）下成功率 95% |
| 图像防御 | BPDA 自适应攻击 | ε≥2/255 时全部防御恢复率归零 |
| 语音对抗 | 13 种攻击 | 无差别攻击 11/13 失效；定点消除关键词逃逸率 25% |
| 视频对抗 | 5 种攻击 | 降分辨率至 0.4 倍逃逸率 67%（零模型访问）|
| 聊天机器人护栏 | 直接/混淆/间接注入 | 攻击成功率全部降到 0，良性误拒率 0 |

详见 [实验报告](docs/实验报告.md)。

---

## 八、注意事项

- 阈值校准脚本 `scripts_calibrate.py` 会在"确定正常"的素材上估计各专家的分数分布，
  按目标误报率设定阈值。**更换素材域后需要重新校准**，否则零样本专家会大量误报。
- 多个实验同时使用 MPS 会互相阻塞，可用 `SOCIALSIM_DEVICE=cpu` 让其中一个走 CPU。
- 模型缓存默认在 `~/.cache/huggingface`，首次下载约 1.5 GB。
- 视频数据集约 500 MB，可用 `scripts_fetch_xdviolence.py --violent N --normal N` 调整规模。
