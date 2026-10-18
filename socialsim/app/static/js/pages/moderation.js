// 内容审核：文本（命中高亮 + 混淆还原过程）、图片 / 视频（多轨时间线 + 证据帧默认模糊）、审核记录队列
import { store, api, postJSON, pollTask, labelName, splitLabels, fmt, pct, VERDICT, verdictOf, bumpStats } from '../lib/api.js';
import { C, gaugeOption } from '../lib/charts.js';

const { ref, computed, onMounted, onBeforeUnmount } = Vue;

const TEXT_SAMPLES = [
  { label: '全角混淆', text: 'ｖｘ：ｋｋ８８９９８ 稳赚不赔' },
  { label: '分隔 + 叠字', text: '傻傻傻 逼' },
  { label: '谐音替换', text: '你真是个煞笔，脑残玩意' },
  { label: '同音字引流', text: '加 薇 信 xk8899，兼职日结300' },
  { label: '赌博广告', text: '棋牌平台首充返水50%，百家乐六合' },
  { label: '正常内容', text: '今天天气真好，出去走走 #周末' },
];
const STAGES = [
  ['raw', '原文'], ['nfkc', 'NFKC 归一化 · 去零宽字符 · 繁转简'], ['compact', '去掉分隔符'], ['dedup', '压缩重复字符'],
  ['homo', '谐音 / 形近字替换'], ['pinyin', '汉字转拼音（捕捉纯拼音绕过）'],
];
const EXPERTS = [
  ['nsfw', 'NSFW 分类器'], ['clip_porn', '色情'], ['clip_violence', '暴力'], ['clip_gore', '血腥'], ['clip_weapon', '武器'],
  ['clip_smoking_drug', '烟草/毒品'], ['clip_protest_crowd', '人群聚集'], ['clip_fire_disaster', '火灾/灾害'], ['clip_horror', '惊悚'],
];
const TIMING = { fast_scan: '快速扫描 · 选关键帧', frame_experts: '帧级视觉专家', deepfake: '深伪检测', ocr: '画面文字 OCR', audio: '语音理解 ASR' };
const DOMAINS = { cartoon: '动画', game: '游戏', text_screen: '文字画面', photo: '实拍' };

/** 在原文中标出未经混淆就命中的词和联系方式结构（只拆片段，不用 v-html） */
function highlight(raw, rule) {
  const ranges = [];
  const add = (w, kind) => {
    if (!w) return;
    const lo = raw.toLowerCase(), wl = w.toLowerCase();
    for (let i = lo.indexOf(wl); i >= 0; i = lo.indexOf(wl, i + 1)) ranges.push([i, i + w.length, kind]);
  };
  (rule?.hits || []).filter(h => !h.obfuscated).forEach(h => add(h.word, 'hit'));
  (rule?.structures || []).forEach(s => add(s.match, 'struct'));
  ranges.sort((a, b) => a[0] - b[0]);
  const out = []; let pos = 0;
  for (const [a, b, kind] of ranges) {
    if (a < pos) continue;
    if (a > pos) out.push({ t: raw.slice(pos, a) });
    out.push({ t: raw.slice(a, b), kind }); pos = b;
  }
  if (pos < raw.length) out.push({ t: raw.slice(pos) });
  return out;
}

/** 多轨时间线：融合分数（按阈值分段着色）/ 9 个视觉专家热力条 / 语音与画面文字命中，三轨共用时间轴缩放 */
function timelineOption(r) {
  const dur = r.meta?.duration || Math.max(1, ...r.frames.map(f => f.t));
  const th = r.thresholds || store.meta.thresholds;
  const frames = r.frames || [];
  const cell = Math.max(4, Math.min(26, 760 / Math.max(1, frames.length)));
  const heat = [];
  frames.forEach(f => EXPERTS.forEach(([k], j) => heat.push([f.t, j, +(f.parts?.[k] ?? 0).toFixed(3)])));
  const hits = [
    ...(r.asr_hits || []).map(h => ({ value: [h.start, 1, h.risk], text: h.text, kind: '语音', end: h.end, labels: h.labels })),
    ...(r.ocr_hits || []).map(h => ({ value: [h.t, 0, h.score], text: h.text, kind: '画面文字', labels: h.labels })),
  ];
  const tAxis = (i, show) => ({ type: 'value', gridIndex: i, min: 0, max: +dur.toFixed(1), axisLabel: { show, formatter: v => `${v}s` },
    axisTick: { show: false }, splitLine: { show: false }, axisLine: { show: show, lineStyle: { color: C.line } } });
  return {
    tooltip: { trigger: 'item', confine: true },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    grid: [{ left: 100, right: 64, top: 18, height: 120 }, { left: 100, right: 64, top: 160, height: 9 * 15 }, { left: 100, right: 64, top: 322, height: 44 }],
    xAxis: [tAxis(0, false), tAxis(1, false), tAxis(2, true)],
    yAxis: [
      { gridIndex: 0, type: 'value', min: 0, max: 1, name: '融合风险', nameLocation: 'middle', nameGap: 34, splitNumber: 2 },
      { gridIndex: 1, type: 'category', data: EXPERTS.map(e => e[1]), axisLabel: { fontSize: 10.5 }, inverse: true, axisLine: { show: false } },
      { gridIndex: 2, type: 'category', data: ['画面文字', '语音'], axisLabel: { fontSize: 11 }, axisLine: { show: false } },
    ],
    visualMap: [
      { show: false, seriesIndex: 0, pieces: [{ lt: th.review, color: C.pass }, { gte: th.review, lt: th.block, color: C.review }, { gte: th.block, color: C.block }] },
      { show: false, seriesIndex: 1, min: 0, max: 1, inRange: { color: ['#f2f6ff', '#b9ceff', '#722ed1', '#f53f3f'] } },
    ],
    dataZoom: [{ type: 'inside', xAxisIndex: [0, 1, 2], zoomOnMouseWheel: 'ctrl', moveOnMouseWheel: false },
      { type: 'slider', xAxisIndex: [0, 1, 2], bottom: 4, height: 16, showDetail: false }],
    series: [
      { type: 'line', xAxisIndex: 0, yAxisIndex: 0, showSymbol: frames.length < 40, symbolSize: 5, data: frames.map(f => [f.t, +f.score.toFixed(3)]),
        tooltip: { formatter: p => `${p.value[0]}s · 帧级风险 <b>${p.value[1]}</b>` },
        areaStyle: { opacity: 0.08 },
        markLine: { symbol: 'none', silent: true, data: [
          { yAxis: th.review, lineStyle: { color: C.review, type: 'dashed' }, label: { formatter: `复核 ${(+th.review).toFixed(2)}`, color: C.review, fontSize: 10 } },
          { yAxis: th.block, lineStyle: { color: C.block, type: 'dashed' }, label: { formatter: `拦截 ${(+th.block).toFixed(2)}`, color: C.block, fontSize: 10 } }] },
        markArea: { silent: true, itemStyle: { color: 'rgba(245,63,63,.08)' }, data: (r.segments || []).map(s => [{ xAxis: s.start }, { xAxis: s.end }]) } },
      { type: 'scatter', xAxisIndex: 1, yAxisIndex: 1, symbol: 'rect', symbolSize: [cell, 13], data: heat,
        tooltip: { formatter: p => `${p.value[0]}s · ${EXPERTS[p.value[1]][1]}：<b>${p.value[2]}</b>` } },
      { type: 'scatter', xAxisIndex: 2, yAxisIndex: 2, symbol: 'roundRect', symbolSize: [14, 14], data: hits,
        itemStyle: { color: p => p.data.value[2] >= th.block ? C.block : p.data.value[2] >= th.review ? C.review : C.primary },
        tooltip: { formatter: p => `${p.data.kind} · ${p.value[0]}s${p.data.end ? '–' + p.data.end + 's' : ''}<br/>${p.data.text}<br/>风险 <b>${(+p.value[2]).toFixed(2)}</b> · ${(p.data.labels || []).map(labelName).join('、')}` } },
    ],
  };
}

export default {
  name: 'ModerationPage',
  setup() {
    const tab = ref('text');
    // ---------- 文本 ----------
    const text = ref(''), textRes = ref(null), textLoading = ref(false), textInput = ref('');
    async function checkText() {
      if (!text.value.trim()) return;
      textLoading.value = true;
      try { textRes.value = await postJSON('/api/moderate/text', { text: text.value }); textInput.value = text.value; }
      catch (e) { ElementPlus.ElMessage.error(e.message); }
      finally { textLoading.value = false; }
    }
    const textVerdict = computed(() => textRes.value ? verdictOf(textRes.value.score) : null);
    const rule = computed(() => textRes.value?.experts?.rule || {});
    const cold = computed(() => textRes.value?.experts?.cold || {});
    const marked = computed(() => textRes.value ? highlight(textInput.value, rule.value) : []);
    const stages = computed(() => {
      const st = rule.value.stages;
      if (!st) return [];
      // 每个混淆词只标在它第一次能被匹配到的那一步
      const first = {};
      for (const h of (rule.value.hits || []).filter(h => h.obfuscated)) {
        const w = h.word.toLowerCase().replace(/\s/g, '');
        const k = h.via === 'pinyin' ? 'pinyin' : ['compact', 'dedup', 'homo'].find(k => (st[k] || '').toLowerCase().includes(w));
        if (k) (first[k] = first[k] || []).push(h.word);
      }
      let prev = null;
      return STAGES.map(([k, name]) => {
        const v = st[k] ?? '';
        const changed = prev !== null && k !== 'pinyin' && v !== prev;
        if (k !== 'pinyin') prev = v;
        return { k, name, v, changed, hits: first[k] || [] };
      });
    });
    const textGauge = computed(() => textRes.value && gaugeOption(textRes.value.score));

    // ---------- 图片 / 视频 ----------
    const samples = ref({ own_videos: [], xd_videos: [], images: [] }), sample = ref('');
    const file = ref(null), preview = ref(''), running = ref(false), elapsed = ref(0), media = ref(null), report = ref(null);
    let timer = null;
    onMounted(async () => { samples.value = await api('/api/samples/media'); });
    onBeforeUnmount(() => clearInterval(timer));
    const onFile = f => {
      file.value = f.raw; sample.value = '';
      if (preview.value.startsWith('blob:')) URL.revokeObjectURL(preview.value);
      preview.value = f.raw.type.startsWith('image/') ? URL.createObjectURL(f.raw) : '';
    };
    const onSample = p => {
      file.value = null;
      preview.value = /\.(jpe?g|png|webp|bmp)$/i.test(p) ? '/media/' + p : '';
    };
    async function runMedia() {
      if (!file.value && !sample.value) return;
      running.value = true; media.value = null; report.value = null; elapsed.value = 0;
      const t0 = Date.now();
      timer = setInterval(() => { elapsed.value = Math.round((Date.now() - t0) / 1000); }, 500);
      try {
        let r;
        if (sample.value) r = await postJSON('/api/moderate/sample', { path: sample.value });
        else {
          const fd = new FormData(); fd.append('file', file.value);
          r = await api('/api/moderate/upload', { method: 'POST', body: fd });
        }
        const t = await pollTask(r.task_id, { interval: 1500 });
        media.value = { ...t.result, is_video: r.is_video, file: r.file };
        if (t.result.report_url) report.value = await api(t.result.report_url.replace('report.html', 'report.json'));
        loadRecords(); bumpStats();
      } catch (e) {
        ElementPlus.ElMessage.error('审核失败：' + e.message);
      } finally { clearInterval(timer); running.value = false; }
    }
    const mediaVerdict = computed(() => media.value && (media.value.verdict || verdictOf(media.value.score || 0)));
    const mediaGauge = computed(() => media.value && gaugeOption(media.value.score || 0, report.value?.thresholds ? { th: report.value.thresholds } : {}));
    const imageParts = computed(() => media.value?.experts?.frame?.parts || null);
    const imageOpt = computed(() => {
      const p = imageParts.value;
      if (!p) return null;
      const rows = [...EXPERTS, ['deepfake', '人脸深伪']].filter(([k]) => p[k] !== undefined);
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => (+v).toFixed(3) },
        grid: { left: 90, right: 30, top: 6, bottom: 20 },
        xAxis: { type: 'value', min: 0, max: 1 }, yAxis: { type: 'category', data: rows.map(r => r[1]), inverse: true },
        series: [{ type: 'bar', barMaxWidth: 12, itemStyle: { borderRadius: [0, 3, 3, 0] },
          data: rows.map(([k]) => ({ value: p[k], itemStyle: { color: p[k] >= 0.5 ? C.block : p[k] >= 0.25 ? C.review : C.primary } })) }],
      };
    });
    const tlOpt = computed(() => report.value && report.value.frames?.length ? timelineOption(report.value) : null);
    const thumbs = computed(() => {
      const r = report.value;
      if (!r) return [];
      const base = media.value.report_url.replace('report.html', '');
      // 旧报告的 report.json 里没有 thumbs 字段，按 write_report 的命名规则还原
      const list = r.thumbs || [...r.frames].sort((a, b) => b.score - a.score).slice(0, 8)
        .map(f => ({ t: f.t, score: f.score, labels: f.labels, file: `thumbs/${`t${f.t.toFixed(2)}.jpg`.replace('.', '_')}` }));
      return list.map(x => ({ ...x, url: base + x.file }));
    });
    const timingOpt = computed(() => {
      const t = report.value?.timing;
      if (!t) return null;
      const keys = Object.keys(t).filter(k => TIMING[k]);
      return {
        tooltip: { trigger: 'item', formatter: p => `${p.seriesName}：${p.value}s` },
        legend: { top: 0, itemGap: 10, textStyle: { fontSize: 11 } }, grid: { left: 10, right: 20, top: 30, bottom: 4, containLabel: true },
        xAxis: { type: 'value', axisLabel: { formatter: v => v + 's' } }, yAxis: { type: 'category', data: ['耗时'], axisLabel: { show: false } },
        series: keys.map((k, i) => ({ name: TIMING[k], type: 'bar', stack: 't', barWidth: 18, data: [t[k]],
          itemStyle: { color: [C.cyan, C.primary, C.purple, C.gold, C.review][i % 5], borderRadius: 0 } })),
      };
    });

    // ---------- 审核记录 ----------
    const records = ref([]), queue = ref('all'), scanning = ref(false), scanRes = ref(null);
    async function loadRecords() { records.value = (await api('/api/moderation_records?limit=200')).records; }
    onMounted(loadRecords);
    const queueCounts = computed(() => ({ all: records.value.length, review: records.value.filter(r => r.verdict === 'review').length,
      block: records.value.filter(r => r.verdict === 'block').length, pass: records.value.filter(r => r.verdict === 'pass').length }));
    const queued = computed(() => queue.value === 'all' ? records.value : records.value.filter(r => r.verdict === queue.value));
    const reportLink = r => r.report_path && r.report_path.includes('/results/') ? '/reports/' + r.report_path.split('/results/')[1] : '';
    async function scanAll() {
      scanning.value = true;
      try {
        const r = await postJSON('/api/moderate/scan_all');
        scanRes.value = (await pollTask(r.task_id)).result;
        bumpStats();
      } catch (e) { ElementPlus.ElMessage.error(e.message); }
      finally { scanning.value = false; }
    }
    const scanOpt = computed(() => {
      const s = scanRes.value;
      if (!s) return null;
      const pass = s.posts + s.comments - s.blocked - s.review;
      const top = Object.entries(s.labels || {}).sort((a, b) => b[1] - a[1]).slice(0, 8).reverse();
      return {
        tooltip: { trigger: 'item' },
        grid: { left: '52%', right: 20, top: 10, bottom: 20, containLabel: true },
        xAxis: { type: 'value' }, yAxis: { type: 'category', data: top.map(([l]) => labelName(l)) },
        series: [
          { type: 'pie', radius: ['45%', '70%'], center: ['22%', '50%'], label: { formatter: '{b}\n{c}', fontSize: 11 },
            data: [{ name: '通过', value: pass, itemStyle: { color: C.pass } }, { name: '待复核', value: s.review, itemStyle: { color: C.review } },
              { name: '拦截', value: s.blocked, itemStyle: { color: C.block } }] },
          { type: 'bar', barMaxWidth: 12, data: top.map(([, v]) => v), itemStyle: { color: C.primary, borderRadius: [0, 3, 3, 0] } },
        ],
      };
    });

    return {
      store, tab, TEXT_SAMPLES, text, textRes, textLoading, checkText, textVerdict, rule, cold, marked, stages, textGauge,
      samples, sample, file, preview, onFile, onSample, running, elapsed, runMedia, media, report, mediaVerdict, mediaGauge,
      imageParts, imageOpt, tlOpt, thumbs, timingOpt, DOMAINS,
      records, queue, queueCounts, queued, reportLink, scanning, scanAll, scanRes, scanOpt,
      fmt, pct, labelName, VERDICT,
    };
  },
  template: `
  <div>
    <div class="page-head">
      <div>
        <h1>内容审核</h1>
        <p>多专家融合：文本走规则词表 + COLD 中文冒犯语言模型；图片走 NSFW + CLIP 零样本风险类目 + 人脸深伪；视频先按镜头切分选关键帧，再并行跑帧级专家、语音理解和画面文字，最后做时序聚合。</p>
      </div>
    </div>
    <el-card class="sn-card" shadow="never">
      <el-tabs v-model="tab">
        <el-tab-pane label="文本审核" name="text">
          <div class="grid-2">
            <div>
              <el-input v-model="text" type="textarea" :rows="5" placeholder="输入要检测的文本" maxlength="500" show-word-limit/>
              <div class="row" style="margin:10px 0">
                <span class="muted">样例：</span>
                <span v-for="s in TEXT_SAMPLES" :key="s.label" class="try-chip" @click="text = s.text; checkText()">{{ s.label }}</span>
              </div>
              <el-button type="primary" :loading="textLoading" :disabled="!text.trim()" @click="checkText"><el-icon><Search/></el-icon>&nbsp;检测</el-button>
              <template v-if="textRes">
                <div class="chart-title" style="margin-top:18px">命中位置 <span class="muted"><span class="mark-hit">橙色</span> = 词表命中，<span class="mark-struct">红色</span> = 联系方式 / 链接结构</span></div>
                <div class="text-preview"><template v-for="(s, i) in marked" :key="i"><span v-if="s.kind" :class="s.kind === 'hit' ? 'mark-hit' : 'mark-struct'">{{ s.t }}</span><template v-else>{{ s.t }}</template></template></div>
                <div v-if="rule.hits && rule.hits.some(h => h.obfuscated)" class="muted" style="margin-top:6px">
                  经过混淆、在原文里找不到的命中词：<b style="color:#cc6400">{{ rule.hits.filter(h => h.obfuscated).map(h => h.word).join('、') }}</b>（见右侧还原过程）</div>
              </template>
            </div>
            <div v-if="textRes">
              <div class="verdict-banner" :class="textVerdict">
                <div class="vb-icon"><el-icon><component :is="VERDICT[textVerdict].icon"/></el-icon></div>
                <div><h3>{{ VERDICT[textVerdict].text }} · 风险 {{ fmt(textRes.score) }}</h3>
                  <p>复核阈值 {{ store.meta.thresholds.review }}，拦截阈值 {{ store.meta.thresholds.block }}</p></div>
              </div>
              <div class="grid-2" style="gap:12px">
                <SnChart :option="textGauge" :height="170"/>
                <div class="expert-bars">
                  <div class="eb"><span>规则专家</span><ScoreBar :score="rule.score" :color="rule.score >= 0.7 ? '#f53f3f' : rule.score >= 0.4 ? '#ff7d00' : '#00b42a'"/></div>
                  <div class="eb"><span>COLD 模型</span><ScoreBar :score="cold.offensive_prob" :color="cold.offensive_prob >= 0.5 ? '#f53f3f' : '#00b42a'"/></div>
                  <div class="eb"><span>刻意混淆</span><span><el-tag :type="rule.obfuscated ? 'warning' : 'info'" size="small">{{ rule.obfuscated ? '是（+0.15 加权）' : '否' }}</el-tag></span></div>
                  <div class="eb"><span>风险标签</span><LabelTags :labels="textRes.labels"/></div>
                </div>
              </div>
              <div class="chart-title" style="margin-top:8px">混淆还原过程 <span class="muted">任一变体命中词表即视为命中</span></div>
              <el-timeline class="norm-steps" style="padding-left:2px">
                <el-timeline-item v-for="s in stages" :key="s.k" :timestamp="s.name" placement="top" size="large"
                  :type="s.hits.length ? 'warning' : s.changed ? 'primary' : ''" :hollow="!s.changed && !s.hits.length">
                  <span :class="{ changed: s.changed }">{{ s.v || '（空）' }}</span>
                  <el-tag v-for="w in s.hits" :key="w" type="warning" size="small" style="margin-left:6px">在这一步命中「{{ w }}」</el-tag>
                </el-timeline-item>
              </el-timeline>
            </div>
            <div v-else class="chat-empty" style="height:auto;padding:60px 0"><el-icon><Document/></el-icon>选一个样例或输入文本，查看命中位置和混淆还原过程</div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="图片 / 视频审核" name="media">
          <div class="grid-1-2">
            <div>
              <el-upload class="upload-zone" drag :auto-upload="false" :show-file-list="false" accept="image/*,video/*" :on-change="onFile">
                <el-icon class="el-icon--upload"><UploadFilled/></el-icon>
                <div>把图片或视频拖到这里，或<em>点击选择</em></div>
                <div class="muted" style="margin-top:4px">文件只在本机处理</div>
              </el-upload>
              <div v-if="file" class="muted" style="margin-top:8px">已选择：{{ file.name }}（{{ (file.size / 1048576).toFixed(1) }} MB）</div>
              <div class="chart-title" style="margin-top:14px">或者选一个本地示例素材</div>
              <el-select v-model="sample" filterable clearable placeholder="自建中文切片 / XD-Violence / 风险类目图片" style="width:100%" @change="onSample">
                <el-option-group label="自建中文违规切片（合成，含语音和画面文字）">
                  <el-option v-for="v in samples.own_videos" :key="v.path" :label="v.name" :value="v.path">
                    <span>{{ v.name }}</span><span class="muted" style="float:right">{{ v.truth.risky ? '违规' : '正常' }}</span></el-option>
                </el-option-group>
                <el-option-group label="XD-Violence 公开基准（影视片段）">
                  <el-option v-for="v in samples.xd_videos" :key="v.path" :label="v.name" :value="v.path"/>
                </el-option-group>
                <el-option-group label="风险类目图片（Wikimedia Commons）">
                  <el-option v-for="v in samples.images" :key="v.path" :label="v.name" :value="v.path"/>
                </el-option-group>
              </el-select>
              <el-button type="primary" style="width:100%;margin-top:12px" :loading="running" :disabled="!file && !sample" @click="runMedia">
                <el-icon v-if="!running"><VideoPlay/></el-icon>&nbsp;{{ running ? '分析中 ' + elapsed + 's' : '开始审核' }}</el-button>
              <div v-if="preview" class="blur-media" style="margin-top:12px"><img :src="preview" alt="预览"></div>
            </div>
            <div>
              <div v-if="running" class="running-box">
                <el-icon class="is-loading" :size="30" color="#165dff"><Loading/></el-icon>
                <div>正在分析，已用时 <b>{{ elapsed }}s</b>（视频首次运行要加载语音模型，可能需要一两分钟）</div>
                <div class="pipeline-chips"><span>镜头切分 · 关键帧</span><span>NSFW / CLIP 视觉专家</span><span>人脸深伪</span><span>画面文字 OCR</span><span>语音理解 ASR</span><span>时序聚合</span></div>
              </div>
              <div v-else-if="!media" class="chat-empty" style="height:auto;padding:80px 0"><el-icon><Film/></el-icon>上传或选择素材后开始审核<br><span class="muted">视频会给出多轨风险时间线和证据帧</span></div>
              <template v-else>
                <div class="verdict-banner" :class="mediaVerdict">
                  <div class="vb-icon"><el-icon><component :is="VERDICT[mediaVerdict].icon"/></el-icon></div>
                  <div style="flex:1"><h3>{{ VERDICT[mediaVerdict].text }} · 风险 {{ fmt(media.score) }}</h3>
                    <p>{{ media.file }}<template v-if="report"> · {{ report.meta.duration.toFixed(1) }}s {{ report.meta.kind === 'short' ? '短视频' : '长视频' }} · 分析 {{ report.n_frames_analyzed }} 帧 · 用时 {{ report.elapsed }}s</template></p></div>
                  <a v-if="media.report_url" :href="media.report_url" target="_blank"><el-button size="small">完整报告 <el-icon><TopRight/></el-icon></el-button></a>
                </div>
                <div class="grid-2" style="gap:12px">
                  <SnChart :option="mediaGauge" :height="170"/>
                  <div class="expert-bars">
                    <template v-if="media.is_video">
                      <div class="eb"><span>视觉通道</span><ScoreBar :score="media.visual_score" color="#165dff"/></div>
                      <div class="eb"><span>语音通道</span><ScoreBar :score="media.asr_score" color="#722ed1"/></div>
                      <div class="eb"><span>画面文字</span><ScoreBar :score="media.ocr_score" color="#f7ba1e"/></div>
                    </template>
                    <template v-else>
                      <div class="eb"><span>画面域</span><span>{{ DOMAINS[imageParts?.domain] || imageParts?.domain || '-' }}</span></div>
                      <div class="eb"><span>人脸数</span><span>{{ media.experts?.faces ?? 0 }}</span></div>
                      <div class="eb"><span>画面文字</span><span class="muted">{{ media.experts?.ocr?.text || '无' }}</span></div>
                    </template>
                    <div class="eb"><span>风险标签</span><LabelTags :labels="media.labels" :max="5"/></div>
                  </div>
                </div>
                <template v-if="tlOpt">
                  <div class="chart-title" style="margin-top:6px">多轨风险时间线 <span class="muted">红底 = 高风险片段；Ctrl + 滚轮或拖动下方滑块缩放</span></div>
                  <SnChart :option="tlOpt" :height="400"/>
                </template>
                <template v-if="imageOpt">
                  <div class="chart-title">各视觉专家分数</div>
                  <SnChart :option="imageOpt" :height="250"/>
                </template>
                <template v-if="thumbs.length">
                  <div class="chart-title" style="margin-top:8px">证据帧 <span class="muted">风险最高的帧，默认模糊，鼠标悬停才显示</span></div>
                  <div class="evidence-frames">
                    <figure v-for="t in thumbs" :key="t.url"><div class="blur-media"><img :src="t.url" @error="e => { e.target.closest('figure').style.display = 'none'; }"></div>
                      <figcaption>{{ t.t }}s · {{ t.score.toFixed(2) }}<br><span class="muted">{{ (t.labels || []).map(labelName).join('、') || '无标签' }}</span></figcaption></figure>
                  </div>
                </template>
                <template v-if="report && (report.asr_hits.length || report.ocr_hits.length || report.audio?.full_text)">
                  <div class="chart-title" style="margin-top:14px">语音与画面文字</div>
                  <el-table :data="[...report.asr_hits.map(h => ({ kind: '语音', time: h.start + '–' + h.end + 's', text: h.text, risk: h.risk, labels: h.labels })), ...report.ocr_hits.map(h => ({ kind: '画面文字', time: h.t + 's', text: h.text, risk: h.score, labels: h.labels }))]" size="small">
                    <el-table-column prop="kind" label="通道" width="86"/><el-table-column prop="time" label="时间" width="100"/>
                    <el-table-column prop="text" label="内容" min-width="200"/>
                    <el-table-column label="风险" width="120"><template #default="{ row }"><RiskLevel :score="row.risk" :digits="2"/></template></el-table-column>
                    <el-table-column label="标签" min-width="120"><template #default="{ row }"><LabelTags :labels="row.labels" :max="3"/></template></el-table-column>
                  </el-table>
                  <div v-if="report.audio?.full_text" class="muted" style="margin-top:8px;line-height:1.7">完整转写：{{ report.audio.full_text.slice(0, 400) }}</div>
                </template>
                <template v-if="timingOpt">
                  <div class="chart-title" style="margin-top:14px">各阶段耗时</div>
                  <SnChart :option="timingOpt" :height="78"/>
                </template>
              </template>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane :label="'审核记录（' + records.length + '）'" name="records">
          <div class="row" style="margin-bottom:12px">
            <el-segmented v-model="queue" :options="[{ label: '全部 ' + queueCounts.all, value: 'all' }, { label: '待复核 ' + queueCounts.review, value: 'review' },
              { label: '已拦截 ' + queueCounts.block, value: 'block' }, { label: '已通过 ' + queueCounts.pass, value: 'pass' }]"/>
            <span class="spacer"></span>
            <el-button :loading="scanning" @click="scanAll"><el-icon><Finished/></el-icon>&nbsp;批量审核存量文本</el-button>
          </div>
          <div v-if="scanRes" class="drawer-section" style="background:var(--sn-fill)">
            <div class="chart-title">批量审核结果 <span class="muted">帖子 {{ scanRes.posts }} 条、评论 {{ scanRes.comments }} 条；拦截 {{ scanRes.blocked }}，转复核 {{ scanRes.review }}</span></div>
            <SnChart :option="scanOpt" :height="200"/>
          </div>
          <el-table :data="queued" class="sn-table" max-height="560">
            <el-table-column prop="created_at" label="时间" width="110"/>
            <el-table-column label="来源" width="90"><template #default="{ row }">{{ { upload: '上传', post: '帖子', comment: '评论' }[row.type] || row.type }}</template></el-table-column>
            <el-table-column label="媒体" width="80"><template #default="{ row }">{{ { text: '文本', image: '图片', video: '视频' }[row.media] || row.media }}</template></el-table-column>
            <el-table-column label="风险" width="120" sortable :sort-method="(a, b) => a.score - b.score"><template #default="{ row }"><RiskLevel :score="row.score"/></template></el-table-column>
            <el-table-column label="判定" width="100"><template #default="{ row }"><VerdictTag :verdict="row.verdict"/></template></el-table-column>
            <el-table-column label="标签" min-width="220"><template #default="{ row }"><LabelTags :labels="row.labels" :max="4"/></template></el-table-column>
            <el-table-column label="耗时" width="80"><template #default="{ row }">{{ row.elapsed }}s</template></el-table-column>
            <el-table-column label="报告" width="80"><template #default="{ row }"><a v-if="reportLink(row)" :href="reportLink(row)" target="_blank">查看</a><span v-else class="muted">-</span></template></el-table-column>
          </el-table>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>`,
};
