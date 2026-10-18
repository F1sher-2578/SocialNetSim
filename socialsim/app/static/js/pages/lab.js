// 攻防实验：把 results/*.json 做成按模态分标签页的实验仪表板，每张图配一句结论，负面结果单独标出
import { api, fmt, pct } from '../lib/api.js';
import { C, axisPct } from '../lib/charts.js';

const { ref, computed, onMounted } = Vue;

const IMG_DEF = {
  none: '无防御', jpeg75: 'JPEG q75', jpeg50: 'JPEG q50', bit4: '位深压缩 4bit', bit3: '位深压缩 3bit', median3: '中值滤波 3×3',
  'blur1.0': '高斯模糊', randresize: '随机缩放填充', randomized_ensemble: '随机化集成',
};
const IMG_NAT = {
  jpeg_q40: 'JPEG q40', jpeg_q20: 'JPEG q20', jpeg_q10: 'JPEG q10', 'downscale_0.5': '缩小到 0.5', 'downscale_0.25': '缩小到 0.25',
  gauss_noise_8: '高斯噪声 σ8', gauss_noise_20: '高斯噪声 σ20', hflip: '水平翻转', rotate_5: '旋转 5°', rotate_15: '旋转 15°',
  white_border: '加白边', blur: '模糊',
};
const AUDIO_ATK = {
  'volume_0.05': '音量压到 5%', keyword_mask: '定点消除关键词', 'speed_1.15x': '加速 1.15×', 'speed_0.85x': '减速 0.85×',
  'pitch_+3': '升调 3 半音', 'pitch_-3': '降调 3 半音', noise_snr10: '加噪 SNR 10dB', noise_snr5: '加噪 SNR 5dB', noise_snr0: '加噪 SNR 0dB',
  bandpass_tel: '电话带通滤波', echo: '回声', time_mask: '时间遮挡', 'psycho_-25dB': '心理声学掩蔽 −25dB',
  'psycho_-15dB': '心理声学掩蔽 −15dB', selective_overlay: '选择性叠加干扰',
};
const AUDIO_DEF = { loudness_norm: '响度归一化', spectral_sub: '谱减降噪', resample_8k: '重采样 8kHz', 'norm+specsub': '归一化 + 谱减' };
const SLICES = { S1_COLD_offensive: 'COLD 冒犯言论', S2_spam_ads: '广告引流（自建）', S3_combined: '合并' };
const EXP = [['rule_only', '规则专家', C.grey], ['model_only', 'COLD 模型', C.primary], ['fused', '融合', C.purple]];
const jitter = i => { const x = Math.sin(i * 12.9898) * 43758.5453; return (x - Math.floor(x) - 0.5) * 0.56; };
const epsLabel = k => k.replace('eps_', 'ε=');

export default {
  name: 'LabPage',
  setup() {
    const d = ref(null), bot = ref(null), tab = ref('overview');
    onMounted(async () => { [d.value, bot.value] = await Promise.all([api('/api/security/results'), api('/api/bot_metrics')]); });

    // ---------- 总览 ----------
    const cards = computed(() => {
      if (!d.value || !bot.value) return [];
      const x = d.value.video_bench_xdv?.xdviolence || {}, tb = d.value.text_bench_combined?.S3_combined || {};
      const df = d.value.deepfake_eval || {}, ai = d.value.adv_image || {}, au = d.value.adv_audio?.summary?.attacks || {};
      const av = d.value.adv_video?.summary || {}, cg = d.value.chatbot_guard || {};
      const unseen = Math.min(...Object.values(bot.value.evasion_generalization || { x: { 'detect_rate@0.5': NaN } }).map(v => v['detect_rate@0.5']));
      const bpda2 = Object.values(ai.adaptive_bpda_by_eps?.['eps_2/255'] || {}).map(v => v.adaptive_recovered);
      const audioZero = Object.values(au).filter(v => v.escape_rate === 0).length;
      const bestVideo = Object.entries(av).sort((a, b) => b[1].escape_rate - a[1].escape_rate)[0];
      return [
        { tab: 'bot', icon: 'Search', title: '水军检测', value: fmt(bot.value.models?.RandomForest?.auc), unit: 'AUC', desc: `194 账号 / 11 团伙，团伙级召回 ${bot.value.campaign_level?._summary?.n_detected}/${bot.value.campaign_level?._summary?.n_campaigns}` },
        { tab: 'bot', icon: 'WarnTriangleFilled', title: '水军检测 · 跨规避等级', value: pct(unseen), unit: '检出率', desc: '训练时没见过养号型水军', negative: true },
        { tab: 'video', icon: 'VideoCamera', title: '视频审核 · XD-Violence', value: fmt(x.auc), unit: 'AUC', desc: `${x.n} 段公开基准，视觉通道 ${fmt(x.auc_visual)}，${x.realtime_factor}× 实时` },
        { tab: 'text', icon: 'Document', title: '文本审核 · COLD + 广告', value: fmt(tb.fused?.auc), unit: '融合 AUC', desc: `纯模型 ${fmt(tb.model_only?.auc)}，纯规则 ${fmt(tb.rule_only?.auc)}` },
        { tab: 'deepfake', icon: 'Avatar', title: '深伪专家 · 分布外', value: pct(df.ood_real_faces?.['fpr@0.5']), unit: '误报率', desc: `同分布 AUC ${fmt(df.auc)}，换成真实肖像后大面积误报`, negative: true },
        { tab: 'image', icon: 'Picture', title: '图像对抗 · PGD', value: pct(ai.whitebox?.['eps_1/255']?.pgd_asr), unit: '成功率', desc: `ε=1/255，PSNR ${fmt(ai.whitebox?.['eps_1/255']?.pgd_psnr_db, 1)} dB，肉眼不可见` },
        { tab: 'image', icon: 'CircleClose', title: '图像防御 · 自适应攻击', value: pct(bpda2.length ? Math.max(...bpda2) : null), unit: '最高恢复率', desc: 'ε=2/255 时 BPDA 让全部防御失效', negative: true },
        { tab: 'audio', icon: 'Microphone', title: '语音对抗', value: `${audioZero}/${Object.keys(au).length}`, unit: '种攻击无效', desc: `定点消除关键词逃逸率 ${pct(au.keyword_mask?.escape_rate)}` },
        { tab: 'video_adv', icon: 'Film', title: '视频对抗 · ' + (bestVideo ? bestVideo[0].replace(/^A\d+[a-z]?_/, '') : ''), value: pct(bestVideo?.[1].escape_rate), unit: '逃逸率', desc: '不需要任何模型访问，降分辨率即可', negative: true },
        { tab: null, icon: 'Lock', title: '聊天机器人护栏', value: pct(cg.guard_on?.direct?.asr), unit: '攻击成功率', desc: `直接 / 混淆 / 间接注入全部降为 0，良性误拒 ${pct(cg.guard_on?.benign?.false_refusal_rate)}`, link: '#/assistant' },
      ];
    });
    const goCard = c => { if (c.link) location.hash = c.link; else if (c.tab === 'bot') location.hash = '#/detect'; else tab.value = c.tab; };

    // ---------- 视频基准 ----------
    const x = computed(() => d.value?.video_bench_xdv?.xdviolence);
    const scatterOpt = computed(() => {
      const v = x.value;
      if (!v?.rows) return null;
      const th = v.adaptive_thresholds || {};
      const pts = truth => v.rows.map((r, i) => ({ r, i })).filter(o => o.r.truth === truth)
        .map(({ r, i }) => ({ value: [truth + jitter(i), r.score], file: r.file, cats: r.categories, verdict: r.verdict }));
      return {
        tooltip: { trigger: 'item', formatter: p => `${p.data.file.split('__')[0]}<br/>真值类别：${(p.data.cats || []).join('、') || '无'}<br/>风险分 <b>${p.value[1].toFixed(3)}</b>` },
        legend: { top: 0, right: 90 }, grid: { left: 48, right: 90, top: 40, bottom: 30 },
        xAxis: { type: 'value', min: -0.5, max: 1.5, interval: 0.5, splitLine: { show: false },
          axisLabel: { formatter: v => v === 0 ? '真值：正常' : v === 1 ? '真值：含暴力' : '', color: C.text2, fontSize: 12 } },
        yAxis: { type: 'value', min: Math.max(0, Math.floor(Math.min(...v.rows.map(r => r.score)) * 10) / 10), max: 1, name: '视频风险分' },
        series: [
          { name: `正常（${v.n - v.n_pos}）`, type: 'scatter', symbolSize: 9, itemStyle: { color: C.sub, opacity: 0.75 }, data: pts(0),
            markLine: { symbol: 'none', silent: true, data: [
              { yAxis: th.review, lineStyle: { color: C.review, type: 'dashed' }, label: { formatter: `复核 ${fmt(th.review, 3)}`, color: C.review, position: 'end' } },
              { yAxis: th.block, lineStyle: { color: C.block, type: 'dashed' }, label: { formatter: `拦截 ${fmt(th.block, 3)}`, color: C.block, position: 'end' } }] } },
          { name: `含暴力（${v.n_pos}）`, type: 'scatter', symbolSize: 9, itemStyle: { color: C.block, opacity: 0.75 }, data: pts(1) },
        ],
      };
    });
    const thresholdOpt = computed(() => {
      const v = x.value;
      if (!v) return null;
      const ts = v.thresholds;
      const line = (name, k, color, dashed) => ({ name, type: 'line', data: ts.map(t => t[k]), itemStyle: { color }, lineStyle: dashed ? { type: 'dashed' } : undefined, symbolSize: 7 });
      return {
        tooltip: { trigger: 'axis', valueFormatter: v => (+v).toFixed(3) }, legend: { top: 0 }, grid: { left: 40, right: 16, top: 36, bottom: 30 },
        xAxis: { type: 'category', data: ts.map(t => `阈值 ${(+t.threshold).toFixed(2)}`) }, yAxis: { type: 'value', min: 0, max: 1 },
        series: [line('精确率', 'precision', C.primary), line('召回率', 'recall', C.pass), line('F1', 'f1', C.purple), line('误报率', 'fpr', C.block, true)],
      };
    });
    const channelOpt = computed(() => {
      const v = x.value, o = d.value?.video_bench_xdv?.own_slice;
      if (!v) return null;
      const keys = [['auc', '融合'], ['auc_visual', '视觉'], ['auc_asr', '语音'], ['auc_ocr', '画面文字']];
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => fmt(v) }, legend: { top: 0 },
        grid: { left: 40, right: 40, top: 36, bottom: 26 },
        xAxis: { type: 'category', data: keys.map(k => k[1]) }, yAxis: { type: 'value', min: 0, max: 1 },
        series: [
          { name: `XD-Violence（n=${v.n}）`, type: 'bar', itemStyle: { color: C.primary }, data: keys.map(k => v[k[0]]),
            markLine: { symbol: 'none', silent: true, data: [{ yAxis: 0.5 }], lineStyle: { color: C.sub, type: 'dashed' }, label: { formatter: '随机', color: C.sub } } },
          ...(o ? [{ name: `自建中文切片（n=${o.n}）`, type: 'bar', itemStyle: { color: C.gold }, data: keys.map(k => o[k[0]]) }] : []),
        ],
      };
    });
    const promptOpt = computed(() => {
      const pt = d.value?.prompt_tuning;
      if (!pt) return null;
      const rows = [], cols = ['max', 'top10', 'top15', 'top30', 'mean'], data = [];
      Object.entries(pt.grid).forEach(([k, v]) => {
        const [ps, ca, ta] = k.split('|');
        const row = `${ps.replace(/^V\d_/, '')} · ${ca.split('=')[1]}`;
        if (!rows.includes(row)) rows.push(row);
        data.push([cols.indexOf(ta.split('=')[1]), rows.indexOf(row), v.auc]);
      });
      return {
        tooltip: { formatter: p => `${rows[p.value[1]]} · 时序聚合 ${cols[p.value[0]]}<br/>AUC <b>${p.value[2]}</b>` },
        grid: { left: 130, right: 16, top: 10, bottom: 40 },
        xAxis: { type: 'category', data: cols.map(c => `时序 ${c}`), splitArea: { show: true } },
        yAxis: { type: 'category', data: rows, splitArea: { show: true }, axisLabel: { fontSize: 11 } },
        visualMap: { min: 0.7, max: 1, show: false, inRange: { color: ['#feecec', '#fff2e6', '#e8efff', '#165dff'] } },
        series: [{ type: 'heatmap', data, label: { show: true, formatter: p => p.value[2].toFixed(3), fontSize: 11 },
          itemStyle: { borderColor: '#fff', borderWidth: 2 } }],
      };
    });

    // ---------- 文本 ----------
    const textOpt = computed(() => {
      const tb = d.value?.text_bench_combined;
      if (!tb) return null;
      const keys = Object.keys(SLICES).filter(k => tb[k]);
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: ps => `<b>${SLICES[keys[ps[0].dataIndex]]}</b>（n=${tb[keys[ps[0].dataIndex]].n}）<br/>` + ps.map(p => {
          const v = tb[keys[p.dataIndex]][EXP[p.seriesIndex][0]];
          return `${p.marker}${p.seriesName}：AUC ${fmt(v.auc)} · F1 ${fmt(v.f1)} · 召回 ${fmt(v.recall)}`;
        }).join('<br/>') },
        legend: { top: 0 }, grid: { left: 40, right: 16, top: 36, bottom: 26 },
        xAxis: { type: 'category', data: keys.map(k => `${SLICES[k]}（n=${tb[k].n}）`) }, yAxis: { type: 'value', min: 0, max: 1, name: 'AUC' },
        series: EXP.map(([k, name, color]) => ({ name, type: 'bar', itemStyle: { color }, data: keys.map(s => tb[s][k]?.auc),
          label: { show: true, position: 'top', formatter: p => (+p.value).toFixed(2), fontSize: 10.5, color: C.text2 } })),
      };
    });
    const obfOpt = computed(() => {
      const t = d.value?.text_bench_cold;
      if (!t) return null;
      const cond = [['clean', '原始文本'], ['obfuscated', '加字符混淆'], ['obfuscated_no_normalization', '混淆 + 关闭归一化']];
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => v == null ? '-' : fmt(v) },
        legend: { top: 0 }, grid: { left: 40, right: 16, top: 36, bottom: 26 },
        xAxis: { type: 'category', data: cond.map(c => c[1]) }, yAxis: { type: 'value', min: 0.4, max: 1, name: 'AUC' },
        series: EXP.map(([k, name, color]) => ({ name, type: 'bar', itemStyle: { color }, data: cond.map(([c]) => t[c]?.[k]?.auc ?? null) })),
      };
    });
    const topicOpt = computed(() => {
      const t = d.value?.text_bench_combined?.cold_by_topic_model_only;
      if (!t) return null;
      const names = { gender: '性别', race: '种族', region: '地域' }, ks = Object.keys(t);
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => fmt(v) }, legend: { top: 0 },
        grid: { left: 40, right: 16, top: 36, bottom: 26 },
        xAxis: { type: 'category', data: ks.map(k => names[k] || k) }, yAxis: { type: 'value', min: 0.6, max: 1 },
        series: [['auc', 'AUC', C.primary], ['f1', 'F1', C.cyan], ['recall', '召回', C.purple]].map(([k, n, c]) => ({ name: n, type: 'bar', itemStyle: { color: c }, data: ks.map(s => t[s][k]) })),
      };
    });

    // ---------- 图像对抗 ----------
    const ai = computed(() => d.value?.adv_image);
    const asrEpsOpt = computed(() => {
      const w = ai.value?.whitebox;
      if (!w) return null;
      const ks = Object.keys(w);
      return {
        tooltip: { trigger: 'axis' }, legend: { top: 0 }, grid: { left: 44, right: 54, top: 36, bottom: 30 },
        xAxis: { type: 'category', data: ks.map(epsLabel) },
        yAxis: [{ type: 'value', min: 0, max: 1, name: '攻击成功率', axisLabel: axisPct }, { type: 'value', name: 'PSNR (dB)', min: 20, max: 55, splitLine: { show: false } }],
        series: [
          { name: 'FGSM', type: 'line', data: ks.map(k => w[k].fgsm_asr), itemStyle: { color: C.review }, symbolSize: 8, tooltip: { valueFormatter: v => pct(v) } },
          { name: 'PGD', type: 'line', data: ks.map(k => w[k].pgd_asr), itemStyle: { color: C.block }, symbolSize: 8, tooltip: { valueFormatter: v => pct(v) } },
          { name: 'PGD 扰动的 PSNR', type: 'bar', yAxisIndex: 1, barMaxWidth: 22, data: ks.map(k => +w[k].pgd_psnr_db.toFixed(1)),
            itemStyle: { color: '#e8efff' }, z: 1, tooltip: { valueFormatter: v => v + ' dB' } },
        ],
      };
    });
    const bpdaOpt = computed(() => {
      const b = ai.value?.adaptive_bpda_by_eps;
      if (!b) return null;
      const eps = Object.keys(b), defs = Object.keys(b[eps[0]]);
      const colors = [C.primary, C.purple, C.cyan, C.gold];
      // 同一种防御的两条线共用一个名字，图例里只出现一次、一起开关；实线 = BPDA，虚线 = 静态攻击
      return {
        tooltip: { trigger: 'axis', formatter: ps => ps.map(p => `${p.marker}${p.seriesName} · ${p.seriesIndex % 2 ? 'BPDA 自适应' : '静态攻击'}：${pct(p.value)}`).join('<br/>') },
        legend: { top: 0, data: defs.map(df => IMG_DEF[df] || df) }, grid: { left: 44, right: 16, top: 36, bottom: 30 },
        xAxis: { type: 'category', data: eps.map(epsLabel) }, yAxis: { type: 'value', min: 0, max: 1, name: '防御后恢复率', axisLabel: axisPct },
        series: defs.flatMap((df, i) => [
          { name: IMG_DEF[df] || df, type: 'line', data: eps.map(e => b[e][df].static_recovered), itemStyle: { color: colors[i % 4] }, lineStyle: { type: 'dashed', width: 1.5 }, symbol: 'emptyCircle' },
          { name: IMG_DEF[df] || df, type: 'line', data: eps.map(e => b[e][df].adaptive_recovered), itemStyle: { color: colors[i % 4] }, symbolSize: 8 },
        ]),
      };
    });
    const naturalOpt = computed(() => {
      const n = ai.value?.natural;
      if (!n) return null;
      const ks = Object.keys(n);
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => pct(v) },
        grid: { left: 90, right: 30, top: 10, bottom: 24 },
        xAxis: { type: 'value', min: 0, max: 1, axisLabel: axisPct }, yAxis: { type: 'category', inverse: true, data: ks.map(k => IMG_NAT[k] || k) },
        series: [{ type: 'bar', barMaxWidth: 12, data: ks.map(k => ({ value: n[k].asr, itemStyle: { color: n[k].asr > 0 ? C.review : '#c9cdd4', borderRadius: [0, 3, 3, 0] } })) }],
      };
    });

    // ---------- 语音对抗 ----------
    const audioOpt = computed(() => {
      const a = d.value?.adv_audio?.summary?.attacks;
      if (!a) return null;
      const ks = Object.keys(a).sort((p, q) => a[q].escape_rate - a[p].escape_rate);
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: ps => { const v = a[ks[ps[0].dataIndex]];
          return `<b>${AUDIO_ATK[ks[ps[0].dataIndex]] || ks[ps[0].dataIndex]}</b><br/>端到端逃逸率 ${pct(v.escape_rate)}<br/>字错误率 CER ${fmt(v.mean_cer)}<br/>信噪比 ${v.mean_snr_db == null ? '-' : fmt(v.mean_snr_db, 1) + ' dB'}`; } },
        grid: { left: 130, right: 40, top: 10, bottom: 24 },
        xAxis: { type: 'value', min: 0, max: 1, axisLabel: axisPct }, yAxis: { type: 'category', inverse: true, data: ks.map(k => AUDIO_ATK[k] || k) },
        series: [{ type: 'bar', barMaxWidth: 12, data: ks.map(k => ({ value: a[k].escape_rate, itemStyle: { color: a[k].escape_rate > 0 ? C.block : '#c9cdd4', borderRadius: [0, 3, 3, 0] } })),
          label: { show: true, position: 'right', formatter: p => p.value > 0 ? pct(p.value) : '0', fontSize: 11, color: C.text2 } }],
      };
    });
    const audioDefOpt = computed(() => {
      const f = d.value?.adv_audio?.summary?.defenses;
      if (!f) return null;
      const ks = Object.keys(f);
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => pct(v, 1) }, grid: { left: 40, right: 16, top: 16, bottom: 26 },
        xAxis: { type: 'category', data: ks.map(k => AUDIO_DEF[k] || k) }, yAxis: { type: 'value', min: 0, max: 1, axisLabel: axisPct },
        series: [{ type: 'bar', barMaxWidth: 36, data: ks.map(k => ({ value: f[k].recovery_rate, itemStyle: { color: f[k].recovery_rate > 0 ? C.pass : C.block } })),
          label: { show: true, position: 'top', formatter: p => pct(p.value), color: C.text2 } }],
      };
    });

    // ---------- 视频对抗 ----------
    const videoAdvOpt = computed(() => {
      const s = d.value?.adv_video?.summary;
      if (!s) return null;
      const ks = Object.keys(s).sort((a, b) => s[b].mean_drop - s[a].mean_drop);
      const th = x.value?.adaptive_thresholds?.review;
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => fmt(v) }, legend: { top: 0 },
        grid: { left: 110, right: 70, top: 36, bottom: 24 },
        xAxis: { type: 'value', min: 0, max: 1 }, yAxis: { type: 'category', inverse: true, data: ks.map(k => k.replace(/^A\d+[a-z]?_/, '')) },
        series: [
          { name: '攻击前', type: 'bar', barMaxWidth: 10, itemStyle: { color: '#c9cdd4', borderRadius: [0, 3, 3, 0] }, data: ks.map(k => s[k].mean_before) },
          { name: '攻击后', type: 'bar', barMaxWidth: 10, data: ks.map(k => ({ value: s[k].mean_after, itemStyle: { color: s[k].escape_rate > 0 ? C.block : C.primary, borderRadius: [0, 3, 3, 0] } })),
            label: { show: true, position: 'right', formatter: p => `逃逸 ${pct(s[ks[p.dataIndex]].escape_rate)}`, fontSize: 11, color: C.text2 },
            markLine: th ? { symbol: 'none', silent: true, data: [{ xAxis: th }], lineStyle: { color: C.review, type: 'dashed' }, label: { formatter: '复核线', color: C.review } } : undefined },
        ],
      };
    });
    const budgetOpt = computed(() => {
      const b = d.value?.adv_video?.defenses;
      if (!b) return null;
      const ks = Object.keys(b);
      return {
        tooltip: { trigger: 'axis' }, legend: { top: 0 }, grid: { left: 44, right: 50, top: 36, bottom: 26 },
        xAxis: { type: 'category', data: ks.map(k => `采样预算 ${k.split('_')[1]} 帧`) },
        yAxis: [{ type: 'value', min: 0, max: 1, name: '攻击后平均分' }, { type: 'value', name: '耗时 (s)', splitLine: { show: false } }],
        series: [{ name: '攻击后平均分', type: 'bar', barMaxWidth: 30, itemStyle: { color: C.primary }, data: ks.map(k => b[k].mean_score) },
          { name: '平均耗时', type: 'line', yAxisIndex: 1, itemStyle: { color: C.review }, data: ks.map(k => b[k].mean_time_s), symbolSize: 8 }],
      };
    });

    // ---------- 深伪 ----------
    const df = computed(() => d.value?.deepfake_eval);
    const deepfakeOpt = computed(() => {
      const f = df.value;
      if (!f) return null;
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => fmt(v) },
        grid: { left: 40, right: 70, top: 16, bottom: 30 },
        xAxis: { type: 'category', data: ['同分布 · 真实人脸', '同分布 · 伪造人脸', '分布外 · 真实肖像'] },
        yAxis: { type: 'value', min: 0, max: 1, name: '平均"伪造"分' },
        series: [{ type: 'bar', barMaxWidth: 46, data: [
          { value: f.score_real?.mean, itemStyle: { color: C.pass } }, { value: f.score_fake?.mean, itemStyle: { color: C.sub } },
          { value: f.ood_real_faces?.mean, itemStyle: { color: C.block } }],
          label: { show: true, position: 'top', formatter: p => (+p.value).toFixed(3), color: C.text2 },
          markLine: { symbol: 'none', silent: true, data: [{ yAxis: 0.5 }], lineStyle: { color: C.review, type: 'dashed' }, label: { formatter: '判定线 0.5', color: C.review } } }],
      };
    });
    const jpegOpt = computed(() => {
      const j = df.value?.jpeg_robustness;
      if (!j) return null;
      const ks = Object.keys(j);
      return {
        tooltip: { trigger: 'axis', valueFormatter: v => fmt(v, 4) }, grid: { left: 50, right: 16, top: 16, bottom: 30 },
        xAxis: { type: 'category', data: ks.map(k => `JPEG ${k}`) }, yAxis: { type: 'value', min: 0.95, max: 1, name: 'AUC' },
        series: [{ type: 'line', data: ks.map(k => j[k].auc), itemStyle: { color: C.primary }, symbolSize: 8, areaStyle: { opacity: 0.08 } }],
      };
    });

    return {
      d, bot, tab, cards, goCard, x, scatterOpt, thresholdOpt, channelOpt, promptOpt, textOpt, obfOpt, topicOpt,
      ai, asrEpsOpt, bpdaOpt, naturalOpt, audioOpt, audioDefOpt, videoAdvOpt, budgetOpt, df, deepfakeOpt, jpegOpt, fmt, pct,
    };
  },
  template: `
  <div>
    <div class="page-head">
      <div>
        <h1>攻防实验</h1>
        <p>experiments/ 下脚本离线生成的全部结果。所有结论以公开基准为准，自建数据只作补充切片；标着「负面结果」的几项说明了当前技术的真实能力边界，是最值得讲的部分。</p>
      </div>
    </div>
    <el-card class="sn-card" shadow="never" v-if="d && bot">
      <el-tabs v-model="tab">
        <el-tab-pane label="总览" name="overview">
          <div class="kpi-grid" style="grid-template-columns:repeat(auto-fill,minmax(250px,1fr))">
            <div v-for="c in cards" :key="c.title" class="kpi" :class="c.negative ? 'tone-danger' : ''" style="cursor:pointer" @click="goCard(c)">
              <div class="kpi-icon"><el-icon><component :is="c.icon"/></el-icon></div>
              <div class="kpi-body">
                <div class="kpi-label">{{ c.title }}<span v-if="c.negative" class="neg-tag">负面结果</span></div>
                <div class="kpi-value">{{ c.value }}<small>{{ c.unit }}</small></div>
                <div class="kpi-hint">{{ c.desc }}</div>
              </div>
            </div>
          </div>
          <Takeaway>点击任意一张卡片跳到对应的实验。高分结果都来自同分布评估；跨分布、跨手法或面对自适应攻击时，同样的模型会出现上面四项负面结果。</Takeaway>
        </el-tab-pane>

        <el-tab-pane label="视频审核" name="video" lazy>
          <div class="kpi-grid" v-if="x">
            <KpiCard label="XD-Violence AUC" :value="fmt(x.auc)" icon="VideoCamera" :hint="'AP ' + fmt(x.ap) + ' · ' + x.n + ' 段（正例 ' + x.n_pos + '）'"/>
            <KpiCard label="视觉通道 AUC" :value="fmt(x.auc_visual)" icon="View" tone="purple"/>
            <KpiCard label="处理速度" :value="x.realtime_factor" suffix="× 实时" icon="Timer" tone="cyan" :hint="'平均每段 ' + fmt(x.mean_duration_s, 1) + 's，用时 ' + fmt(x.mean_elapsed_s, 1) + 's'"/>
            <KpiCard label="时序定位 IoU" :value="fmt(x.temporal?.mean_iou)" icon="Aim" tone="warning" :hint="'命中率 ' + pct(x.temporal?.mean_hit_rate) + '（' + x.temporal?.n + ' 段有帧级标注）'"/>
          </div>
          <div class="grid-2">
            <div><div class="chart-title">62 段视频的风险分分布 <span class="muted">按真值分两列，虚线是校准后的阈值</span></div><SnChart :option="scatterOpt" :height="320"/></div>
            <div><div class="chart-title">不同阈值下的表现</div><SnChart :option="thresholdOpt" :height="320"/></div>
          </div>
          <div class="grid-2" style="margin-top:16px">
            <div><div class="chart-title">各通道 AUC</div><SnChart :option="channelOpt" :height="260"/>
              <Takeaway>语音和画面文字通道面向中文违规话术，在以影视暴力为主的 XD-Violence 上接近或低于随机，整体判断主要靠视觉；它们的价值体现在自建中文切片上（语音通道 {{ fmt(d.video_bench_xdv?.own_slice?.auc_asr) }}）。自建切片上的 AUC {{ fmt(d.video_bench_xdv?.own_slice?.auc) }} 只有 8 段，且与规则同源，不能当作主要证据。</Takeaway></div>
            <div><div class="chart-title">提示词集 × 聚合方式网格搜索 <span class="muted">CLIP 零样本，AUC</span></div><SnChart :option="promptOpt" :height="260"/>
              <Takeaway>最好的组合是 <b>{{ d.prompt_tuning?.best?.prompt_set }} · 类目 {{ d.prompt_tuning?.best?.cat_agg }} · 时序 {{ d.prompt_tuning?.best?.temporal_agg }}</b>（AUC {{ fmt(d.prompt_tuning?.best?.auc) }}）；"更贴合类目"或"加困难负样本"的提示词反而更差。</Takeaway></div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="文本审核" name="text" lazy>
          <div class="grid-2">
            <div><div class="chart-title">三个切片上的 AUC <span class="muted">规则 / 模型 / 融合</span></div><SnChart :option="textOpt" :height="300"/></div>
            <div><div class="chart-title">字符混淆攻击 <span class="muted">COLD 测试集 {{ d.text_bench_cold?.n }} 条</span></div><SnChart :option="obfOpt" :height="300"/></div>
          </div>
          <Takeaway>COLD 里的冒犯语言大多是隐式的地域、性别、种族偏见，词表几乎抓不到（规则 AUC ≈ 0.5）；融合后在合并集上 AUC {{ fmt(d.text_bench_combined?.S3_combined?.fused?.auc) }}，高于任一单独专家。</Takeaway>
          <Takeaway negative>广告引流切片上规则 AUC 1.000 是循环论证：词表和这批自建数据出自同一套规则，只能说明"规则实现正确"，不能说明泛化能力。</Takeaway>
          <div class="grid-2" style="margin-top:16px">
            <div><div class="chart-title">COLD 按话题拆分 <span class="muted">只用模型</span></div><SnChart :option="topicOpt" :height="240"/></div>
            <div style="padding-top:28px"><Takeaway>加字符混淆后模型 AUC 从 {{ fmt(d.text_bench_cold?.clean?.model_only?.auc) }} 降到 {{ fmt(d.text_bench_cold?.obfuscated?.model_only?.auc) }}，降幅不大：COLD 原本就很少依赖可被混淆的显式脏词。对显式违规词，归一化防御才是关键（见「内容审核」页的混淆还原过程）。</Takeaway></div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="深伪专家" name="deepfake" lazy>
          <div class="kpi-grid" v-if="df">
            <KpiCard label="同分布测试集 AUC" :value="fmt(df.auc, 4)" icon="TrendCharts" tone="success" :hint="df.n + ' 张，等错误率 ' + pct(df.eer, 1)"/>
            <KpiCard label="JPEG q30 压缩后 AUC" :value="fmt(df.jpeg_robustness?.q30?.auc, 4)" icon="Picture" tone="cyan" hint="对压缩很鲁棒"/>
            <KpiCard label="分布外真实肖像误报率" :value="pct(df.ood_real_faces?.['fpr@0.5'])" icon="WarnTriangleFilled" tone="danger" negative :hint="df.ood_real_faces?.n + ' 张 ' + df.ood_real_faces?.source"/>
          </div>
          <div class="grid-2">
            <div><div class="chart-title">同分布 vs 分布外</div><SnChart :option="deepfakeOpt" :height="280"/></div>
            <div><div class="chart-title">JPEG 压缩鲁棒性 <span class="muted">同分布测试集</span></div><SnChart :option="jpegOpt" :height="280"/></div>
          </div>
          <Takeaway negative>同分布 AUC {{ fmt(df?.auc, 3) }}、对压缩也鲁棒，看起来近乎完美；但换成 Wikimedia 上的真实肖像，<b>{{ pct(df?.ood_real_faces?.['fpr@0.5']) }}</b> 被判为伪造。所以流水线只在实拍画面域里使用这个专家，并要求分数显著高于正常人脸的分布，不能直接上线。</Takeaway>
        </el-tab-pane>

        <el-tab-pane label="图像对抗" name="image" lazy>
          <div class="kpi-grid" v-if="ai">
            <KpiCard label="干净样本检出率" :value="pct(ai.clean_detection_rate)" icon="CircleCheck" tone="success" :hint="'被攻击模型：深伪专家，' + ai.n_attacked + ' 张伪造样本'"/>
            <KpiCard label="PGD · ε=1/255" :value="pct(ai.whitebox?.['eps_1/255']?.pgd_asr)" icon="Aim" tone="danger" :hint="'PSNR ' + fmt(ai.whitebox?.['eps_1/255']?.pgd_psnr_db, 1) + ' dB，肉眼不可见'"/>
            <KpiCard label="黑盒 SPSA · ε=16/255" :value="pct(ai.blackbox_spsa?.['eps_16/255']?.asr, 1)" icon="Lock" tone="warning" :hint="'平均 ' + ai.blackbox_spsa?.['eps_16/255']?.avg_queries + ' 次查询；ε=8/255 时 ' + pct(ai.blackbox_spsa?.['eps_8/255']?.asr)"/>
            <KpiCard label="特征压缩检测 AUC" :value="fmt(ai.feature_squeezing?.auc, 2)" icon="WarnTriangleFilled" tone="danger" negative hint="ε=1/255 时比随机还差"/>
          </div>
          <div class="grid-2">
            <div><div class="chart-title">攻击成功率随扰动强度 ε 的变化</div><SnChart :option="asrEpsOpt" :height="300"/></div>
            <div><div class="chart-title">防御面对静态攻击 vs 自适应攻击（BPDA） <span class="muted">虚线 = 静态，实线 = BPDA</span> <span class="neg-tag">负面结果</span></div><SnChart :option="bpdaOpt" :height="300"/></div>
          </div>
          <Takeaway negative>只对着"不知道有防御"的攻击评估时，JPEG、位深压缩、中值滤波看起来都能恢复一部分；攻击者用 BPDA 把防御考虑进去后，ε ≥ 2/255 时所有防御的恢复率都归零。只报静态防御效果会严重高估鲁棒性。</Takeaway>
          <div class="grid-2" style="margin-top:16px">
            <div><div class="chart-title">非梯度的自然扰动</div><SnChart :option="naturalOpt" :height="300"/></div>
            <div style="padding-top:28px"><Takeaway>压缩、缩放、加噪、翻转、旋转这类不利用梯度的扰动几乎都无效（缩小到 0.25 倍 {{ pct(ai?.natural?.['downscale_0.25']?.asr) }}、JPEG q10 {{ pct(ai?.natural?.jpeg_q10?.asr) }}，其余为 0），说明深伪专家对常规图像处理是鲁棒的，真正的威胁来自基于梯度的对抗扰动。</Takeaway></div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="语音对抗" name="audio" lazy>
          <div class="grid-2-1">
            <div><div class="chart-title">15 种攻击的端到端审核逃逸率 <span class="muted">逃逸 = 审核判定从违规变成通过</span></div><SnChart :option="audioOpt" :height="420"/></div>
            <div><div class="chart-title">防御恢复率</div><SnChart :option="audioDefOpt" :height="240"/>
              <Takeaway>变速、变调、加噪、电话带通、心理声学掩蔽等无差别攻击全部无效：SenseVoice 对这类信号失真很鲁棒。真正有效的是音量压到 5% 和定点消除关键词。</Takeaway>
              <Takeaway negative>重采样到 8kHz 的防御恢复率为 0；定点消除关键词这种"语义级"攻击，靠信号处理防御不了。</Takeaway></div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="视频对抗" name="video_adv" lazy>
          <div class="grid-2">
            <div><div class="chart-title">5 种攻击前后的平均风险分 <span class="muted">{{ d.adv_video?.n_targets }} 个违规视频</span></div><SnChart :option="videoAdvOpt" :height="300"/></div>
            <div><div class="chart-title">防御：增加关键帧采样预算</div><SnChart :option="budgetOpt" :height="300"/></div>
          </div>
          <Takeaway negative>最便宜的攻击最有效：不需要任何模型访问，把分辨率降到 0.4 倍就让 2/3 的违规视频逃过审核。关键帧躲避、镜头切换灌水这类针对采样策略的攻击，反而被快慢双通道关键帧选择挡住了。</Takeaway>
        </el-tab-pane>
      </el-tabs>
    </el-card>
    <el-skeleton v-else :rows="10" animated/>
  </div>`,
};
