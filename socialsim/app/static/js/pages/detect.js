// 水军检测（蓝队）：关系网络、模型对比、特征消融、泛化与规避、账号排行
import {
  api, postJSON, openAccount, bumpStats, fmt, pct, groupColor, groupLabel, parseGroup,
  FEATURE_GROUPS, FEATURE_GROUP_COLORS, featureName,
} from '../lib/api.js';
import { C, axisPct } from '../lib/charts.js';

const { ref, computed, onMounted } = Vue;

const MODEL_NAMES = {
  RandomForest: '随机森林', GradientBoosting: '梯度提升', LogisticRegression: '逻辑回归', LightGBM: 'LightGBM',
  'IsolationForest(unsup)': '孤立森林（无监督）', RuleBaseline: '规则基线',
};
const MODEL_COLORS = {
  RandomForest: C.primary, GradientBoosting: C.cyan, LogisticRegression: C.purple, LightGBM: '#3491fa',
  'IsolationForest(unsup)': C.gold, RuleBaseline: C.grey,
};
const COMMUNITY_COLORS = ['#f53f3f', '#722ed1', '#165dff', '#0fc6c2', '#f7ba1e', '#eb0aa4', '#00b42a', '#f77234'];
const byGroupId = (a, b) => (+a.slice(1).split('-')[0]) - (+b.slice(1).split('-')[0]);

function graphOption(g, edgeMode, colorMode) {
  const nodes = g.nodes;
  let categories, catOf;
  if (colorMode === 'group') {
    const groups = [...new Set(nodes.filter(n => n.group).map(n => n.group))].sort(byGroupId);
    categories = [{ name: '正常用户', itemStyle: { color: '#c9cdd4' } },
      ...groups.map(gr => ({ name: groupLabel(gr), itemStyle: { color: groupColor(gr) } }))];
    catOf = n => n.group ? 1 + groups.indexOf(n.group) : 0;
  } else if (colorMode === 'community') {
    categories = [{ name: '未归入协同社区', itemStyle: { color: '#d5d8de' } },
      ...g.communities.map((c, i) => ({ name: `社区 ${c.id}（${c.size} 人）`, itemStyle: { color: COMMUNITY_COLORS[i % COMMUNITY_COLORS.length] } }))];
    catOf = n => n.community || 0;
  } else {
    categories = [{ name: '检测分 < 0.3', itemStyle: { color: C.pass } }, { name: '0.3 – 0.5', itemStyle: { color: C.review } },
      { name: '≥ 0.5（判为水军）', itemStyle: { color: C.block } }];
    catOf = n => n.bot_score >= 0.5 ? 2 : n.bot_score >= 0.3 ? 1 : 0;
  }
  const data = nodes.map(n => ({
    id: String(n.id), name: n.name, x: n.x, y: -n.y, category: catOf(n), value: n.bot_score, raw: n,
    symbolSize: n.is_bot ? 8 + n.bot_score * 8 : 6 + n.bot_score * 6,
    itemStyle: n.bot_flag ? { borderColor: '#f53f3f', borderWidth: 1.6 } : { borderColor: '#fff', borderWidth: 0.8 },
  }));
  const links = [];
  if (edgeMode !== 'sync') {
    for (const [a, b] of g.edges.follow) links.push({ source: String(a), target: String(b), kind: 'follow',
      // 轻微弯曲：一群账号同时关注同一个目标时，直线会叠成一条像边框的粗线
      lineStyle: { color: '#aab2c0', width: 0.6, opacity: edgeMode === 'both' ? 0.22 : 0.35, curveness: 0.18 } });
  }
  if (edgeMode !== 'follow') {
    for (const [a, b, w] of g.edges.sync) links.push({ source: String(a), target: String(b), kind: 'sync', w,
      lineStyle: { color: '#f53f3f', width: Math.min(4, 0.8 + w * 0.4), opacity: 0.8, curveness: 0.15 } });
  }
  return {
    tooltip: {
      formatter: p => {
        if (p.dataType === 'edge') return p.data.kind === 'sync' ? `协同行为：同一目标 15 分钟内共同行动 <b>${p.data.w}</b> 次` : '关注关系';
        const n = p.data.raw;
        return `<b>${n.avatar} ${n.name}</b><br/>真值：${n.is_bot ? `<span style="color:#f53f3f">水军</span> · ${groupLabel(n.group)}` : '正常用户'}`
          + `<br/>检测分：<b>${n.bot_score.toFixed(3)}</b>${n.bot_flag ? '（已标记）' : ''}`
          + `<br/>关注 ${n.following} · 粉丝 ${n.followers}${n.community ? `<br/>无监督协同社区 ${n.community}` : ''}`
          + `<br/><span style="color:#86909c">点击查看判定依据</span>`;
      },
    },
    legend: [{ data: categories.map(c => c.name), type: 'scroll', orient: 'vertical', right: 6, top: 12, bottom: 12,
      itemWidth: 10, itemHeight: 10, icon: 'circle', textStyle: { fontSize: 11.5 } }],
    series: [{
      type: 'graph', layout: 'none', roam: true, data, links, categories,
      left: 16, right: 190, top: 16, bottom: 16, scaleLimit: { min: 0.5, max: 8 },
      emphasis: { focus: 'adjacency', lineStyle: { width: 2.4, opacity: 1 } },
      blur: { itemStyle: { opacity: 0.12 }, lineStyle: { opacity: 0.04 } },
      animationDurationUpdate: 400,
    }],
  };
}

export default {
  name: 'DetectPage',
  setup() {
    const m = ref(null), g = ref(null), users = ref([]), running = ref(false), tab = ref('graph');
    const edgeMode = ref('both'), colorMode = ref('group');
    const filter = ref('all'), query = ref(''), page = ref(1), sort = ref({ prop: 'bot_score', order: 'descending' });

    async function load() {
      const [mm, gg, uu] = await Promise.all([api('/api/bot_metrics'), api('/api/graph'), api('/api/users?limit=1000')]);
      m.value = mm; g.value = gg; users.value = uu.users;
    }
    async function rerun() {
      running.value = true;
      try {
        await postJSON('/api/detect_bots');
        await load();
        bumpStats();
        ElementPlus.ElMessage.success('已重新提取 41 维特征并完成 5 折交叉验证');
      } catch (e) { ElementPlus.ElMessage.error('评估失败：' + e.message); }
      finally { running.value = false; }
    }
    onMounted(load);

    // ---------- KPI ----------
    const rf = computed(() => m.value?.models?.RandomForest || {});
    const unseen = computed(() => {
      const v = Object.values(m.value?.evasion_generalization || {}).map(x => x['detect_rate@0.5']);
      return v.length ? Math.min(...v) : null;
    });
    const sourceText = computed(() => !m.value ? '' : m.value.source === 'live'
      ? `网页即时评估 · ${m.value.live_at}` : `离线实验 · ${m.value.experiment_at}`);

    // ---------- 关系网络 ----------
    const graphOpt = computed(() => g.value && graphOption(g.value, edgeMode.value, colorMode.value));
    const graphFacts = computed(() => {
      if (!g.value) return null;
      const inFollow = new Set(g.value.edges.follow.flat()), inSync = new Set(g.value.edges.sync.flatMap(e => [e[0], e[1]]));
      const bots = g.value.nodes.filter(n => n.is_bot);
      const noFollow = bots.filter(n => !inFollow.has(n.id));
      return { bots: bots.length, noFollow: noFollow.length, rescued: noFollow.filter(n => inSync.has(n.id)).length,
        isolated: g.value.stats.n_isolated_bots };
    });
    const onGraphClick = p => { if (p.dataType === 'node') openAccount(+p.data.id); };

    // ---------- 模型对比 ----------
    const modelKeys = computed(() => Object.keys(m.value?.models || {}).filter(k => MODEL_NAMES[k]));
    const barOpt = computed(() => {
      const metrics = [['auc', 'AUC'], ['ap', 'AP'], ['f1', 'F1'], ['precision', '精确率'], ['recall', '召回率']];
      return {
        tooltip: { trigger: 'axis', valueFormatter: v => v == null ? '-' : (+v).toFixed(4) },
        legend: { top: 0 }, grid: { left: 40, right: 16, top: 40, bottom: 28 },
        xAxis: { type: 'category', data: metrics.map(x => x[1]) },
        yAxis: { type: 'value', min: 0, max: 1 },
        series: modelKeys.value.map(k => ({
          name: MODEL_NAMES[k], type: 'bar', itemStyle: { color: MODEL_COLORS[k] },
          data: metrics.map(([mk]) => m.value.models[k][mk] ?? null),
        })),
      };
    });
    const rocOpt = computed(() => {
      const roc = m.value?.roc || {};
      return {
        tooltip: { trigger: 'item', formatter: p => `${p.seriesName}<br/>FPR ${p.value[0]} · TPR ${p.value[1]}` },
        legend: { top: 0 }, grid: { left: 44, right: 16, top: 66, bottom: 40 },
        xAxis: { type: 'value', min: 0, max: 1, name: '误报率 FPR', nameLocation: 'middle', nameGap: 26 },
        yAxis: { type: 'value', min: 0, max: 1, name: '召回率 TPR', nameLocation: 'middle', nameGap: 32 },
        series: [
          ...Object.entries(roc).map(([k, [fpr, tpr]]) => ({
            name: `${MODEL_NAMES[k] || k}（AUC ${fmt(m.value.models[k]?.auc)}）`, type: 'line', showSymbol: false, step: false,
            itemStyle: { color: MODEL_COLORS[k] }, data: fpr.map((x, i) => [x, tpr[i]]),
          })),
          { name: '随机猜测', type: 'line', showSymbol: false, data: [[0, 0], [1, 1]],
            lineStyle: { type: 'dashed', color: C.grey, width: 1 }, itemStyle: { color: C.grey }, silent: true },
        ],
      };
    });
    const confusion = computed(() => {
      const u = users.value;
      const c = (truth, flag) => u.filter(x => x.is_bot === truth && x.bot_flag === flag).length;
      return [[c(false, false), c(false, true)], [c(true, false), c(true, true)]];
    });
    const cmOpt = computed(() => {
      const cm = confusion.value, max = Math.max(1, ...cm.flat());
      return {
        tooltip: { formatter: p => `${['真值正常', '真值水军'][p.value[1]]} · ${['判为正常', '判为水军'][p.value[0]]}：<b>${p.value[2]}</b>` },
        grid: { left: 70, right: 16, top: 10, bottom: 40 },
        xAxis: { type: 'category', data: ['判为正常', '判为水军'], splitArea: { show: true } },
        yAxis: { type: 'category', data: ['真值正常', '真值水军'], splitArea: { show: true } },
        visualMap: { min: 0, max, show: false, inRange: { color: ['#f2f6ff', '#165dff'] } },
        series: [{ type: 'heatmap', data: [[0, 0, cm[0][0]], [1, 0, cm[0][1]], [0, 1, cm[1][0]], [1, 1, cm[1][1]]],
          label: { show: true, fontSize: 18, fontWeight: 600, formatter: p => p.value[2], color: '#1d2129' },
          itemStyle: { borderColor: '#fff', borderWidth: 4, borderRadius: 6 } }],
      };
    });

    // ---------- 特征与消融 ----------
    const importanceOpt = computed(() => {
      const groupOf = {};
      Object.entries({
        profile: ['account_age_days', 'nick_digit_ratio', 'bio_empty', 'default_avatar', 'reg_burst'],
        content: ['n_texts', 'self_dup', 'cross_dup', 'len_cv', 'emoji_rate', 'punct_rate'],
        graph: ['n_follow_out', 'n_followers', 'ff_ratio', 'reciprocity', 'kcore', 'clustering'],
        coordination: ['sync_partners', 'sync_ratio', 'community_size', 'community_new_ratio', 'colike_jaccard_max', 'cocomment_jaccard_max', 'min_dt_median', 'xcorr_peak'],
      }).forEach(([gname, fs]) => fs.forEach(f => { groupOf[f] = gname; }));
      const top = Object.entries(m.value?.feature_importance || {}).slice(0, 15).reverse();
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => (+v).toFixed(4) },
        grid: { left: 150, right: 30, top: 10, bottom: 24 },
        xAxis: { type: 'value' }, yAxis: { type: 'category', data: top.map(([f]) => featureName(f)), axisLabel: { fontSize: 11.5, color: C.text2 } },
        series: [{ type: 'bar', barMaxWidth: 14, itemStyle: { borderRadius: [0, 3, 3, 0] },
          data: top.map(([f, v]) => ({ value: v, itemStyle: { color: FEATURE_GROUP_COLORS[groupOf[f] || 'behavior'] } })) }],
      };
    });
    const ablationOpt = computed(() => {
      const ab = m.value?.ablation || {}, keys = Object.keys(FEATURE_GROUPS);
      const all = ab.all?.auc;
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: ps => ps.map(p => {
          const v = ab[(p.seriesIndex === 0 ? 'only_' : 'drop_') + keys[p.dataIndex]] || {};
          return `${p.marker}${p.seriesName}：AUC ${fmt(v.auc, 4)} · F1 ${fmt(v.f1, 3)} · ${v.n_feat} 维`;
        }).join('<br/>') },
        legend: { top: 0 }, grid: { left: 70, right: 84, top: 34, bottom: 24 },
        xAxis: { type: 'value', min: 0.7, max: 1 },
        yAxis: { type: 'category', data: keys.map(k => FEATURE_GROUPS[k]), inverse: true },
        series: [
          { name: '只用该组', type: 'bar', barMaxWidth: 12, itemStyle: { color: C.primary, borderRadius: [0, 3, 3, 0] },
            data: keys.map(k => ab['only_' + k]?.auc ?? null),
            markLine: all ? { symbol: 'none', label: { formatter: `全部 41 维 ${fmt(all)}`, color: C.sub, fontSize: 11 },
              lineStyle: { color: C.sub, type: 'dashed' }, data: [{ xAxis: all }] } : undefined },
          { name: '去掉该组', type: 'bar', barMaxWidth: 12, itemStyle: { color: '#b9ceff', borderRadius: [0, 3, 3, 0] },
            data: keys.map(k => ab['drop_' + k]?.auc ?? null) },
        ],
      };
    });
    const ablationFacts = computed(() => {
      const ab = m.value?.ablation || {};
      const only = Object.keys(FEATURE_GROUPS).map(k => [k, ab['only_' + k]?.auc]).filter(x => x[1] != null).sort((a, b) => a[1] - b[1]);
      return { weakest: only[0], strongest: only[only.length - 1], dropProfile: ab.drop_profile?.auc };
    });

    // ---------- 泛化与规避 ----------
    const genOpt = computed(() => {
      const ga = m.value?.generalization?.all_features || {}, gn = m.value?.generalization?.no_profile_features || {};
      const groups = Object.keys(ga).filter(k => !k.startsWith('_')).sort(byGroupId);
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: ps => `<b>${groupLabel(groups[ps[0].dataIndex])}</b><br/>` + ps.map(p => {
          const src = p.seriesIndex === 0 ? ga : gn, v = src[groups[p.dataIndex]] || {};
          return `${p.marker}${p.seriesName}：最低分 ${fmt(v.min_score)} · 平均分 ${fmt(v.mean_score)} · 检出率 ${pct(v['detect_rate@0.5'])}`;
        }).join('<br/>') },
        legend: { top: 0 }, grid: { left: 40, right: 78, top: 36, bottom: 56 },
        xAxis: { type: 'category', data: groups.map(x => groupLabel(x).replace(' · ', '\n')), axisLabel: { interval: 0, fontSize: 10.5, lineHeight: 14 } },
        yAxis: { type: 'value', min: 0, max: 1, name: '团伙内最低检测分' },
        series: [
          { name: '全部特征', type: 'bar', data: groups.map(k => ({ value: ga[k]?.min_score, itemStyle: { color: groupColor(k) } })),
            markLine: { symbol: 'none', data: [{ yAxis: 0.5 }], lineStyle: { color: C.block, type: 'dashed' },
              label: { formatter: '判定线 0.5', color: C.block, fontSize: 11, position: 'end' } } },
          { name: '去掉画像特征', type: 'bar', data: groups.map(k => gn[k]?.min_score ?? null), itemStyle: { color: '#d5d8de' } },
        ],
      };
    });
    const genFacts = computed(() => {
      const ga = m.value?.generalization?.all_features || {};
      const rows = Object.entries(ga).filter(([k]) => !k.startsWith('_'));
      const worst = rows.sort((a, b) => a[1].min_score - b[1].min_score)[0];
      return { n: rows.length, detected: rows.filter(([, v]) => v['detect_rate@0.5'] >= 0.5).length, worst,
        fpr: ga['_normal_fpr@0.5'] };
    });
    const evasionOpt = computed(() => {
      const ev = m.value?.evasion_curve || {}, ks = Object.keys(ev).sort();
      const line = (name, key, color, dashed) => ({ name, type: 'line', data: ks.map(k => ev[k][key]), itemStyle: { color },
        lineStyle: dashed ? { type: 'dashed' } : undefined, symbolSize: 8 });
      return {
        tooltip: { trigger: 'axis', valueFormatter: v => (+v).toFixed(3) },
        legend: { top: 0 }, grid: { left: 40, right: 20, top: 36, bottom: 30 },
        xAxis: { type: 'category', data: ks.map(k => `规避等级 ${k.split('_')[1]}`), boundaryGap: false },
        yAxis: { type: 'value', min: 0, max: 1 },
        series: [line('检出率 @0.5', 'detect_rate@0.5', C.primary), line('检出率 @0.8', 'detect_rate@0.8', C.cyan),
          line('平均分', 'mean_score', C.purple, true), line('最低分', 'min_score', C.block, true)],
      };
    });
    const unseenOpt = computed(() => {
      const ev = m.value?.evasion_curve || {}, eg = m.value?.evasion_generalization || {};
      const levels = Object.keys(eg).map(k => +k.split('=').pop()).sort();
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: v => pct(v, 1) },
        legend: { top: 0 }, grid: { left: 44, right: 20, top: 36, bottom: 30 },
        xAxis: { type: 'category', data: levels.map(l => `检测规避等级 ${l}`) },
        yAxis: { type: 'value', min: 0, max: 1, axisLabel: axisPct },
        series: [
          { name: '训练时见过该等级', type: 'bar', barMaxWidth: 40, itemStyle: { color: C.pass },
            data: levels.map(l => ev['evasion_' + l]?.['detect_rate@0.5'] ?? null),
            label: { show: true, position: 'top', formatter: p => pct(p.value), color: C.text2 } },
          { name: '训练时没见过', type: 'bar', barMaxWidth: 40, itemStyle: { color: C.block },
            data: levels.map(l => eg[`train<${l}_test=${l}`]?.['detect_rate@0.5'] ?? null),
            label: { show: true, position: 'top', formatter: p => pct(p.value, 1), color: C.block, fontWeight: 600 } },
        ],
      };
    });

    // ---------- 账号排行 ----------
    const filtered = computed(() => {
      const q = query.value.trim().toLowerCase();
      let list = users.value.filter(u => filter.value === 'all' || (filter.value === 'bot' && u.is_bot)
        || (filter.value === 'human' && !u.is_bot) || (filter.value === 'flag' && u.bot_flag)
        || (filter.value === 'miss' && u.is_bot !== u.bot_flag));
      if (q) list = list.filter(u => (u.nickname + u.username + (u.bot_group || '')).toLowerCase().includes(q));
      const { prop, order } = sort.value;
      if (prop && order) {
        const dir = order === 'ascending' ? 1 : -1;
        list = [...list].sort((a, b) => (a[prop] > b[prop] ? 1 : a[prop] < b[prop] ? -1 : 0) * dir);
      }
      return list;
    });
    const paged = computed(() => filtered.value.slice((page.value - 1) * 12, page.value * 12));
    const counts = computed(() => ({
      all: users.value.length, bot: users.value.filter(u => u.is_bot).length, human: users.value.filter(u => !u.is_bot).length,
      flag: users.value.filter(u => u.bot_flag).length, miss: users.value.filter(u => u.is_bot !== u.bot_flag).length,
    }));
    const onSort = ({ prop, order }) => { sort.value = { prop, order }; page.value = 1; };

    return {
      m, g, users, running, rerun, tab, rf, unseen, sourceText, fmt, pct,
      edgeMode, colorMode, graphOpt, graphFacts, onGraphClick,
      barOpt, rocOpt, cmOpt, confusion, importanceOpt, ablationOpt, ablationFacts, FEATURE_GROUPS, FEATURE_GROUP_COLORS,
      genOpt, genFacts, evasionOpt, unseenOpt, groupLabel,
      filter, query, page, paged, filtered, counts, onSort, openAccount, parseGroup,
    };
  },
  template: `
  <div>
    <div class="page-head">
      <div>
        <h1>水军检测</h1>
        <p>41 维五组特征（画像 / 行为时序 / 内容 / 关系图 / 协同）+ 随机森林，5 折交叉验证。除了整体 AUC，更重要的是看跨团伙、跨规避等级的泛化：只报告一个高 AUC 是没有意义的。</p>
      </div>
      <div class="page-actions">
        <el-tag v-if="m" type="info" effect="plain" round><el-icon><Clock/></el-icon>&nbsp;{{ sourceText }}</el-tag>
        <el-button type="primary" :loading="running" @click="rerun"><el-icon v-if="!running"><Refresh/></el-icon>&nbsp;重新提取特征并评估</el-button>
      </div>
    </div>

    <div class="kpi-grid" v-if="m">
      <KpiCard label="检测 AUC（随机森林）" :value="fmt(rf.auc)" icon="TrendCharts" :hint="'F1 ' + fmt(rf.f1) + ' · 5 折交叉验证'"/>
      <KpiCard label="误报率 ≤ 1% 时的召回" :value="pct(rf['recall@fpr1%']?.recall, 1)" icon="Aim" tone="success" hint="真实风控更关心低误报区间"/>
      <KpiCard label="团伙级召回" :value="(m.campaign_level?._summary?.n_detected ?? '-') + ' / ' + (m.campaign_level?._summary?.n_campaigns ?? '-')" icon="Flag" tone="purple" hint="团伙内过半账号被标记即算检出"/>
      <KpiCard label="无监督协同发现" :value="m.unsupervised_coordination?.n_communities ?? '-'" suffix="个社区" icon="Connection" tone="cyan"
        :hint="'纯度 ' + pct(m.unsupervised_coordination?.mean_purity) + ' · 覆盖水军 ' + pct(m.unsupervised_coordination?.recall, 1) + '，完全不用标签'"/>
      <KpiCard label="没见过的规避手法" :value="unseen == null ? '-' : pct(unseen, 0)" icon="WarnTriangleFilled" tone="danger" negative
        hint="只用朴素水军训练，检测养号型水军的检出率"/>
    </div>

    <el-card class="sn-card" shadow="never" v-if="m">
      <el-tabs v-model="tab">
        <el-tab-pane label="关系网络" name="graph" lazy>
          <div class="graph-toolbar">
            <span><label>边</label><el-segmented v-model="edgeMode" size="small" :options="[{label:'叠加',value:'both'},{label:'关注关系',value:'follow'},{label:'协同行为',value:'sync'}]"/></span>
            <span><label>着色</label><el-segmented v-model="colorMode" size="small" :options="[{label:'真值团伙',value:'group'},{label:'无监督社区',value:'community'},{label:'检测分',value:'score'}]"/></span>
            <span class="muted">滚轮缩放 · 拖动平移 · 悬停高亮邻居 · 点击节点看判定依据</span>
          </div>
          <div class="grid-2-1" style="grid-template-columns:minmax(0,1fr) 280px">
            <div class="graph-wrap"><SnChart v-if="graphOpt" :option="graphOpt" :height="560" @click="onGraphClick"/></div>
            <div>
              <div class="legend-note" style="flex-direction:column;gap:8px;margin:0 0 12px">
                <span><i style="background:#aab2c0;border-radius:1px;height:2px;width:18px"></i>关注关系（{{ g.stats.n_follow }} 条）</span>
                <span><i style="background:#f53f3f;border-radius:1px;height:3px;width:18px"></i>协同行为：同一目标 15 分钟内共同行动 ≥2 次（{{ g.stats.n_sync }} 条）</span>
                <span><i style="background:#fff;border:2px solid #f53f3f"></i>红色描边 = 检测器已标记</span>
                <span><i style="background:#c9cdd4"></i>节点大小 = 检测分；右侧网格 = 两种关系都没有的账号</span>
              </div>
              <Takeaway v-if="graphFacts">
                {{ graphFacts.bots }} 个真值水军里，有 <b>{{ graphFacts.noFollow }}</b> 个没有任何关注边，只看关注图根本"高亮"不出来；
                切到「协同行为」后其中 {{ graphFacts.rescued }} 个通过同步点赞、评论暴露出来，剩下 {{ graphFacts.isolated }} 个只能靠画像和行为时序特征识别。
              </Takeaway>
              <div class="chart-title" style="margin-top:14px">无监督协同社区 <span class="muted">Louvain，不使用任何标签</span></div>
              <div class="community-list">
                <div class="cm" v-for="(c, i) in g.communities" :key="c.id">
                  <i :style="{ background: ['#f53f3f','#722ed1','#165dff','#0fc6c2','#f7ba1e','#eb0aa4'][i % 6] }"></i>
                  <span>社区 {{ c.id }} · {{ c.size }} 人</span>
                  <span class="muted">水军 {{ c.n_bots }}</span>
                  <span class="spacer"></span><GroupTag :group="c.dominant_group"/>
                </div>
              </div>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="模型对比" name="models" lazy>
          <div class="grid-2">
            <div><div class="chart-title">各模型指标 <span class="muted">5 折交叉验证</span></div><SnChart :option="barOpt" :height="300"/></div>
            <div><div class="chart-title">ROC 曲线</div><SnChart :option="rocOpt" :height="300"/></div>
          </div>
          <div class="grid-2" style="margin-top:16px">
            <div><div class="chart-title">混淆矩阵 <span class="muted">当前沙箱，判定线 0.5</span></div><SnChart :option="cmOpt" :height="220"/></div>
            <div>
              <div class="chart-title">怎么读这些数字</div>
              <Takeaway>监督模型在当前沙箱上几乎满分，但这是<b>同分布</b>评估：训练和测试里都有同样手法的水军。孤立森林完全不用标签，AUC 只有 {{ fmt(m.models['IsolationForest(unsup)']?.auc) }}，说明"异常 ≠ 水军"。</Takeaway>
              <Takeaway negative>满分不代表能用于真实平台：「泛化与规避」页显示，换一种训练时没见过的规避手法，检出率会直接掉到 {{ pct(unseen) }}。</Takeaway>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="特征与消融" name="features" lazy>
          <div class="grid-2">
            <div><div class="chart-title">特征重要性 Top 15 <span class="muted">颜色 = 特征组</span></div>
              <SnChart :option="importanceOpt" :height="380"/>
              <div class="legend-note"><span v-for="(n, k) in FEATURE_GROUPS" :key="k"><i :style="{ background: FEATURE_GROUP_COLORS[k] }"></i>{{ n }}</span></div>
            </div>
            <div><div class="chart-title">特征组消融 <span class="muted">只用 / 去掉某一组时的 AUC</span></div>
              <SnChart :option="ablationOpt" :height="300"/>
              <Takeaway v-if="ablationFacts.weakest">
                单独使用时 <b>{{ FEATURE_GROUPS[ablationFacts.weakest[0]] }}</b> 最弱（AUC {{ fmt(ablationFacts.weakest[1]) }}）：内容可以被 LLM 改写去模板化；
                <b>{{ FEATURE_GROUPS[ablationFacts.strongest[0]] }}</b> 最强（{{ fmt(ablationFacts.strongest[1]) }}）。
                去掉最容易伪造的画像特征后 AUC 仍为 {{ fmt(ablationFacts.dropProfile) }}，检测并不依赖"新号、默认头像"这类表面信号。
              </Takeaway>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="泛化与规避" name="general" lazy>
          <div class="chart-title">留一团伙泛化 <span class="muted">训练时剔除该团伙，再检测它；柱子是团伙内最低的检测分</span></div>
          <SnChart :option="genOpt" :height="300"/>
          <Takeaway v-if="genFacts.worst">
            {{ genFacts.n }} 个团伙在"没见过该团伙"的条件下检出 {{ genFacts.detected }} 个，正常用户误报率 {{ pct(genFacts.fpr, 1) }}；
            但 <b>{{ groupLabel(genFacts.worst[0]) }}</b> 里最难的账号只有 {{ fmt(genFacts.worst[1].min_score) }} 分，已经逼近判定线。
          </Takeaway>
          <div class="grid-2" style="margin-top:18px">
            <div><div class="chart-title">规避强度 → 检测分 <span class="muted">训练集包含全部规避等级</span></div><SnChart :option="evasionOpt" :height="280"/></div>
            <div><div class="chart-title">训练时没见过的规避手法 <span class="neg-tag"><el-icon><WarnTriangleFilled/></el-icon>负面结果</span></div><SnChart :option="unseenOpt" :height="280"/></div>
          </div>
          <Takeaway negative>
            只用朴素水军（规避等级 0）训练时，养号型水军（等级 1）的检出率是 <b>{{ pct(unseen) }}</b>；即使训练集加入了养号样本，深度伪装型（等级 2）也只检出
            <b>{{ pct(m.evasion_generalization?.['train<2_test=2']?.['detect_rate@0.5'], 1) }}</b>。检测器学到的是"见过的手法"，而不是"水军"本身。这是本项目最重要的负面结果。
          </Takeaway>
        </el-tab-pane>

        <el-tab-pane :label="'账号排行（' + users.length + '）'" name="accounts" lazy>
          <div class="row" style="margin-bottom:12px">
            <el-segmented v-model="filter" @change="page = 1" :options="[
              {label:'全部 ' + counts.all, value:'all'}, {label:'真值水军 ' + counts.bot, value:'bot'}, {label:'正常用户 ' + counts.human, value:'human'},
              {label:'检测标记 ' + counts.flag, value:'flag'}, {label:'判错 ' + counts.miss, value:'miss'}]"/>
            <span class="spacer"></span>
            <el-input v-model="query" placeholder="搜索昵称 / 账号 / 团伙" clearable style="width:220px" @input="page = 1"><template #prefix><el-icon><Search/></el-icon></template></el-input>
          </div>
          <el-table :data="paged" class="sn-table clickable" @row-click="r => openAccount(r.id)" @sort-change="onSort"
            :default-sort="{ prop: 'bot_score', order: 'descending' }">
            <el-table-column label="账号" min-width="190"><template #default="{ row }"><UserCell :user="row"/></template></el-table-column>
            <el-table-column label="真值" width="96"><template #default="{ row }">
              <el-tag v-if="row.is_bot" type="danger" size="small">水军</el-tag><el-tag v-else type="success" size="small">正常</el-tag></template></el-table-column>
            <el-table-column label="团伙" min-width="150"><template #default="{ row }"><GroupTag :group="row.bot_group"/></template></el-table-column>
            <el-table-column prop="bot_score" label="检测分" min-width="170" sortable="custom"><template #default="{ row }"><ScoreBar :score="row.bot_score"/></template></el-table-column>
            <el-table-column label="判定" width="100"><template #default="{ row }">
              <el-tag v-if="row.bot_flag" type="danger" effect="dark" size="small">已标记</el-tag><el-tag v-else type="info" size="small">未标记</el-tag></template></el-table-column>
            <el-table-column prop="created_at" label="注册" width="110" sortable="custom"/>
            <el-table-column prop="posts" label="帖子" width="80" sortable="custom"/>
            <el-table-column prop="followers" label="粉丝" width="80" sortable="custom"/>
          </el-table>
          <div class="row" style="justify-content:flex-end;margin-top:12px">
            <el-pagination v-model:current-page="page" :page-size="12" layout="total, prev, pager, next" :total="filtered.length" background/>
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>
    <el-skeleton v-else :rows="8" animated/>
  </div>`,
};
