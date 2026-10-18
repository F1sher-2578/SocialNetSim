// 水军控制台（红队）：发起 5 策略 × 3 规避等级的水军任务，量化它们对热门榜的影响
import {
  store, api, postJSON, bumpStats, openPost, fmt, pct, num, STRATEGIES, EVASION, campaignGroup,
} from '../lib/api.js';
import { C } from '../lib/charts.js';

const { ref, reactive, computed, watch, onMounted } = Vue;

const TARGET_HINT = {
  boost: { label: '目标帖子', placeholder: '选择要刷上热门的帖子', kind: 'post' },
  smear: { label: '目标帖子', placeholder: '选择要抹黑的帖子', kind: 'post' },
  hashtag_push: { label: '话题', placeholder: '#话题名', kind: 'topic' },
  follow_farm: { label: '目标用户 ID', placeholder: '要刷粉的用户 ID（留空则只互粉）', kind: 'user' },
  spam: { label: '目标', placeholder: '留空：自动选热门帖投放广告评论', kind: 'none' },
};
const ACTIONS = [['likes', '点赞', C.primary], ['comments', '评论', C.purple], ['posts', '发帖', C.gold], ['follows', '关注', C.cyan]];

export default {
  name: 'BotsPage',
  setup() {
    const form = reactive({ name: '', strategy: 'boost', target: '', n_bots: 10, intensity: 1.0, timing: 'burst', variation: 0.5, evasion: 0 });
    const hotPosts = ref([]), campaigns = ref([]), metrics = ref(null), exposure = ref(null), running = ref(false);

    async function loadAll() {
      const [f, c, m, e] = await Promise.all([api('/api/feed?mode=hot&limit=12'), api('/api/campaigns'),
        api('/api/bot_metrics'), api('/api/exposure?top_n=30')]);
      hotPosts.value = f.items; campaigns.value = c.campaigns; metrics.value = m; exposure.value = e;
    }
    onMounted(loadAll);
    watch(() => store.statsVersion, loadAll);
    watch(() => form.strategy, s => {
      form.target = s === 'hashtag_push' ? '#' : s === 'follow_farm' && store.meta.me ? String(store.meta.me.id) : '';
    });

    const hint = computed(() => TARGET_HINT[form.strategy]);
    const evasionStat = computed(() => {
      const m = metrics.value, L = form.evasion;
      if (!m) return null;
      return { seen: m.evasion_curve?.['evasion_' + L]?.['detect_rate@0.5'], mean: m.evasion_curve?.['evasion_' + L]?.mean_score,
        unseen: L > 0 ? m.evasion_generalization?.[`train<${L}_test=${L}`]?.['detect_rate@0.5'] : null };
    });
    const summary = computed(() => {
      const s = STRATEGIES[form.strategy], ev = EVASION[form.evasion];
      return `注册 ${form.n_bots} 个水军账号，以「${ev.name}」的方式执行「${s.name}」，${form.timing === 'burst' ? '集中在短时间内突发行动' : '把行动分散到更长时间'}；内容变体 ${form.variation.toFixed(1)}，强度 ${form.intensity.toFixed(1)}。`;
    });

    async function run() {
      running.value = true;
      try {
        const body = { ...form, name: form.name.trim() || `${STRATEGIES[form.strategy].name}-ev${form.evasion}`, target: form.target.trim() === '#' ? '' : form.target.trim() };
        const r = await postJSON('/api/campaign', body);
        const s = r.stats;
        ElementPlus.ElNotification({ type: 'success', title: `任务完成：${s.group}`, duration: 6000,
          message: `新增 ${s.bots} 个账号，点赞 ${s.likes}、评论 ${s.comments}、发帖 ${s.posts}、关注 ${s.follows}。到「水军检测」点「重新提取特征并评估」，看检测器能否识别这批新水军。` });
        bumpStats();
      } catch (e) { ElementPlus.ElMessage.error('执行失败：' + e.message); }
      finally { running.value = false; }
    }

    const detection = c => metrics.value?.campaign_level?.[campaignGroup(c)];
    const actionTotal = c => ACTIONS.reduce((n, [k]) => n + (c.stats?.[k] || 0), 0);

    const exposureOpt = computed(() => {
      const e = exposure.value;
      if (!e) return null;
      const rows = e.detail.slice(0, 30);
      return {
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: ps => {
          const d = rows[ps[0].dataIndex];
          return `<b>热门第 ${d.rank} 位</b> · 帖子 #${d.post_id}${d.author_is_bot ? '（水军自发）' : ''}<br/>` + ps.map(p => `${p.marker}${p.seriesName} ${p.value}`).join('<br/>')
            + `<br/>水军占比 <b>${pct(d.bot_like_ratio)}</b><br/><span style="color:#86909c">点击查看帖子</span>`;
        } },
        legend: { top: 0 }, grid: { left: 54, right: 20, top: 30, bottom: 20 },
        xAxis: { type: 'value', name: '点赞数' },
        yAxis: { type: 'category', inverse: true, data: rows.map(d => `#${d.rank}`), axisLabel: { fontSize: 10.5,
          color: v => (rows.find(d => `#${d.rank}` === v)?.author_is_bot ? C.block : C.sub) } },
        series: [
          { name: '真实用户点赞', type: 'bar', stack: 'l', barMaxWidth: 12, itemStyle: { color: '#b9ceff' }, data: rows.map(d => d.likes - d.bot_likes) },
          { name: '水军点赞', type: 'bar', stack: 'l', barMaxWidth: 12, itemStyle: { color: C.block, borderRadius: [0, 3, 3, 0] }, data: rows.map(d => d.bot_likes) },
        ],
      };
    });
    const onExposureClick = p => { const d = exposure.value?.detail[p.dataIndex]; if (d) openPost(d.post_id); };

    return {
      store, form, hint, hotPosts, campaigns, metrics, exposure, running, run, evasionStat, summary,
      STRATEGIES, EVASION, ACTIONS, detection, actionTotal, campaignGroup, exposureOpt, onExposureClick, fmt, pct, num,
    };
  },
  template: `
  <div>
    <div class="page-head">
      <div>
        <h1>水军控制台</h1>
        <p>红队视角：按策略和规避等级生成带真值标签的协同行为，用来训练和考验检测器。规避等级越高，账号越像真人，代价是控评效率下降。</p>
      </div>
    </div>
    <el-alert type="warning" :closable="false" show-icon style="margin-bottom:16px;border-radius:10px"
      title="本控制台只操作本机 SQLite 沙箱数据库，不与任何真实社交平台交互"
      description="目的只是生成带真值标签的水军行为数据，用于训练与评估检测模型。请勿将其中任何部分用于真实平台。"/>

    <el-card class="sn-card" shadow="never">
      <template #header><div class="card-title"><el-icon><Aim/></el-icon>发起水军任务</div></template>
      <div class="strategy-cards">
        <div v-for="(s, k) in STRATEGIES" :key="k" class="strategy-card" :class="{ on: form.strategy === k }" :style="{ '--c': s.color }" @click="form.strategy = k">
          <div class="sc-icon"><el-icon><component :is="s.icon"/></el-icon></div>
          <b>{{ s.name }}</b><p>{{ s.desc }}</p>
        </div>
      </div>
      <div class="grid-2-1" style="margin-top:18px">
        <el-form label-position="top" :model="form">
          <div class="grid-2" style="gap:0 20px">
            <el-form-item label="任务名称"><el-input v-model="form.name" :placeholder="STRATEGIES[form.strategy].name + '-ev' + form.evasion"/></el-form-item>
            <el-form-item :label="hint.label">
              <el-select v-if="hint.kind === 'post'" v-model="form.target" filterable allow-create clearable :placeholder="hint.placeholder" style="width:100%">
                <el-option v-for="p in hotPosts" :key="p.id" :value="String(p.id)" :label="'#' + p.id + ' ' + p.content.slice(0, 24)">
                  <span>#{{ p.id }} {{ p.content.slice(0, 22) }}</span><span class="muted" style="float:right">热门第 {{ p.rank }}</span></el-option>
              </el-select>
              <el-input v-else v-model="form.target" :placeholder="hint.placeholder" :disabled="hint.kind === 'none'"/>
            </el-form-item>
            <el-form-item label="账号数">
              <div class="row" style="width:100%;flex-wrap:nowrap"><el-slider v-model="form.n_bots" :min="2" :max="60" style="flex:1"/><el-input-number v-model="form.n_bots" :min="2" :max="60" size="small" style="width:96px"/></div>
            </el-form-item>
            <el-form-item label="行动强度"><el-slider v-model="form.intensity" :min="0.2" :max="3" :step="0.1" show-input :show-input-controls="false" input-size="small"/></el-form-item>
            <el-form-item label="内容变体（0 = 完全模板，1 = 全部改写）"><el-slider v-model="form.variation" :min="0" :max="1" :step="0.1" show-input :show-input-controls="false" input-size="small"/></el-form-item>
            <el-form-item label="行动节奏">
              <el-segmented v-model="form.timing" :options="[{ label: '集中突发', value: 'burst' }, { label: '分散执行', value: 'spread' }]"/>
            </el-form-item>
          </div>
          <el-form-item label="规避等级">
            <el-radio-group v-model="form.evasion">
              <el-tooltip v-for="e in EVASION" :key="e.level" :content="e.desc" placement="top">
                <el-radio-button :value="e.level">{{ e.level }} · {{ e.name }}</el-radio-button>
              </el-tooltip>
            </el-radio-group>
            <div class="form-hint" style="width:100%">{{ EVASION[form.evasion].desc }}</div>
          </el-form-item>
        </el-form>
        <div class="stack">
          <div class="evasion-hint" v-if="evasionStat">
            <div class="chart-title" style="margin-top:0">检测器对「{{ EVASION[form.evasion].name }}」的表现 <span class="muted">离线实验</span></div>
            <div class="grid-2" style="gap:10px">
              <div><div class="big" style="color:var(--sn-pass)">{{ pct(evasionStat.seen) }}</div>训练时见过这种手法</div>
              <div v-if="evasionStat.unseen !== null"><div class="big" style="color:var(--sn-block)">{{ pct(evasionStat.unseen, 1) }}</div>训练时没见过 <span class="neg-tag">负面结果</span></div>
              <div v-else><div class="big">{{ fmt(evasionStat.mean, 2) }}</div>平均检测分</div>
            </div>
            <div style="margin-top:8px" v-if="form.evasion > 0">同样是这批账号，检测器训练集里有没有同类样本，结果天差地别。</div>
          </div>
          <div class="evasion-hint"><b>这次任务会做什么：</b>{{ summary }}</div>
          <el-popconfirm title="确认在本地沙箱里执行这次水军任务？" confirm-button-text="执行" cancel-button-text="取消" @confirm="run" width="260">
            <template #reference><el-button type="primary" size="large" :loading="running" style="width:100%"><el-icon v-if="!running"><VideoPlay/></el-icon>&nbsp;执行任务</el-button></template>
          </el-popconfirm>
        </div>
      </div>
    </el-card>

    <div class="grid-2">
      <el-card class="sn-card" shadow="never">
        <template #header><div class="card-head"><div class="card-title"><el-icon><List/></el-icon>已执行任务 <span class="sub">{{ campaigns.length }} 个团伙</span></div>
          <span class="muted">检测结果来自最近一次评估</span></div></template>
        <el-table :data="campaigns" class="sn-table" max-height="470" row-key="id">
          <el-table-column type="expand">
            <template #default="{ row }">
              <div style="padding:4px 16px 4px 48px;font-size:12.5px;line-height:1.9">
                <div>团伙标识：<GroupTag :group="campaignGroup(row)"/> · 目标：{{ row.target || '-' }} · 强度 {{ row.intensity }} · {{ row.created_at }}</div>
                <div>产生行为：<span v-for="[k, n] in ACTIONS" :key="k" style="margin-right:12px">{{ n }} {{ row.stats[k] || 0 }}</span></div>
                <div v-if="detection(row)">检测：团伙内 {{ pct(detection(row).flagged_frac) }} 的账号被标记，平均检测分 {{ fmt(detection(row).mean_score) }}</div>
                <div v-else class="muted">这个团伙还没有参与评估，到「水军检测」重新评估后可见</div>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="团伙" min-width="176"><template #default="{ row }"><GroupTag :group="campaignGroup(row)"/></template></el-table-column>
          <el-table-column label="账号" prop="n_bots" width="58"/>
          <el-table-column label="规避" width="70"><template #default="{ row }"><EvasionDots :level="row.stats.evasion || 0"/></template></el-table-column>
          <el-table-column label="产生行为" min-width="110"><template #default="{ row }">
            <el-tooltip placement="top"><template #content><span v-for="[k, n] in ACTIONS" :key="k" style="margin-right:8px">{{ n }} {{ row.stats[k] || 0 }}</span></template>
              <div class="mini-stack"><i v-for="[k, n, c] in ACTIONS" :key="k" :style="{ width: (100 * (row.stats[k] || 0) / Math.max(1, actionTotal(row))) + '%', background: c }"></i></div>
            </el-tooltip></template></el-table-column>
          <el-table-column label="检测" width="104"><template #default="{ row }">
            <el-tag v-if="!detection(row)" type="info" size="small">未评估</el-tag>
            <el-tag v-else-if="detection(row).campaign_detected" type="success" size="small">检出 {{ pct(detection(row).flagged_frac) }}</el-tag>
            <el-tag v-else type="danger" size="small">漏过</el-tag></template></el-table-column>
        </el-table>
      </el-card>

      <el-card class="sn-card" shadow="never">
        <template #header><div class="card-title"><el-icon><TrendCharts/></el-icon>水军对热门榜的影响 <span class="sub">热门前 30 的点赞构成</span></div></template>
        <template v-if="exposure">
          <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:8px">
            <KpiCard label="被水军撑起的热门帖" :value="exposure.n_inflated" :suffix="'/ ' + exposure.top_n" icon="TrendCharts" tone="danger" hint="水军点赞占比 ≥ 30%"/>
            <KpiCard label="占热门榜比例" :value="pct(exposure.inflated_ratio)" icon="PieChart" tone="warning"/>
            <KpiCard label="水军自己发的热门帖" :value="exposure.bot_authored_in_top" icon="User" tone="purple" hint="纵轴标红的行"/>
          </div>
          <SnChart :option="exposureOpt" :height="300" @click="onExposureClick"/>
          <Takeaway>没有推荐排序，刷量控评就不会转化成曝光。热门前 {{ exposure.top_n }} 里有 <b>{{ exposure.n_inflated }}</b> 条帖子的点赞主要来自水军，普通用户刷到的"热门"有相当一部分是被制造出来的。</Takeaway>
        </template>
      </el-card>
    </div>
  </div>`,
};
