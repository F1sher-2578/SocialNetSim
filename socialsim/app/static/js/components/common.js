// 全站通用的小组件：判定徽章、风险等级、水军分徽章、团伙标签、KPI 卡、结论条等
import {
  VERDICT, riskLevel, botColor, labelName, splitLabels, groupColor, groupLabel, fmt,
} from '../lib/api.js';

const { computed } = Vue;

/** 三态判定徽章（通过 / 待复核 / 拦截），全站统一 */
export const VerdictTag = {
  props: { verdict: String, size: { type: String, default: 'small' }, effect: { type: String, default: 'light' } },
  setup(p) { return { m: computed(() => VERDICT[p.verdict] || { text: p.verdict || '-', type: 'info', icon: 'InfoFilled' }) }; },
  template: `<el-tag class="verdict-tag" :type="m.type" :size="size" :effect="effect" disable-transitions>
    <el-icon><component :is="m.icon"/></el-icon>{{ m.text }}</el-tag>`,
};

/** 分数 + 离散等级："0.873 · 高" */
export const RiskLevel = {
  props: { score: Number, digits: { type: Number, default: 3 } },
  setup(p) { return { lv: computed(() => riskLevel(p.score || 0)), fmt }; },
  template: `<span class="risk"><i :style="{ background: lv.color }"></i>{{ fmt(score, digits) }}<em>· {{ lv.text }}</em></span>`,
};

/** 水军检测分换算成 0–5 分显示（参照 Botometer：防止把分数误读成概率） */
export const BotBadge = {
  props: { score: Number },
  setup(p) {
    const c = computed(() => botColor(p.score || 0));
    return { c, v: computed(() => ((p.score || 0) * 5).toFixed(1)) };
  },
  template: `<el-tooltip :content="'水军检测分 ' + (score ?? 0).toFixed(3) + '，换算为 0–5 分显示'" placement="top">
    <span class="bot-badge" :style="{ color: c, background: c + '14' }"><el-icon><Cpu/></el-icon>{{ v }}<small>/5</small></span>
  </el-tooltip>`,
};

/** 带阈值着色的分数条 */
export const ScoreBar = {
  props: { score: Number, color: String, digits: { type: Number, default: 3 } },
  setup(p) { return { c: computed(() => p.color || botColor(p.score || 0)), fmt }; },
  template: `<span class="score-bar"><span class="track"><span class="fill" :style="{ width: Math.min(100, (score || 0) * 100) + '%', background: c }"></span></span><span>{{ fmt(score, digits) }}</span></span>`,
};

export const GroupTag = {
  props: { group: String },
  setup(p) {
    const c = computed(() => groupColor(p.group));
    return { c, label: computed(() => groupLabel(p.group)) };
  },
  template: `<span v-if="group" class="group-tag" :style="{ color: c, borderColor: c + '55', background: c + '12' }"><i :style="{ background: c }"></i>{{ label }}</span>
    <span v-else class="muted">-</span>`,
};

export const EvasionDots = {
  props: { level: { type: Number, default: 0 } },
  template: `<el-tooltip :content="'规避等级 ' + level" placement="top"><span class="ev-dots"><i v-for="i in 3" :key="i" :class="{ on: i <= level }"></i></span></el-tooltip>`,
};

export const LabelTags = {
  props: { labels: [String, Array], max: { type: Number, default: 8 }, size: { type: String, default: 'small' } },
  setup(p) { return { list: computed(() => splitLabels(p.labels)), labelName }; },
  template: `<span class="label-tags" v-if="list.length">
      <el-tag v-for="l in list.slice(0, max)" :key="l" :size="size" type="info" effect="plain" round>{{ labelName(l) }}</el-tag>
      <el-tag v-if="list.length > max" :size="size" type="info" effect="plain" round>+{{ list.length - max }}</el-tag>
    </span><span v-else class="muted">无风险标签</span>`,
};

export const KpiCard = {
  props: {
    label: String, value: [String, Number], suffix: String, hint: String, icon: { type: String, default: 'DataLine' },
    tone: { type: String, default: '' }, negative: Boolean,
  },
  template: `<div class="kpi" :class="tone ? 'tone-' + tone : ''">
    <div class="kpi-icon"><el-icon><component :is="icon"/></el-icon></div>
    <div class="kpi-body">
      <div class="kpi-label">{{ label }}<span v-if="negative" class="neg-tag"><el-icon><WarnTriangleFilled/></el-icon>负面结果</span></div>
      <div class="kpi-value">{{ value ?? '-' }}<small v-if="suffix">{{ suffix }}</small></div>
      <div class="kpi-hint" v-if="hint">{{ hint }}</div>
    </div>
  </div>`,
};

/** 每张图下面的一句话结论；negative 用红色调标出负面结果 */
export const Takeaway = {
  props: { negative: Boolean },
  template: `<div class="takeaway" :class="{ negative }"><el-icon><component :is="negative ? 'WarnTriangleFilled' : 'Opportunity'"/></el-icon><div><slot/></div></div>`,
};

/** 用户头像单元格（表格里用） */
export const UserCell = {
  props: { user: Object },
  template: `<div class="user-cell"><div class="av">{{ user.avatar || '👤' }}</div><div style="min-width:0"><b>{{ user.nickname }}</b><small>@{{ user.username }}</small></div></div>`,
};

export function registerCommon(app) {
  Object.entries({ VerdictTag, RiskLevel, BotBadge, ScoreBar, GroupTag, EvasionDots, LabelTags, KpiCard, Takeaway, UserCell })
    .forEach(([k, c]) => app.component(k, c));
}
