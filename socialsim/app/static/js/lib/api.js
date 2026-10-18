// 接口封装、全局状态和各页面共用的领域常量（判定、风险标签、水军策略与团伙配色）
const { reactive } = Vue;

export const store = reactive({
  meta: { sim_now: '', me: null, thresholds: { review: 0.4, block: 0.7 } },
  lens: false,               // 审核透视：在用户端界面上叠加平台视角的信息
  account: { id: null, admin: true },
  postId: null,
  statsVersion: 0,           // 发帖、跑任务、重新评估后自增，提醒各页刷新统计
});

export async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); if (j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

export const postJSON = (path, body) => api(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}),
});

/** 轮询后台任务直到结束；onTick 每次轮询都会收到任务当前状态 */
export function pollTask(id, { interval = 1200, onTick } = {}) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const t = await api('/api/task/' + id);
        onTick && onTick(t);
        if (t.status === 'running') return setTimeout(tick, interval);
        if (t.status === 'done') resolve(t);
        else reject(Object.assign(new Error(t.error || '任务失败'), { task: t }));
      } catch (e) { reject(e); }
    };
    tick();
  });
}

export const openAccount = (id, admin = true) => { store.account = { id, admin }; };
export const openPost = id => { store.postId = id; };
export const bumpStats = () => { store.statsVersion++; };

// ---------------------------------------------------------------- 格式化
export const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(+x)) ? '-' : (+x).toFixed(d);
export const pct = (x, d = 0) => (x === null || x === undefined) ? '-' : `${(x * 100).toFixed(d)}%`;
export const num = x => (x ?? 0).toLocaleString('zh-CN');

/**
 * 微博式时间：今天 19:18 / 昨天 20:12 / 09-15 08:30。
 * 沙箱数据用的是仿真时间（种子会生成仿真当天全天的帖子），"今天"以仿真日期为准；
 * 在网页上新发的帖子用的是真实时间，晚于仿真日期，显示为刚刚。
 */
export function relTime(s) {
  if (!s) return '';
  const ref = store.meta.sim_now ? new Date(store.meta.sim_now.replace(' ', 'T')) : new Date();
  const t = new Date(s.replace(' ', 'T'));
  const day = d => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const hm = s.slice(11, 16), days = Math.round((day(ref) - day(t)) / 86400000);
  if (days < 0) return '刚刚';
  const m = Math.round((ref - t) / 60000);
  if (m >= 0 && m < 60) return m < 1 ? '刚刚' : `${m} 分钟前`;
  if (days === 0) return `今天 ${hm}`;
  if (days === 1) return `昨天 ${hm}`;
  return t.getFullYear() === ref.getFullYear() ? `${s.slice(5, 10)} ${hm}` : s.slice(0, 10);
}

// ---------------------------------------------------------------- 判定与风险
export const VERDICT = {
  pass: { text: '通过', type: 'success', color: '#00b42a', icon: 'CircleCheckFilled' },
  review: { text: '待复核', type: 'warning', color: '#ff7d00', icon: 'WarningFilled' },
  block: { text: '拦截', type: 'danger', color: '#f53f3f', icon: 'CircleCloseFilled' },
  pending: { text: '待审', type: 'info', color: '#86909c', icon: 'Clock' },
};
export const verdictOf = s => s >= store.meta.thresholds.block ? 'block' : s >= store.meta.thresholds.review ? 'review' : 'pass';

/** 分数旁边同时给离散等级（参照 Botometer / Azure 严重度），避免把分数误读成概率 */
export function riskLevel(score, th = store.meta.thresholds) {
  if (score >= th.block) return { text: '高', color: '#f53f3f' };
  if (score >= th.review) return { text: '中', color: '#ff7d00' };
  return { text: '低', color: '#00b42a' };
}

/** 水军检测分的配色：0.5 是检测器的判定线 */
export const botColor = s => s >= 0.5 ? '#f53f3f' : s >= 0.3 ? '#ff7d00' : '#00b42a';

const LABELS = {
  contact_wechat: '微信引流', contact_qq: 'QQ 引流', contact_phone: '手机号引流', external_link: '外部链接',
  short_link: '短链接', spam: '垃圾广告', fraud: '诈骗', gambling: '赌博', drugs: '毒品', porn: '色情',
  violence: '暴力', insult: '辱骂', offensive: '冒犯言论', obfuscation: '刻意混淆', gore: '血腥', weapon: '武器',
  smoking_drug: '烟草/毒品', protest_crowd: '人群聚集', fire_disaster: '火灾/灾害', horror: '惊悚',
  nsfw: '色情', nsfw_cartoon: '色情（动画）', deepfake_suspect: '疑似深度伪造', unreadable: '无法解析', error: '处理异常',
  system_prompt_leak: '系统提示泄露', system_prompt_marker: '系统提示特征',
};
const DOMAINS = { cartoon: '动画', game: '游戏', text_screen: '文字画面', photo: '实拍' };
const PREFIXES = { asr_: '语音·', ocr_: '画面文字·', content_: '输出·' };

export function labelName(l) {
  if (!l) return '';
  for (const [p, name] of Object.entries(PREFIXES)) if (l.startsWith(p)) return name + labelName(l.slice(p.length));
  if (LABELS[l]) return LABELS[l];
  const m = l.match(/^(.*)_(cartoon|game|text_screen)$/);
  if (m && LABELS[m[1]]) return `${LABELS[m[1]]}（${DOMAINS[m[2]]}）`;
  return l;
}
export const splitLabels = v => (Array.isArray(v) ? v : String(v || '').split(',')).map(s => s.trim()).filter(Boolean);

// ---------------------------------------------------------------- 水军策略与团伙
export const STRATEGIES = {
  boost: { name: '控评刷量', icon: 'Top', color: '#165dff', desc: '集中点赞、转发、好评，把目标帖推上热门' },
  smear: { name: '抹黑攻击', icon: 'ChatLineRound', color: '#eb0aa4', desc: '在目标帖下集中发布负面、辱骂评论' },
  hashtag_push: { name: '话题带节奏', icon: 'Promotion', color: '#722ed1', desc: '短时间集中发同一话题并互赞，制造热搜' },
  follow_farm: { name: '互粉抱团', icon: 'Connection', color: '#0fc6c2', desc: '批量互相关注，刷高目标的粉丝数' },
  spam: { name: '广告引流', icon: 'Message', color: '#f7ba1e', desc: '在热门帖下发布引流、诈骗类评论' },
};
export const EVASION = [
  { level: 0, name: '朴素水军', desc: '新注册、集中突发、模板化内容' },
  { level: 1, name: '养号 + 分散', desc: '提前 30–180 天养号、行动分散、内容改写' },
  { level: 2, name: '深度伪装', desc: '在 1 的基础上用真人化昵称/头像/简介，并混入大量正常行为' },
];

/** "C6-boost-ev1" → { id: 'C6', strategy: 'boost', evasion: 1 } */
export function parseGroup(g) {
  const m = String(g || '').match(/^(C\d+)-([a-z_]+?)(?:-ev(\d))?$/);
  return m ? { id: m[1], strategy: m[2], evasion: m[3] ? +m[3] : 0 } : null;
}
export const campaignGroup = c => `C${c.id}-${c.strategy}${c.stats?.evasion ? `-ev${c.stats.evasion}` : ''}`;

function mix(hex, target, p) {
  const a = hex.match(/\w\w/g).map(x => parseInt(x, 16)), b = target.match(/\w\w/g).map(x => parseInt(x, 16));
  return '#' + a.map((v, i) => Math.round(v * (1 - p) + b[i] * p).toString(16).padStart(2, '0')).join('');
}
/** 色相 = 策略，明度 = 规避等级（ev0 深 → ev2 浅）；关系图、热力图和表格标签共用，保证同一团伙颜色一致 */
export function groupColor(g) {
  const p = parseGroup(g);
  if (!p) return '#c9cdd4';
  const base = (STRATEGIES[p.strategy] || {}).color || '#86909c';
  return p.evasion === 0 ? mix(base, '#000000', 0.18) : p.evasion === 1 ? base : mix(base, '#ffffff', 0.38);
}
export function groupLabel(g) {
  const p = parseGroup(g);
  if (!p) return g || '';
  return `${p.id} ${(STRATEGIES[p.strategy] || {}).name || p.strategy}${p.evasion ? ` · ev${p.evasion}` : ''}`;
}

// ---------------------------------------------------------------- 水军检测特征
export const FEATURE_GROUPS = {
  profile: '画像', behavior: '行为时序', content: '内容', graph: '关系图', coordination: '协同',
};
export const FEATURE_GROUP_COLORS = {
  profile: '#86909c', behavior: '#165dff', content: '#f7ba1e', graph: '#0fc6c2', coordination: '#722ed1',
};
export const FEATURES = {
  account_age_days: '账号年龄（天）', nick_digit_ratio: '昵称数字占比', bio_empty: '简介为空', default_avatar: '默认头像',
  reg_burst: '同期批量注册', n_actions: '行为总数', actions_per_day: '日均行为数', active_days: '活跃天数',
  n_posts: '发帖数', n_comments: '评论数', n_likes: '点赞数', hour_entropy: '活跃时段熵', night_ratio: '深夜行为占比',
  iet_cv: '行为间隔变异系数', iet_median_min: '行为间隔中位数（分钟）', burst_ratio: '突发行为占比',
  comment_like_ratio: '评论/点赞比', target_concentration: '目标集中度', unique_target_ratio: '目标去重率',
  iet_ks_vs_human: '间隔分布与真人的 KS 距离', seq_compress_ratio: '行为序列可压缩率', n_texts: '文本数',
  self_dup: '自身内容重复度', cross_dup: '跨账号内容重复度', len_cv: '文本长度变异系数', emoji_rate: '表情占比',
  punct_rate: '标点占比', n_follow_out: '关注数', n_followers: '粉丝数', ff_ratio: '关注/粉丝比', reciprocity: '互关率',
  kcore: 'k-核数', clustering: '聚类系数', sync_partners: '同步伙伴数', sync_ratio: '同步行为占比',
  community_size: '所在协同社区规模', community_new_ratio: '社区内新号占比', colike_jaccard_max: '共同点赞最大重合度',
  cocomment_jaccard_max: '共同评论最大重合度', min_dt_median: '与他人最小时间差中位数', xcorr_peak: '活动曲线互相关峰值',
};
export const featureName = f => FEATURES[f] || f;

// ---------------------------------------------------------------- 文本
const SEG_RE = /(#[^\s#，。！？,.!?]+)|(@[\w一-龥]+)/g;
/** 把正文拆成文本 / 话题 / 提及片段，模板里用 v-for 渲染（用户内容一律不用 v-html） */
export function segments(text) {
  const s = text || '', out = [];
  let last = 0;
  for (const m of s.matchAll(SEG_RE)) {
    if (m.index > last) out.push({ t: 'text', v: s.slice(last, m.index) });
    out.push({ t: m[1] ? 'topic' : 'mention', v: m[0] });
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push({ t: 'text', v: s.slice(last) });
  return out;
}
