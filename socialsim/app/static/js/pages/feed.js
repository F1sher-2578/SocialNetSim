// 信息流：普通用户看到的平台。「审核透视」打开后叠加平台视角（真值、检测分、判定、水军推高）
import {
  store, api, postJSON, segments, relTime, openAccount, openPost, bumpStats, groupColor, fmt, pct, num,
} from '../lib/api.js';

const { ref, computed, watch, onMounted } = Vue;

const TRIES = [
  { label: '正常动态', text: '周末去大唐不夜城逛了逛，人超多但很热闹 #西安 #周末' },
  { label: '短链推广', text: '新品上架，详情戳 t.cn/A6x9Kq 了解一下' },
  { label: '混淆引流', text: '加 微 信 xk8899 领取内部优惠，兼职日结，名额有限' },
  { label: '拼音辱骂', text: '楼上真是个 naocan，别出来丢人了' },
];

const RECEIPT = {
  pass: { icon: 'CircleCheckFilled', color: '#00b42a', title: '发布成功，已通过审核', desc: '内容已进入信息流。' },
  review: { icon: 'WarningFilled', color: '#ff7d00', title: '已发布，但可见性受限', desc: '风险分超过复核阈值，已进入人工复核队列；复核完成前其他用户会看到「可见性受限」提示。' },
  block: { icon: 'CircleCloseFilled', color: '#f53f3f', title: '发布失败：内容违反社区规定', desc: '风险分超过拦截阈值，已被系统拦截，其他用户看不到这条内容。' },
};

const PostItem = {
  props: { p: Object, inflate: Number },
  setup(props) {
    const open = ref(false);
    const ring = computed(() => store.lens && props.p.author.is_bot ? groupColor(props.p.author.group) : 'transparent');
    const cmd = c => c === 'author' ? openAccount(props.p.author.id, store.lens) : openPost(props.p.id);
    return { store, open, ring, cmd, segments, relTime, openAccount, openPost };
  },
  template: `
  <article class="post">
    <div class="post-av" :style="{ '--ring': ring }" @click="openAccount(p.author.id, store.lens)">{{ p.author.avatar }}</div>
    <div class="post-body">
      <div class="post-head">
        <span class="nick" @click="openAccount(p.author.id, store.lens)">{{ p.author.nickname }}</span>
        <span class="handle">@{{ p.author.username }} · {{ relTime(p.created_at) }}</span>
        <template v-if="store.lens">
          <el-tag v-if="p.author.is_bot" type="danger" size="small" effect="dark">真值：水军</el-tag>
          <BotBadge :score="p.author.bot_score"/>
        </template>
        <el-dropdown trigger="click" @command="cmd" style="margin-left:auto">
          <span class="more"><el-icon><MoreFilled/></el-icon></span>
          <template #dropdown><el-dropdown-menu>
            <el-dropdown-item command="detail" icon="Document">查看详情与评论</el-dropdown-item>
            <el-dropdown-item command="author" icon="User">查看作者主页</el-dropdown-item>
          </el-dropdown-menu></template>
        </el-dropdown>
      </div>
      <div v-if="p.mod_status === 'block' && !store.lens && !open" class="post-blocked">
        <el-icon><Hide/></el-icon>该内容因违反社区规定已被拦截
        <el-link type="primary" :underline="false" style="margin-left:auto" @click="open = true">仍要查看</el-link>
      </div>
      <div v-else class="post-text" @click="openPost(p.id)"><template v-for="(s, i) in segments(p.content)" :key="i"><span v-if="s.t !== 'text'" :class="s.t">{{ s.v }}</span><template v-else>{{ s.v }}</template></template></div>
      <div v-if="p.mod_status === 'review' && !store.lens" class="post-limited"><el-icon><WarningFilled/></el-icon>可见性受限：该内容疑似违规，正在等待人工复核</div>
      <div v-if="store.lens" class="post-lens">
        <span class="lens-k"><el-icon><View/></el-icon>平台视角</span>
        <VerdictTag :verdict="p.mod_status"/>
        <RiskLevel :score="p.risk_score"/>
        <LabelTags v-if="p.risk_labels" :labels="p.risk_labels" :max="3"/>
        <span v-if="inflate >= 0.3" class="inflate-badge"><el-icon><TrendCharts/></el-icon>水军推高 {{ Math.round(inflate * 100) }}%</span>
        <span class="muted" style="margin-left:auto">热门第 {{ p.rank }} 位 · 热度 {{ p.hot.toFixed(2) }}</span>
      </div>
      <div class="post-actions">
        <span @click="openPost(p.id)"><el-icon><ChatDotRound/></el-icon>{{ p.comments }}</span>
        <span class="repost static"><el-icon><RefreshRight/></el-icon>{{ p.reposts }}</span>
        <span class="like static"><el-icon><Star/></el-icon>{{ p.likes }}</span>
        <span class="static"><el-icon><Share/></el-icon></span>
      </div>
    </div>
  </article>`,
};

export default {
  name: 'FeedPage',
  components: { PostItem },
  setup() {
    const draft = ref(''), publishing = ref(false), receipt = ref(null);
    const mode = ref('hot'), showBlocked = ref(false);
    const items = ref([]), loading = ref(false), hasMore = ref(true);
    const trending = ref([]), stats = ref(null), suspects = ref([]), meStats = ref(null);
    const inflation = ref({});
    let offset = 0;

    async function loadFeed(reset = true) {
      if (reset) offset = 0;
      loading.value = true;
      try {
        const q = new URLSearchParams({ mode: mode.value, limit: 20, offset, hide_blocked: !showBlocked.value });
        if (mode.value === 'following' && store.meta.me) q.set('user_id', store.meta.me.id);
        const d = await api('/api/feed?' + q);
        items.value = reset ? d.items : items.value.concat(d.items);
        hasMore.value = d.items.length === 20;
        offset += d.items.length;
      } catch (e) { ElementPlus.ElMessage.error('加载信息流失败：' + e.message); }
      finally { loading.value = false; }
    }
    async function loadSide() {
      const [t, s, u] = await Promise.all([api('/api/trending?top_n=10'), api('/api/stats'), api('/api/users?limit=5')]);
      trending.value = t.topics; stats.value = s; suspects.value = u.users;
      if (store.meta.me) meStats.value = (await api('/api/user/' + store.meta.me.id)).user;
    }
    async function loadInflation() {
      const d = await api('/api/exposure?top_n=30');
      inflation.value = Object.fromEntries(d.detail.map(x => [x.post_id, x.bot_like_ratio]));
    }

    async function publish() {
      const text = draft.value.trim();
      if (!text) return;
      publishing.value = true;
      try {
        const r = await postJSON('/api/post', { content: text });
        const m = r.moderation || {};
        receipt.value = { id: r.post_id, verdict: m.verdict || 'pass', score: m.score || 0, labels: m.labels || [], experts: r.experts || {} };
        draft.value = '';
        bumpStats();
      } catch (e) { ElementPlus.ElMessage.error('发布失败：' + e.message); }
      finally { publishing.value = false; }
    }

    const rc = computed(() => receipt.value && RECEIPT[receipt.value.verdict]);
    const lensSummary = computed(() => {
      const n = items.value.length;
      return { n, bots: items.value.filter(p => p.author.is_bot).length,
        inflated: items.value.filter(p => (inflation.value[p.id] || 0) >= 0.3).length };
    });
    const hotMark = (t, i) => i === 0 ? ['boil', '沸'] : i < 3 ? ['hot', '热'] : t.posts <= 15 ? ['new', '新'] : null;

    watch([mode, showBlocked], () => loadFeed(true));
    watch(() => store.lens, on => { if (on) loadInflation(); });
    watch(() => store.statsVersion, () => { loadFeed(true); loadSide(); if (store.lens) loadInflation(); });
    watch(() => store.meta.me, me => { if (me && !meStats.value) loadSide(); });
    onMounted(() => { loadFeed(true); loadSide(); });

    return {
      store, TRIES, draft, publishing, publish, receipt, rc, mode, showBlocked, items, loading, hasMore, loadFeed,
      trending, stats, suspects, meStats, inflation, lensSummary, hotMark, openAccount, fmt, pct, num,
    };
  },
  template: `
  <div class="feed-page">
    <div class="feed-main">
      <div class="feed-card composer">
        <div class="av">{{ store.meta.me?.avatar || '🙂' }}</div>
        <div class="composer-body">
          <el-input v-model="draft" type="textarea" :autosize="{ minRows: 2, maxRows: 6 }" maxlength="280"
            placeholder="有什么新鲜事？（发布后会立即经过多专家审核）" @keydown.ctrl.enter="publish" @keydown.meta.enter="publish"/>
          <div class="composer-foot">
            <div class="tries"><span class="muted" style="line-height:24px">试试：</span>
              <span v-for="t in TRIES" :key="t.label" class="try-chip" @click="draft = t.text">{{ t.label }}</span></div>
            <span class="count">{{ draft.length }}/280</span>
            <el-button type="primary" round :loading="publishing" :disabled="!draft.trim()" @click="publish">发布</el-button>
          </div>
          <transition name="receipt">
            <div v-if="receipt" class="receipt" :class="receipt.verdict">
              <el-progress type="circle" :percentage="Math.round(receipt.score * 100)" :width="56" :stroke-width="6" :color="rc.color">
                <template #default><span style="font-size:13px;font-weight:600">{{ receipt.score.toFixed(2) }}</span></template>
              </el-progress>
              <div class="receipt-main">
                <h4><el-icon :style="{ color: rc.color }"><component :is="rc.icon"/></el-icon>{{ rc.title }}</h4>
                <p>{{ rc.desc }}（复核阈值 {{ store.meta.thresholds.review }}，拦截阈值 {{ store.meta.thresholds.block }}）</p>
                <div style="margin-top:6px"><LabelTags :labels="receipt.labels"/></div>
                <div class="experts">
                  <span v-if="receipt.experts.rule">规则专家 {{ fmt(receipt.experts.rule.score) }}</span>
                  <span v-if="receipt.experts.cold && receipt.experts.cold.offensive_prob !== undefined">COLD 冒犯语言模型 {{ fmt(receipt.experts.cold.offensive_prob) }}</span>
                  <span v-if="receipt.experts.rule && receipt.experts.rule.obfuscated" style="color:#cc6400">检测到刻意混淆</span>
                  <span>帖子 #{{ receipt.id }}</span>
                </div>
              </div>
              <el-icon class="close" @click="receipt = null"><Close/></el-icon>
            </div>
          </transition>
        </div>
      </div>

      <div class="feed-toolbar">
        <el-segmented v-model="mode" :options="[{ label: '热门', value: 'hot' }, { label: '关注', value: 'following' }]"/>
        <span class="spacer"></span>
        <span class="opt">显示被拦截内容 <el-switch v-model="showBlocked" size="small"/></span>
      </div>

      <div class="feed-card timeline" v-loading="loading && !items.length">
        <PostItem v-for="p in items" :key="p.id" :p="p" :inflate="store.lens ? (inflation[p.id] || 0) : 0"/>
        <div v-if="!items.length && !loading" class="feed-empty">这里还没有内容</div>
        <div v-if="items.length" class="feed-more">
          <el-button v-if="hasMore" text type="primary" :loading="loading" @click="loadFeed(false)">加载更多</el-button>
          <span v-else class="muted">没有更多了</span>
        </div>
      </div>
    </div>

    <aside class="feed-side">
      <div class="feed-card side-card" v-if="store.meta.me">
        <div class="me-card">
          <div class="av">{{ store.meta.me.avatar }}</div>
          <div style="min-width:0"><b>{{ store.meta.me.nickname }}</b><p>@{{ store.meta.me.username }}</p><p>{{ store.meta.me.bio }}</p></div>
        </div>
        <div class="me-stats" v-if="meStats">
          <div><b>{{ meStats.posts }}</b><span>帖子</span></div>
          <div><b>{{ meStats.following }}</b><span>关注</span></div>
          <div><b>{{ meStats.followers }}</b><span>粉丝</span></div>
        </div>
      </div>

      <div v-if="!store.lens" class="lens-tip">
        想知道平台眼中的这条时间线是什么样？打开右上角的 <b>「审核透视」</b>：每条帖子会叠加真值标签、水军检测分和审核判定，热搜榜会显示各话题的水军发帖占比。
      </div>
      <div v-else class="lens-tip">
        <b>审核透视已开启。</b>当前加载的 {{ lensSummary.n }} 条帖子中，真值水军发布了 <b>{{ lensSummary.bots }}</b> 条；
        热门里被水军点赞撑起来（水军赞 ≥ 30%）的有 <b>{{ lensSummary.inflated }}</b> 条。
      </div>

      <div class="feed-card side-card">
        <h3><el-icon color="#f53f3f"><TrendCharts/></el-icon>热搜榜 <span class="muted">按 6 小时内讨论人数排序</span></h3>
        <div v-for="(t, i) in trending" :key="t.topic" class="trend">
          <span class="rank" :class="i < 3 ? 'top' + (i + 1) : ''">{{ i + 1 }}</span>
          <div class="t-main">
            <span class="t-name">#{{ t.topic }}</span>
            <span class="t-sub">{{ t.peak_authors }} 人在讨论 · {{ t.posts }} 条帖子</span>
          </div>
          <span v-if="store.lens" class="bot-ratio" :style="{ color: t.bot_ratio > 0.6 ? '#f53f3f' : '#86909c', background: t.bot_ratio > 0.6 ? '#feecec' : '#f2f3f5' }">水军 {{ pct(t.bot_ratio) }}</span>
          <i v-else-if="hotMark(t, i)" class="hot-mark" :class="hotMark(t, i)[0]">{{ hotMark(t, i)[1] }}</i>
        </div>
      </div>

      <div class="feed-card side-card" v-if="stats">
        <h3><el-icon color="#165dff"><DataBoard/></el-icon>平台概览</h3>
        <div class="mini-stats">
          <div><b>{{ num(stats.users) }}</b><span>用户</span></div>
          <div><b>{{ num(stats.posts) }}</b><span>帖子</span></div>
          <div><b>{{ num(stats.comments) }}</b><span>评论</span></div>
          <div><b>{{ num(stats.likes) }}</b><span>点赞</span></div>
          <template v-if="store.lens">
            <div class="danger"><b>{{ stats.bots }}</b><span>真值水军</span></div>
            <div><b>{{ stats.flagged }}</b><span>检测器标记</span></div>
            <div class="danger"><b>{{ stats.blocked_posts }}</b><span>被拦截帖子</span></div>
            <div><b>{{ stats.review_posts }}</b><span>待复核帖子</span></div>
          </template>
        </div>
      </div>

      <div class="feed-card side-card" v-if="store.lens && suspects.length">
        <h3><el-icon color="#ff7d00"><Warning/></el-icon>可疑账号 <span class="muted">检测分最高</span></h3>
        <div v-for="u in suspects" :key="u.id" class="suspect" @click="openAccount(u.id)">
          <div class="av">{{ u.avatar }}</div>
          <div class="s-main"><b>{{ u.nickname }}</b><GroupTag v-if="u.bot_group" :group="u.bot_group"/></div>
          <BotBadge :score="u.bot_score"/>
        </div>
      </div>
      <div class="side-foot">SocialNetSim · 课程实验《应用综合实验》<br>所有数据均为本地仿真，不连接任何真实平台</div>
    </aside>
  </div>`,
};
