// 账号画像与帖子详情两个全局抽屉（参照 Bluesky Ozone：对象在上，证据和事件在下）
import {
  store, api, segments, relTime, fmt, pct, groupColor, featureName, FEATURE_GROUPS, FEATURE_GROUP_COLORS,
  openAccount, botColor,
} from '../lib/api.js';
import { gaugeOption, radarOption, C } from '../lib/charts.js';

const { ref, computed, watch } = Vue;

let usersCache = null, usersVersion = -1;
async function allUsers() {
  if (!usersCache || usersVersion !== store.statsVersion) {
    usersCache = (await api('/api/users?limit=1000')).users;
    usersVersion = store.statsVersion;
  }
  return usersCache;
}

export const AccountDrawer = {
  setup() {
    const visible = ref(false), loading = ref(false), data = ref(null), users = ref([]);
    const admin = computed(() => store.account.admin);
    watch(() => store.account, async a => {
      if (!a.id) return;
      visible.value = true; loading.value = true; data.value = null;
      try {
        const [d, us] = await Promise.all([api('/api/user/' + a.id), allUsers()]);
        data.value = d; users.value = us;
      } catch (e) {
        ElementPlus.ElMessage.error('加载账号失败：' + e.message);
      } finally { loading.value = false; }
    });
    const u = computed(() => data.value?.user);
    const score = computed(() => u.value?.bot_score ?? 0);
    // 类似 Botometer 的 CAP：检测分不低于该账号的账号里，真值水军占多少
    const cap = computed(() => {
      if (!u.value || !users.value.length) return null;
      const above = users.value.filter(x => x.bot_score >= score.value - 1e-9);
      return { n: above.length, ratio: above.filter(x => x.is_bot).length / Math.max(1, above.length) };
    });
    const ring = computed(() => admin.value && u.value?.is_bot ? groupColor(u.value.bot_group) : 'transparent');
    const gauge = computed(() => gaugeOption(score.value, {
      title: '水军检测分', colors: [[0.3, C.pass], [0.5, C.review], [1, C.block]], th: { review: 0.3, block: 0.5 },
    }));
    const radar = computed(() => radarOption(data.value?.explain?.group_profile, botColor(score.value)));
    const evidence = computed(() => data.value?.explain?.evidence || []);
    const goDetect = () => { visible.value = false; location.hash = '#/detect'; };
    return {
      store, visible, loading, data, u, admin, score, cap, ring, gauge, radar, evidence, goDetect,
      fmt, pct, relTime, featureName, FEATURE_GROUPS, FEATURE_GROUP_COLORS,
    };
  },
  template: `
  <el-drawer v-model="visible" :size="520" class="sn-drawer" :title="admin ? '账号画像 · 平台视角' : '用户主页'" append-to-body>
    <div v-loading="loading" style="min-height:200px">
      <template v-if="u">
        <div class="drawer-section">
          <div class="profile-head" :style="{ '--ring': ring }">
            <div class="av">{{ u.avatar }}</div>
            <div style="min-width:0">
              <h3>{{ u.nickname }}
                <template v-if="admin">
                  <el-tag v-if="u.is_bot" type="danger" size="small" effect="dark">真值：水军</el-tag>
                  <el-tag v-else type="success" size="small">真值：正常用户</el-tag>
                  <GroupTag v-if="u.bot_group" :group="u.bot_group"/>
                </template>
              </h3>
              <p>@{{ u.username }} · {{ u.created_at }} 注册</p>
              <p v-if="u.bio" style="color:var(--sn-text-2)">{{ u.bio }}</p>
              <p v-else>这个人很懒，什么都没写</p>
            </div>
          </div>
          <div class="profile-stats">
            <div><b>{{ u.posts }}</b><span>帖子</span></div>
            <div><b>{{ u.following }}</b><span>关注</span></div>
            <div><b>{{ u.followers }}</b><span>粉丝</span></div>
          </div>
        </div>

        <template v-if="admin">
          <div class="drawer-section">
            <h4><el-icon><Odometer/></el-icon>检测结论
              <el-tag v-if="u.bot_flag" type="danger" size="small" style="margin-left:auto">检测器已标记</el-tag>
              <el-tag v-else type="info" size="small" style="margin-left:auto">未标记</el-tag>
            </h4>
            <SnChart :option="gauge" :height="170"/>
            <div class="cap-note" v-if="cap">
              检测分 ≥ <b>{{ fmt(score) }}</b> 的 {{ cap.n }} 个账号中，真值水军占 <b :style="{ color: cap.ratio > 0.5 ? 'var(--sn-block)' : 'var(--sn-pass)' }">{{ pct(cap.ratio, 1) }}</b>。
              <span class="muted">（参照 Botometer 的 CAP：给出"这个分数意味着什么"的经验解释，而不是把分数当概率）</span>
            </div>
          </div>
          <div class="drawer-section" v-if="data.explain && data.explain.group_profile">
            <h4><el-icon><Aim/></el-icon>五组特征画像 <span class="muted">朝水军方向偏离的平均分位，虚线为全体中位数</span></h4>
            <SnChart :option="radar" :height="250"/>
          </div>
          <div class="drawer-section" v-if="evidence.length">
            <h4><el-icon><List/></el-icon>关键证据 <span class="muted">按"模型权重 × 偏离程度"排序</span></h4>
            <div class="evidence-row muted" style="border-bottom:1px solid #f0f1f3"><span>特征</span><span>取值</span><span>在全体中的分位</span></div>
            <div class="evidence-row" v-for="e in evidence" :key="e.feature">
              <div class="feat"><b>{{ featureName(e.feature) }}</b>
                <small><span :style="{ color: FEATURE_GROUP_COLORS[e.group] }">●</span> {{ FEATURE_GROUPS[e.group] || e.group }} · 权重 {{ fmt(e.weight) }}</small></div>
              <span class="num">{{ fmt(e.value, e.value >= 100 ? 0 : 3) }}</span>
              <ScoreBar :score="e.percentile" :digits="2" :color="e.percentile > 0.8 || e.percentile < 0.2 ? '#ff7d00' : '#86909c'"/>
            </div>
          </div>
        </template>
        <el-alert v-else type="info" :closable="false" show-icon style="margin-bottom:14px"
          title="这是普通用户能看到的主页" description="打开右上角「审核透视」，可以看到平台对这个账号的检测结论和判定依据。"/>

        <div class="drawer-section">
          <h4><el-icon><Document/></el-icon>最近发帖</h4>
          <div v-if="!data.posts.length" class="muted">还没有发过帖子</div>
          <div v-for="p in data.posts" :key="p.id" class="comment">
            <div class="c-body">
              <div>{{ p.content }}</div>
              <div class="c-head" style="margin-top:4px"><span>{{ relTime(p.created_at) }} · ♥ {{ p.likes }} · 💬 {{ p.comments }}</span>
                <VerdictTag v-if="admin" :verdict="p.mod_status"/></div>
            </div>
          </div>
        </div>
        <el-button v-if="admin" type="primary" plain style="width:100%" @click="goDetect"><el-icon><Search/></el-icon>&nbsp;在水军检测中查看全部账号</el-button>
      </template>
    </div>
  </el-drawer>`,
};

export const PostDrawer = {
  setup() {
    const visible = ref(false), loading = ref(false), data = ref(null), shown = ref({});
    watch(() => store.postId, async id => {
      if (!id) return;
      visible.value = true; loading.value = true; data.value = null; shown.value = {};
      try { data.value = await api('/api/post/' + id); }
      catch (e) { ElementPlus.ElMessage.error('加载帖子失败：' + e.message); }
      finally { loading.value = false; }
    });
    const reportUrl = computed(() => {
      const p = data.value?.post?.mod_report;
      return p && p.includes('/results/') ? '/reports/' + p.split('/results/')[1] : '';
    });
    const close = () => { visible.value = false; store.postId = null; };
    return { store, visible, loading, data, shown, segments, relTime, reportUrl, openAccount, close };
  },
  template: `
  <el-drawer v-model="visible" :size="520" class="sn-drawer" :title="store.lens ? '帖子详情 · 平台视角' : '帖子详情'" append-to-body @closed="store.postId = null">
    <div v-loading="loading" style="min-height:200px">
      <template v-if="data">
        <div class="drawer-section">
          <div class="profile-head" style="margin-bottom:10px">
            <div class="av" style="width:44px;height:44px;font-size:22px;cursor:pointer" @click="openAccount(data.post.author.id, store.lens)">{{ data.post.author.avatar }}</div>
            <div>
              <h3 style="font-size:15px">{{ data.post.author.nickname }}
                <template v-if="store.lens"><el-tag v-if="data.post.author.is_bot" type="danger" size="small">真值：水军</el-tag><BotBadge :score="data.post.author.bot_score"/></template>
              </h3>
              <p>@{{ data.post.author.username }} · {{ data.post.created_at }}</p>
            </div>
          </div>
          <div class="post-text" style="cursor:default"><template v-for="(s, i) in segments(data.post.content)" :key="i"><span v-if="s.t !== 'text'" :class="s.t">{{ s.v }}</span><template v-else>{{ s.v }}</template></template></div>
          <div class="muted" style="margin-top:10px">♥ {{ data.post.likes }} · 💬 {{ data.post.comments }} · 🔁 {{ data.post.reposts }}</div>
        </div>
        <div class="drawer-section" v-if="store.lens">
          <h4><el-icon><View/></el-icon>审核结论</h4>
          <div class="row"><VerdictTag :verdict="data.post.mod_status" size="default"/><RiskLevel :score="data.post.risk_score"/></div>
          <div style="margin-top:10px"><LabelTags :labels="data.post.risk_labels"/></div>
          <a v-if="reportUrl" :href="reportUrl" target="_blank" style="display:inline-block;margin-top:10px">查看视频审核报告 →</a>
        </div>
        <div class="drawer-section">
          <h4><el-icon><ChatDotRound/></el-icon>评论 <span class="muted">{{ data.comments.length }} 条</span></h4>
          <div v-if="!data.comments.length" class="muted">还没有评论</div>
          <div v-for="c in data.comments" :key="c.id" class="comment">
            <div class="av" @click="openAccount(c.author.id, store.lens)">{{ c.author.avatar }}</div>
            <div class="c-body">
              <div class="c-head"><b>{{ c.author.nickname }}</b><span>{{ c.created_at }}</span>
                <template v-if="store.lens"><el-tag v-if="c.author.is_bot" type="danger" size="small">水军</el-tag><BotBadge :score="c.author.bot_score"/><VerdictTag :verdict="c.mod_status"/></template>
              </div>
              <div v-if="c.mod_status === 'block' && !store.lens && !shown[c.id]" class="post-blocked" style="margin-top:4px;padding:6px 10px">
                <el-icon><Hide/></el-icon>该评论因违反社区规定已被隐藏 <el-link type="primary" :underline="false" @click="shown[c.id] = true">仍要查看</el-link></div>
              <div v-else style="margin-top:2px">{{ c.content }}</div>
              <div v-if="store.lens && c.risk_labels" style="margin-top:4px"><LabelTags :labels="c.risk_labels" :max="4"/></div>
            </div>
          </div>
        </div>
      </template>
    </div>
  </el-drawer>`,
};
