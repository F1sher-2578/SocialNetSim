// 应用外壳：侧边导航（按"用户端 / 红队 / 蓝队 / 评估"分组讲解）、顶栏、hash 路由与演示导览
import { store, api } from './lib/api.js';
import { SnChart } from './lib/charts.js';
import { registerCommon } from './components/common.js';
import { AccountDrawer, PostDrawer } from './components/drawers.js';
import FeedPage from './pages/feed.js';
import BotsPage from './pages/bots.js';
import DetectPage from './pages/detect.js';
import ModerationPage from './pages/moderation.js';
import AssistantPage from './pages/assistant.js';
import LabPage from './pages/lab.js';

const { createApp, ref, computed, onMounted, markRaw } = Vue;

const ROUTES = [
  { path: 'feed', title: '信息流', group: '用户端', icon: 'ChatLineSquare', comp: FeedPage },
  { path: 'bots', title: '水军控制台', group: '红队 · 攻击方', icon: 'Aim', comp: BotsPage },
  { path: 'detect', title: '水军检测', group: '蓝队 · 防御方', icon: 'Search', comp: DetectPage },
  { path: 'moderation', title: '内容审核', group: '蓝队 · 防御方', icon: 'VideoCamera', comp: ModerationPage },
  { path: 'assistant', title: '安全助手', group: '蓝队 · 防御方', icon: 'Lock', comp: AssistantPage },
  { path: 'lab', title: '攻防实验', group: '评估', icon: 'DataAnalysis', comp: LabPage },
].map(r => ({ ...r, comp: markRaw(r.comp) }));

const Shell = {
  components: { AccountDrawer, PostDrawer },
  setup() {
    const route = ref('feed');
    const readHash = () => {
      const p = location.hash.replace(/^#\/?/, '');
      route.value = ROUTES.some(r => r.path === p) ? p : 'feed';
    };
    readHash();
    window.addEventListener('hashchange', readHash);
    const current = computed(() => ROUTES.find(r => r.path === route.value));
    const groups = computed(() => {
      const out = [];
      for (const r of ROUTES) {
        let g = out.find(x => x.name === r.group);
        if (!g) out.push(g = { name: r.group, routes: [] });
        g.routes.push(r);
      }
      return out;
    });
    const go = p => { location.hash = '#/' + p; };
    const collapsed = ref(window.innerWidth < 1100);
    const tourOpen = ref(false);
    onMounted(async () => {
      try { Object.assign(store.meta, await api('/api/meta')); } catch (e) { ElementPlus.ElMessage.error('无法连接后端：' + e.message); }
    });
    return { store, route, current, groups, go, collapsed, tourOpen };
  },
  template: `
  <el-container class="sn-layout">
    <el-aside :width="collapsed ? '64px' : '228px'" class="sn-aside">
      <div class="sn-brand">
        <div class="sn-logo"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8"><circle cx="7" cy="8" r="2.6" fill="#fff"/><circle cx="17" cy="6.5" r="2" fill="#fff" opacity=".85"/><circle cx="15.5" cy="17" r="3" fill="#fff"/><path d="M7 8l10-1.5M7 8l8.5 9M17 6.5L15.5 17" opacity=".75"/></svg></div>
        <div v-show="!collapsed"><b>SocialNetSim</b><small>社交网络拟真沙箱</small></div>
      </div>
      <el-scrollbar>
        <el-menu class="sn-menu" :default-active="route" :collapse="collapsed" :collapse-transition="false" @select="go">
          <el-menu-item-group v-for="g in groups" :key="g.name" :id="'menu-' + g.routes[0].path">
            <template #title>{{ g.name }}</template>
            <el-menu-item v-for="r in g.routes" :key="r.path" :index="r.path">
              <el-icon><component :is="r.icon"/></el-icon><template #title>{{ r.title }}</template>
            </el-menu-item>
          </el-menu-item-group>
        </el-menu>
      </el-scrollbar>
      <div class="sn-aside-foot" v-show="!collapsed"><span class="dot"></span>本地沙箱 · 不连接任何真实平台</div>
    </el-aside>
    <el-container style="min-width:0">
      <el-header class="sn-header">
        <el-button text circle @click="collapsed = !collapsed"><el-icon size="18"><component :is="collapsed ? 'Expand' : 'Fold'"/></el-icon></el-button>
        <div class="sn-crumb">{{ current.group }} / <b>{{ current.title }}</b></div>
        <div class="spacer"></div>
        <span class="sn-clock hide-sm"><el-icon><Clock/></el-icon>仿真时间 {{ store.meta.sim_now || '--' }}</span>
        <el-tooltip placement="bottom" :show-after="300"
          content="把平台视角叠加到用户端界面上：真值标签、检测分、审核判定、被水军推高的热门帖">
          <span id="lens-switch" class="sn-lens" :class="{ on: store.lens }"><el-icon><View/></el-icon>审核透视
            <el-switch v-model="store.lens" size="small" style="--el-switch-on-color:#ff7d00"/></span>
        </el-tooltip>
        <el-button id="tour-btn" @click="tourOpen = true"><el-icon><Guide/></el-icon>&nbsp;演示导览</el-button>
      </el-header>
      <el-main class="sn-main">
        <keep-alive><component :is="current.comp" :key="current.path"/></keep-alive>
      </el-main>
    </el-container>
    <AccountDrawer/>
    <PostDrawer/>
    <el-tour v-model="tourOpen" :z-index="3000">
      <el-tour-step title="SocialNetSim 是什么"
        description="一个完全在本地运行的仿真社交平台：一边模拟水军和违规内容（红队），一边用检测、审核和护栏去防御（蓝队），最后用公开基准评估两边的真实能力边界。所有操作只作用于本机 SQLite 沙箱。"/>
      <el-tour-step target="#menu-feed" title="用户端" placement="right"
        description="信息流就是普通用户看到的平台：热门排序、热搜榜、发帖即审核。被拦截的内容会折叠，待复核的内容会显示「可见性受限」。"/>
      <el-tour-step target="#lens-switch" title="审核透视" placement="bottom"
        description="一键切换「用户看到什么」和「平台看到什么」：打开后每条帖子都会叠加真值标签、0–5 分的水军检测分、审核判定，以及被水军撑起来的热门帖。"/>
      <el-tour-step target="#menu-bots" title="红队 · 攻击方" placement="right"
        description="水军控制台可以发起 5 种策略 × 3 级规避强度的水军任务，生成带真值标签的协同行为，用来训练和考验检测器。"/>
      <el-tour-step target="#menu-detect" title="蓝队 · 防御方" placement="right"
        description="水军检测（关系网络、消融、泛化）、多模态内容审核（文本 / 图片 / 视频）、带四层护栏的安全助手。"/>
      <el-tour-step target="#menu-lab" title="评估" placement="right"
        description="攻防实验汇总全部离线实验。标着「负面结果」的几项最值得讲：它们说明了当前技术的真实能力边界。"/>
    </el-tour>
  </el-container>`,
};

const app = createApp(Shell);
for (const [name, comp] of Object.entries(ElementPlusIconsVue)) app.component(name, comp);
app.component('SnChart', SnChart);
registerCommon(app);
app.use(ElementPlus, { locale: ElementPlusLocaleZhCn });
app.config.errorHandler = (err, _vm, info) => { console.error(err, info); ElementPlus.ElMessage.error(`页面出错：${err.message}`); };
app.mount('#app');
