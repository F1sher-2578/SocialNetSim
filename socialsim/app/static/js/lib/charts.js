// ECharts 主题、通用图表组件和几个跨页面复用的 option 构造函数
import { store, FEATURE_GROUPS } from './api.js';

const { ref, watch, onMounted, onBeforeUnmount, onActivated } = Vue;

export const FONT = '"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",-apple-system,sans-serif';
export const C = {
  primary: '#165dff', pass: '#00b42a', review: '#ff7d00', block: '#f53f3f', purple: '#722ed1', cyan: '#0fc6c2',
  gold: '#f7ba1e', magenta: '#eb0aa4', text: '#1d2129', text2: '#4e5969', sub: '#86909c', line: '#e5e6eb', grey: '#c9cdd4',
};
export const PALETTE = [C.primary, C.cyan, C.gold, C.purple, '#f77234', '#3491fa', '#9fdb1d', C.magenta, C.sub];

echarts.registerTheme('socialsim', {
  color: PALETTE,
  backgroundColor: 'transparent',
  textStyle: { fontFamily: FONT, color: C.text2 },
  title: { textStyle: { color: C.text, fontSize: 13, fontWeight: 600 }, subtextStyle: { color: C.sub } },
  legend: { textStyle: { color: C.text2, fontSize: 12 }, icon: 'roundRect', itemWidth: 12, itemHeight: 8, itemGap: 14 },
  tooltip: {
    backgroundColor: 'rgba(255,255,255,0.97)', borderColor: C.line, borderWidth: 1, padding: [8, 12],
    textStyle: { color: C.text, fontSize: 12 }, extraCssText: 'box-shadow:0 6px 20px rgba(0,0,0,.08);border-radius:8px;',
  },
  categoryAxis: {
    axisLine: { lineStyle: { color: C.line } }, axisTick: { show: false },
    axisLabel: { color: C.sub, fontSize: 11 }, splitLine: { show: false },
  },
  valueAxis: {
    axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: C.sub, fontSize: 11 },
    splitLine: { lineStyle: { color: '#f0f1f3' } }, nameTextStyle: { color: C.sub, fontSize: 11 },
  },
  line: { symbolSize: 6, lineStyle: { width: 2.2 } },
  bar: { barMaxWidth: 26, itemStyle: { borderRadius: [3, 3, 0, 0] } },
});

/**
 * <SnChart :option="..." :height="300" @click="..."/>
 * 实例保存在闭包变量里而不是 ref/reactive（深层代理会拖慢甚至弄坏 ECharts）；
 * 用 ResizeObserver 跟随容器尺寸，keep-alive 切回时补一次 resize，卸载时 dispose。
 */
export const SnChart = {
  name: 'SnChart',
  props: { option: Object, height: { type: [Number, String], default: 300 } },
  emits: ['click', 'ready'],
  template: `<div ref="el" class="sn-chart" :style="{ height: typeof height === 'number' ? height + 'px' : height }"></div>`,
  setup(props, { emit }) {
    const el = ref(null);
    let chart = null, ro = null;
    const render = () => { if (chart && props.option) chart.setOption(props.option, { notMerge: true }); };
    onMounted(() => {
      chart = echarts.init(el.value, 'socialsim');
      chart.on('click', p => emit('click', p));
      render();
      emit('ready', chart);
      ro = new ResizeObserver(() => chart && chart.resize());
      ro.observe(el.value);
    });
    watch(() => props.option, render);
    onActivated(() => chart && chart.resize());
    onBeforeUnmount(() => { ro && ro.disconnect(); chart && chart.dispose(); chart = null; });
    return { el };
  },
};

export const axisPct = { formatter: v => `${Math.round(v * 100)}%` };

/** 风险仪表：按复核 / 拦截阈值分三段着色 */
export function gaugeOption(score, { th = store.meta.thresholds, title = '风险分', colors } = {}) {
  const segs = colors || [[th.review, C.pass], [th.block, C.review], [1, C.block]];
  return {
    series: [{
      type: 'gauge', startAngle: 205, endAngle: -25, min: 0, max: 1, radius: '92%', center: ['50%', '56%'],
      axisLine: { lineStyle: { width: 12, color: segs } },
      pointer: { length: '52%', width: 5, offsetCenter: [0, 0], itemStyle: { color: C.text } },
      anchor: { show: true, size: 10, itemStyle: { color: C.text, borderColor: '#fff', borderWidth: 2 } },
      axisTick: { show: false }, splitLine: { length: 12, distance: -12, lineStyle: { color: '#fff', width: 2 } },
      axisLabel: { distance: 16, color: C.sub, fontSize: 10, formatter: v => (v === 0 || v === 1 || Math.abs(v - th.review) < 0.051 || Math.abs(v - th.block) < 0.051) ? v.toFixed(1) : '' },
      title: { offsetCenter: [0, '62%'], color: C.sub, fontSize: 12 },
      detail: { offsetCenter: [0, '36%'], fontSize: 22, fontWeight: 650, color: C.text, formatter: v => (+v).toFixed(3) },
      data: [{ value: +(score || 0).toFixed(4), name: title }],
    }],
  };
}

/** 五组特征雷达：数值是该账号朝"水军方向"偏离的平均分位，0.5 = 与全体中位数持平 */
export function radarOption(profile, color = C.primary) {
  const keys = Object.keys(FEATURE_GROUPS);
  return {
    tooltip: { trigger: 'item' },
    radar: {
      indicator: keys.map(k => ({ name: FEATURE_GROUPS[k], max: 1 })), radius: '66%', center: ['50%', '54%'],
      splitNumber: 4, axisName: { color: C.text2, fontSize: 12 },
      splitArea: { areaStyle: { color: ['#fafbfc', '#fff'] } }, splitLine: { lineStyle: { color: C.line } },
      axisLine: { lineStyle: { color: C.line } },
    },
    series: [{
      type: 'radar', symbolSize: 5,
      data: [
        { value: keys.map(() => 0.5), name: '全体中位数', lineStyle: { type: 'dashed', color: C.grey, width: 1 },
          itemStyle: { color: C.grey }, areaStyle: { opacity: 0 }, symbol: 'none' },
        { value: keys.map(k => +(profile?.[k] ?? 0.5).toFixed(3)), name: '该账号', lineStyle: { color },
          itemStyle: { color }, areaStyle: { color, opacity: 0.18 } },
      ],
    }],
  };
}
