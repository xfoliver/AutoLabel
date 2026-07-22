<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const titles = {
  '/': { eyebrow: 'TRAINING CONTROL', title: '自动化训练工作台', subtitle: '从一句自然语言指令到可复现模型，全流程可见、可控。' },
  '/tasks': { eyebrow: 'RUN HISTORY', title: '训练任务', subtitle: '查看历史任务、失败证据与训练产物。' },
  '/datasets': { eyebrow: 'DATA ASSETS', title: '数据集', subtitle: '管理可用于自动匹配的视觉数据资产。' },
  '/models': { eyebrow: 'MODEL REGISTRY', title: '模型中心', subtitle: '管理 Baseline、DVA 项目与 DVP 模型。' },
  '/settings': { eyebrow: 'SYSTEM', title: '系统设置', subtitle: '配置平台连接和自动化执行参数。' },
}
const meta = computed(() => titles[route.path] || titles['/'])
</script>

<template>
  <header class="topbar">
    <div class="title">
      <span class="eyebrow">{{ meta.eyebrow }}</span>
      <h1>{{ meta.title }}</h1>
      <p>{{ meta.subtitle }}</p>
    </div>
    <div class="top-actions">
      <button class="ghost-btn doc-button"><span>?</span> 使用指南</button>
      <button class="icon-btn notification" aria-label="通知">⌁<i /></button>
      <button class="primary-btn">＋ 新建训练</button>
    </div>
  </header>
</template>

<style scoped>
.topbar {
  display: flex;
  min-height: 74px;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 20px;
}

.title h1 { margin: 0; color: var(--text-strong); font-size: clamp(23px, 2vw, 30px); letter-spacing: -0.035em; }
.title p { margin: 7px 0 0; color: var(--muted); font-size: 12px; }
.top-actions { display: flex; align-items: center; gap: 8px; padding-top: 7px; }
.doc-button span { display: inline-grid; width: 17px; height: 17px; place-items: center; margin-right: 5px; border: 1px solid #cdd2de; border-radius: 50%; font-size: 9px; }
.notification { position: relative; width: 38px; padding-inline: 0; }
.notification i { position: absolute; top: 8px; right: 8px; width: 5px; height: 5px; border: 1px solid #fff; border-radius: 50%; background: var(--danger); }

@media (max-width: 680px) {
  .topbar { min-height: 0; align-items: flex-start; flex-direction: column; }
  .title p { line-height: 1.5; }
  .top-actions { width: 100%; padding-top: 0; }
  .top-actions .primary-btn { margin-left: auto; }
  .doc-button { display: none; }
}
</style>
