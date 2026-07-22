<script setup>
import { useTaskStore } from '@/stores/taskStore'
const task = useTaskStore()
</script>

<template>
  <div class="overview">
    <div class="metric"><span>识别任务</span><strong>{{ task.metrics.taskType }}</strong><i class="metric-icon violet">AI</i></div>
    <div class="metric"><span>候选数据集</span><strong>{{ task.metrics.dataset }}</strong><i class="metric-icon cyan">DB</i></div>
    <div class="metric"><span>预计耗时</span><strong>{{ task.metrics.estimate }}</strong><i class="metric-icon amber">⏱</i></div>
    <div class="metric"><span>执行模式</span><strong>{{ task.isRunning ? '自动执行中' : '全自动' }}</strong><i class="metric-icon green">↯</i></div>
  </div>
</template>

<style scoped>
.overview { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.metric { position: relative; min-height: 78px; padding-right: 45px; background: rgba(255,255,255,.82); box-shadow: 0 5px 18px rgba(31,40,69,.035); }
.metric-icon { position: absolute; top: 13px; right: 13px; width: 27px; height: 27px; display: grid; place-items: center; border-radius: 8px; font-size: 8px; font-style: normal; font-weight: 850; }
.violet { color: var(--primary); background: var(--primary-soft); }
.cyan { color: #188e88; background: var(--cyan-soft); }
.amber { color: #b96f0b; background: #fff3df; }
.green { color: var(--success); background: #e8f7f1; }
@media (max-width: 720px) { .overview { grid-template-columns: 1fr 1fr; } }
</style>
