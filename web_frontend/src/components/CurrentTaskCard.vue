<script setup>
import { computed } from 'vue'
import { useTaskStore } from '@/stores/taskStore'

const task = useTaskStore()
const state = computed(() => {
  if (task.isRunning) return { label: '执行中', class: 'running' }
  if (task.currentTask?.status === 'done') return { label: '已完成', class: 'done' }
  if (task.currentTask?.status === 'error') return { label: '失败', class: 'error' }
  return { label: '待命', class: 'waiting' }
})
</script>

<template>
  <section class="card task-card">
    <div class="card-head">
      <div><span class="eyebrow">CURRENT RUN</span><h3>本次任务</h3></div>
      <span :class="['status', state.class]">{{ state.label }}</span>
    </div>

    <div class="task-summary">
      <div class="task-glyph">{{ task.metrics.taskType === '-' ? 'AI' : task.metrics.taskType }}</div>
      <div>
        <strong>{{ task.currentTask ? `${task.metrics.taskType} 自动训练` : '等待训练指令' }}</strong>
        <p>{{ task.currentTask ? task.currentTask.prompt : '提交指令后，任务状态会在这里实时更新。' }}</p>
      </div>
    </div>

    <div class="task-progress">
      <div><span>总体进度</span><b>{{ task.progress.toFixed(0) }}%</b></div>
      <div class="bar"><i :style="{ width: `${task.progress}%` }" /></div>
    </div>

    <div class="task-stats">
      <div><span>当前阶段</span><strong>{{ task.currentStage >= 0 ? `${task.currentStage + 1} / 6` : '—' }}</strong></div>
      <div><span>已耗时</span><strong>{{ task.currentTask?.elapsed != null ? `${task.currentTask.elapsed}s` : '—' }}</strong></div>
      <div><span>产物目录</span><strong>{{ task.currentTask ? 'work/run_id' : '—' }}</strong></div>
    </div>

    <button v-if="task.isRunning" class="stop-button" @click="task.stopRun">停止任务</button>
    <button v-else class="result-button" :disabled="!task.currentTask">{{ task.currentTask ? '查看训练产物 →' : '尚无运行记录' }}</button>
  </section>
</template>

<style scoped>
.task-card { padding: 18px; }
.task-summary { display: grid; grid-template-columns: 39px minmax(0,1fr); gap: 11px; align-items: center; padding: 12px; border-radius: 12px; background: linear-gradient(135deg, #f7f5ff, #fafbfe); }
.task-glyph { width: 39px; height: 39px; display: grid; place-items: center; border-radius: 11px; background: linear-gradient(135deg, var(--primary), #7d70ee); color: #fff; font-size: 9px; font-weight: 850; }
.task-summary strong { display: block; color: var(--text-strong); font-size: 10px; }
.task-summary p { display: -webkit-box; overflow: hidden; margin: 4px 0 0; color: var(--muted); font-size: 8px; line-height: 1.45; -webkit-box-orient: vertical; -webkit-line-clamp: 2; }
.task-progress { margin-top: 13px; }
.task-progress > div:first-child { display: flex; justify-content: space-between; color: var(--muted); font-size: 9px; }
.task-progress b { color: var(--text); }
.bar { height: 6px; margin-top: 7px; overflow: hidden; border-radius: 999px; background: #eceef4; }
.bar i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--primary), #9d8df7); transition: width .4s ease; }
.task-stats { display: grid; grid-template-columns: repeat(3,1fr); gap: 6px; margin-top: 13px; }
.task-stats div { min-width: 0; padding: 9px 7px; border: 1px solid var(--line); border-radius: 9px; }
.task-stats span, .task-stats strong { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.task-stats span { color: var(--muted); font-size: 7px; }
.task-stats strong { margin-top: 4px; color: #3e485c; font-size: 8px; }
.stop-button, .result-button { width: 100%; margin-top: 12px; padding: 10px; border-radius: 10px; cursor: pointer; font-size: 9px; font-weight: 750; }
.stop-button { border: 1px solid #ffd8df; background: #fff5f6; color: var(--danger); }
.result-button { border: 1px solid var(--line); background: #fff; color: var(--primary); }
.result-button:disabled { color: var(--muted-2); cursor: default; }
</style>
