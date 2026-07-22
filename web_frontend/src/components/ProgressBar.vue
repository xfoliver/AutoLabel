<script setup>
import { useTaskStore } from '@/stores/taskStore'

const task = useTaskStore()
</script>

<template>
  <div class="progress-wrap">
    <div class="progress-label">
      <span>模型训练进度</span>
      <strong>{{ task.progress.toFixed(1) }}%</strong>
    </div>
    <div class="bar">
      <i :style="{ width: `${task.progress}%` }" />
    </div>
  </div>
</template>

<style scoped>
.progress-wrap {
  background: var(--panel-2);
  border: 1px solid var(--line);
  padding: 16px;
  border-radius: 16px;
  margin-bottom: 16px;
}

.progress-label {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  margin-bottom: 9px;
}

.bar {
  height: 10px;
  background: #e8eaf2;
  border-radius: 999px;
  overflow: hidden;
}

.bar > i {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, var(--primary), #9c78ff);
  border-radius: inherit;
  transition: width 0.4s ease;
}
</style>
