<script setup>
import { onMounted } from 'vue'
import { useTaskStore } from '@/stores/taskStore'

const task = useTaskStore()
onMounted(() => task.loadRecentTasks())
</script>

<template>
  <div class="card" style="margin-top: 22px">
    <div class="card-head">
      <h3>最近任务</h3>
      <button class="ghost-btn" style="padding: 7px 10px" @click="task.loadRecentTasks">刷新</button>
    </div>
    <div class="history">
      <div v-for="t in task.recentTasks.slice(0, 5)" :key="t.id" class="history-item">
        <div>
          <strong>{{ t.name }}</strong>
          <br />
          <small>{{ t.detail }}</small>
        </div>
        <span :class="['mini-status', t.status]" />
      </div>
      <div v-if="!task.recentTasks.length" class="empty-state">暂无历史任务</div>
    </div>
  </div>
</template>

<style scoped>
.history {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.history-item {
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  padding: 13px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
}

.history-item strong {
  font-size: 13px;
}

.history-item small {
  color: var(--muted);
}

.mini-status {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--muted);
  margin-top: 4px;
}

.mini-status.done {
  background: var(--success);
}

.mini-status.running {
  background: var(--primary);
}

.mini-status.error {
  background: var(--danger);
}
</style>
