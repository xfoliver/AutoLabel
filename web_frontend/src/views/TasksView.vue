<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '@/api'
import TopBar from '@/components/TopBar.vue'

const tasks = ref([])
const filter = ref('all')
const loading = ref(true)

const filters = [
  { key: 'all', label: '全部' },
  { key: 'running', label: '运行中' },
  { key: 'done', label: '已完成' },
  { key: 'error', label: '失败' },
]

const filteredTasks = computed(() =>
  filter.value === 'all' ? tasks.value : tasks.value.filter((t) => t.status === filter.value)
)

onMounted(async () => {
  try {
    tasks.value = await api.tasks()
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <TopBar />
  <div class="card">
    <div class="card-head">
      <h3>任务列表</h3>
      <div class="filters">
        <button
          v-for="f in filters"
          :key="f.key"
          :class="['filter', { active: filter === f.key }]"
          @click="filter = f.key"
        >
          {{ f.label }}
        </button>
      </div>
    </div>
    <div v-if="loading" class="empty-state">加载中...</div>
    <div v-else-if="!filteredTasks.length" class="empty-state">暂无任务</div>
    <div v-else class="task-list">
      <div v-for="t in filteredTasks" :key="t.id" class="task-item">
        <div class="task-main">
          <strong>{{ t.name }}</strong>
          <small>{{ t.detail }}</small>
        </div>
        <span :class="['status', t.status]">
          {{ { done: '已完成', running: '运行中', error: '失败', waiting: '等待中' }[t.status] }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.filters {
  display: flex;
  gap: 8px;
}

.filter {
  border: 1px solid var(--line);
  background: white;
  border-radius: 999px;
  padding: 7px 10px;
  font-size: 12px;
  cursor: pointer;
}

.filter.active {
  background: #f0edff;
  color: var(--primary);
  border-color: #dcd4ff;
}

.task-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.task-item {
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  padding: 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.task-main strong {
  display: block;
  font-size: 14px;
}

.task-main small {
  color: var(--muted);
  font-size: 12px;
}
</style>
