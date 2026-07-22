<script setup>
import { onMounted, ref } from 'vue'
import { api } from '@/api'
import TopBar from '@/components/TopBar.vue'

const models = ref([])
const loading = ref(true)

onMounted(async () => {
  try {
    models.value = await api.models()
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <TopBar />
  <div class="card">
    <div class="card-head">
      <h3>模型中心</h3>
      <span class="tag">{{ models.length }} 个</span>
    </div>
    <div v-if="loading" class="empty-state">加载中...</div>
    <div v-else-if="!models.length" class="empty-state">暂无模型</div>
    <div v-else class="model-list">
      <div v-for="m in models" :key="m.path" class="model-item">
        <div class="model-main">
          <strong>{{ m.name }}</strong>
          <small>{{ m.path }}</small>
        </div>
        <span class="tag">{{ m.kind }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.model-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.model-item {
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  padding: 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.model-main strong {
  display: block;
  font-size: 14px;
}

.model-main small {
  color: var(--muted);
  font-size: 12px;
  word-break: break-all;
}
</style>
