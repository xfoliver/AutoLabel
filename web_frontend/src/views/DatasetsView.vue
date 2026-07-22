<script setup>
import { onMounted, ref } from 'vue'
import { api } from '@/api'
import TopBar from '@/components/TopBar.vue'

const datasets = ref([])
const loading = ref(true)

onMounted(async () => {
  try {
    datasets.value = await api.datasets()
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <TopBar />
  <div class="card">
    <div class="card-head">
      <h3>数据集</h3>
      <span class="tag">{{ datasets.length }} 个</span>
    </div>
    <div v-if="loading" class="empty-state">加载中...</div>
    <div v-else-if="!datasets.length" class="empty-state">未找到数据集</div>
    <div v-else class="grid">
      <div v-for="ds in datasets" :key="ds.name" class="dataset-card">
        <div class="ds-head">
          <strong>{{ ds.name }}</strong>
          <span class="tag">{{ ds.type }}</span>
        </div>
        <p>{{ ds.path }}</p>
        <div class="ds-meta">
          <span>{{ ds.count }} 张图片</span>
          <span>{{ ds.updatedAt }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 14px;
}

.dataset-card {
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  padding: 16px;
  background: #fbfcff;
}

.ds-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.ds-head strong {
  font-size: 14px;
}

.dataset-card p {
  color: var(--muted);
  font-size: 12px;
  margin: 0 0 12px;
  word-break: break-all;
}

.ds-meta {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--muted);
}
</style>
