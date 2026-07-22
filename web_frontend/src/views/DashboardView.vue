<script setup>
import { onMounted } from 'vue'
import { useAppStore } from '@/stores/appStore'
import { useTaskStore } from '@/stores/taskStore'
import { api } from '@/api'

import TopBar from '@/components/TopBar.vue'
import PromptCard from '@/components/PromptCard.vue'
import MetricOverview from '@/components/MetricOverview.vue'
import PipelineStages from '@/components/PipelineStages.vue'
import QuickTemplates from '@/components/QuickTemplates.vue'
import CurrentTaskCard from '@/components/CurrentTaskCard.vue'
import AssistantPanel from '@/components/AssistantPanel.vue'

const app = useAppStore()
const task = useTaskStore()

onMounted(async () => {
  try {
    const health = await api.health()
    app.health = { backend: !!health.backend, frontend: !!health.frontend }
  } catch {
    app.health = { backend: false, frontend: false }
  }
  task.loadRecentTasks()
})
</script>

<template>
  <TopBar />
  <div class="dashboard-grid">
    <section class="primary-column">
      <PromptCard />
      <MetricOverview />
      <PipelineStages />
    </section>
    <aside class="right-rail">
      <QuickTemplates />
      <CurrentTaskCard />
      <AssistantPanel />
    </aside>
  </div>
</template>

<style scoped>
.dashboard-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 330px;
  gap: 18px;
  align-items: start;
}
.primary-column, .right-rail { min-width: 0; display: flex; flex-direction: column; gap: 16px; }
.right-rail { position: sticky; top: 20px; }

@media (min-width: 1480px) {
  .dashboard-grid { grid-template-columns: minmax(700px, 1fr) 360px; gap: 22px; }
}

@media (max-width: 1180px) {
  .dashboard-grid { grid-template-columns: minmax(0, 1fr) 300px; }
}

@media (max-width: 920px) {
  .dashboard-grid { grid-template-columns: 1fr; }
  .right-rail { position: static; display: grid; grid-template-columns: 1fr 1fr; }
  .right-rail > :last-child { grid-column: 1 / -1; }
}

@media (max-width: 620px) {
  .right-rail { display: flex; }
}
</style>
