import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from '@/views/DashboardView.vue'
import TasksView from '@/views/TasksView.vue'
import DatasetsView from '@/views/DatasetsView.vue'
import ModelsView from '@/views/ModelsView.vue'
import SettingsView from '@/views/SettingsView.vue'

const routes = [
  { path: '/', name: 'Dashboard', component: DashboardView },
  { path: '/tasks', name: 'Tasks', component: TasksView },
  { path: '/datasets', name: 'Datasets', component: DatasetsView },
  { path: '/models', name: 'Models', component: ModelsView },
  { path: '/settings', name: 'Settings', component: SettingsView },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
