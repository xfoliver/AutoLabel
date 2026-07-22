import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useAppStore = defineStore('app', () => {
  const navItems = ref([
    { name: '工作台', path: '/', icon: '⌂' },
    { name: '训练任务', path: '/tasks', icon: '◫' },
    { name: '数据集', path: '/datasets', icon: '⌁' },
    { name: '模型中心', path: '/models', icon: '◉' },
    { name: '系统设置', path: '/settings', icon: '⚙' },
  ])

  const health = ref({ backend: false, frontend: false })
  const toast = ref(null)

  function showToast(message, type = 'info') {
    toast.value = { message, type }
    setTimeout(() => {
      toast.value = null
    }, 3000)
  }

  return { navItems, health, toast, showToast }
})
