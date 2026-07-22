<script setup>
import { RouterView } from 'vue-router'
import SideBar from '@/components/SideBar.vue'
import { useAppStore } from '@/stores/appStore'

const app = useAppStore()
</script>

<template>
  <div class="app-shell">
    <SideBar />
    <main class="main-content">
      <RouterView />
    </main>
    <Transition name="toast">
      <div v-if="app.toast" class="toast" :class="app.toast.type">
        <span class="toast-dot" />
        {{ app.toast.message }}
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.app-shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 224px minmax(0, 1fr);
}

.main-content {
  min-width: 0;
  padding: 24px 28px 36px;
}

.toast {
  position: fixed;
  z-index: 1000;
  top: 20px;
  right: 22px;
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 12px 16px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fff;
  box-shadow: var(--shadow);
  font-size: 12px;
}

.toast-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--primary);
}

.toast.error .toast-dot { background: var(--danger); }

.toast-enter-active,
.toast-leave-active { transition: transform 0.25s, opacity 0.25s; }
.toast-enter-from,
.toast-leave-to { opacity: 0; transform: translateY(-8px); }

@media (max-width: 1080px) {
  .app-shell { grid-template-columns: 76px minmax(0, 1fr); }
}

@media (max-width: 760px) {
  .app-shell { display: block; }
  .main-content { padding: 18px 16px 30px; }
}
</style>
