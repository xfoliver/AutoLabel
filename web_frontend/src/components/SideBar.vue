<script setup>
import { useRoute } from 'vue-router'
import { useAppStore } from '@/stores/appStore'

const route = useRoute()
const app = useAppStore()
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark"><span>AT</span></div>
      <div class="brand-copy">
        <strong>AutoTrain</strong>
        <small>VISION OPS · 4.0</small>
      </div>
    </div>

    <div class="workspace-label">WORKSPACE</div>
    <nav class="nav">
      <router-link
        v-for="item in app.navItems"
        :key="item.path"
        :to="item.path"
        :class="['nav-btn', { active: route.path === item.path }]"
      >
        <span class="nav-icon">{{ item.icon }}</span>
        <span class="nav-name">{{ item.name }}</span>
        <span v-if="route.path === item.path" class="nav-indicator" />
      </router-link>
    </nav>

    <div class="sidebar-spacer" />

    <div class="system-card">
      <div class="system-head">
        <span class="pulse-dot" />
        <strong>系统在线</strong>
      </div>
      <div class="system-row">
        <span>Training API</span>
        <b>{{ app.health.backend ? '9887' : '—' }}</b>
      </div>
      <div class="system-row">
        <span>Web Console</span>
        <b>{{ app.health.frontend ? '8842' : '—' }}</b>
      </div>
      <div class="system-foot">本地工作站 · 延迟 12ms</div>
    </div>

    <button class="profile" type="button">
      <span class="avatar">A</span>
      <span class="profile-copy">
        <strong>Admin</strong>
        <small>本地管理员</small>
      </span>
      <span class="more">•••</span>
    </button>
  </aside>
</template>

<style scoped>
.sidebar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  height: 100vh;
  flex-direction: column;
  padding: 22px 15px 16px;
  overflow: hidden;
  color: #dce1ee;
  background:
    radial-gradient(circle at 40% 0%, rgba(111, 94, 243, 0.23), transparent 25%),
    linear-gradient(180deg, var(--nav) 0%, #0a0f1d 100%);
  border-right: 1px solid rgba(255, 255, 255, 0.05);
}

.brand {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 2px 6px 24px;
}

.brand-mark {
  width: 38px;
  height: 38px;
  flex: 0 0 auto;
  display: grid;
  place-items: center;
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 12px;
  background: linear-gradient(145deg, #7769f1, #5144c7);
  box-shadow: 0 10px 28px rgba(94, 76, 223, 0.35);
  color: #fff;
  font-size: 12px;
  font-weight: 850;
}

.brand-copy strong { display: block; color: #fff; font-size: 15px; }
.brand-copy small { display: block; margin-top: 3px; color: #727d98; font-size: 8px; letter-spacing: 0.12em; }

.workspace-label {
  padding: 0 11px 9px;
  color: #58627a;
  font-size: 8px;
  font-weight: 800;
  letter-spacing: 0.17em;
}

.nav { display: flex; flex-direction: column; gap: 5px; }

.nav-btn {
  position: relative;
  display: flex;
  min-height: 43px;
  align-items: center;
  gap: 11px;
  padding: 0 11px;
  border-radius: 11px;
  color: #8892aa;
  text-decoration: none;
  transition: 0.18s ease;
}

.nav-btn:hover { color: #e9ecf5; background: rgba(255, 255, 255, 0.045); }
.nav-btn.active { color: #fff; background: rgba(111, 94, 243, 0.18); }

.nav-icon {
  width: 25px;
  height: 25px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.055);
  font-size: 13px;
}

.nav-btn.active .nav-icon { background: #6e5fee; box-shadow: 0 6px 16px rgba(98, 85, 231, 0.3); }
.nav-name { font-size: 12px; font-weight: 650; }
.nav-indicator { position: absolute; right: 9px; width: 4px; height: 4px; border-radius: 50%; background: #8ff2dc; }

.sidebar-spacer { flex: 1; }

.system-card {
  margin: 16px 0 10px;
  padding: 13px;
  border: 1px solid rgba(255, 255, 255, 0.075);
  border-radius: 13px;
  background: rgba(255, 255, 255, 0.035);
}

.system-head { display: flex; align-items: center; gap: 7px; margin-bottom: 10px; color: #e9edf6; font-size: 10px; }
.pulse-dot { width: 7px; height: 7px; border-radius: 50%; background: #4bd1a5; box-shadow: 0 0 0 4px rgba(75, 209, 165, 0.12); }
.system-row { display: flex; justify-content: space-between; padding: 4px 0; color: #77829c; font-size: 9px; }
.system-row b { color: #c5cada; font-weight: 700; }
.system-foot { margin-top: 9px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,.06); color: #566179; font-size: 8px; }

.profile {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 9px;
  padding: 9px;
  border: 0;
  border-radius: 11px;
  background: transparent;
  color: inherit;
  cursor: pointer;
  text-align: left;
}
.profile:hover { background: rgba(255,255,255,.04); }
.avatar { width: 29px; height: 29px; display: grid; place-items: center; border-radius: 9px; background: #242c42; color: #bfc7da; font-size: 10px; font-weight: 800; }
.profile-copy { min-width: 0; flex: 1; }
.profile-copy strong, .profile-copy small { display: block; }
.profile-copy strong { color: #dfe4ef; font-size: 10px; }
.profile-copy small { margin-top: 2px; color: #636e87; font-size: 8px; }
.more { color: #59647d; font-size: 9px; }

@media (max-width: 1080px) {
  .sidebar { align-items: center; padding-inline: 10px; }
  .brand-copy, .workspace-label, .nav-name, .nav-indicator, .system-card, .profile-copy, .more { display: none; }
  .brand { padding-inline: 0; }
  .nav-btn { width: 45px; justify-content: center; padding: 0; }
  .nav-icon { width: 30px; height: 30px; }
  .profile { justify-content: center; padding: 8px 0; }
}

@media (max-width: 760px) {
  .sidebar {
    position: static;
    width: 100%;
    height: 68px;
    flex-direction: row;
    padding: 10px 14px;
  }
  .brand { padding: 0; }
  .brand-mark { width: 34px; height: 34px; }
  .nav { flex: 1; flex-direction: row; justify-content: flex-end; gap: 4px; }
  .nav-btn { width: 38px; min-height: 38px; }
  .nav-btn:nth-child(2), .nav-btn:nth-child(3), .nav-btn:nth-child(4) { display: none; }
  .sidebar-spacer, .profile { display: none; }
}
</style>
