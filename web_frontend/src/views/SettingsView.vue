<script setup>
import { reactive } from 'vue'
import TopBar from '@/components/TopBar.vue'

const settings = reactive({
  backendUrl: 'http://127.0.0.1:9887',
  frontendUrl: 'http://localhost:8842',
  platformExe: 'C:\\dlcv\\Lib\\site-packages\\dlcv_train\\DLCV AI Training Platform-win_x64.exe',
  cdpPort: 9222,
  adapter: 'playwright',
})

function save() {
  localStorage.setItem('autotrain_settings', JSON.stringify(settings))
  alert('设置已保存到本地')
}
</script>

<template>
  <TopBar />
  <div class="card">
    <div class="card-head">
      <h3>系统设置</h3>
      <span class="tag">本地配置</span>
    </div>
    <div class="form">
      <label>
        <span>推理后端地址</span>
        <input v-model="settings.backendUrl" type="text" />
      </label>
      <label>
        <span>前端地址</span>
        <input v-model="settings.frontendUrl" type="text" />
      </label>
      <label>
        <span>壳端 EXE 路径</span>
        <input v-model="settings.platformExe" type="text" />
      </label>
      <label>
        <span>CDP 调试端口</span>
        <input v-model.number="settings.cdpPort" type="number" />
      </label>
      <label>
        <span>默认 Adapter</span>
        <select v-model="settings.adapter">
          <option value="playwright">playwright</option>
          <option value="webview2_cdp">webview2_cdp</option>
          <option value="stub">stub</option>
        </select>
      </label>
      <button class="primary-btn" @click="save">保存设置</button>
    </div>
  </div>
</template>

<style scoped>
.form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

label {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

label span {
  font-size: 13px;
  color: var(--muted);
}

input,
select {
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fbfcff;
  outline: none;
}

input:focus,
select:focus {
  border-color: #b8afff;
}

button {
  align-self: flex-start;
}
</style>
