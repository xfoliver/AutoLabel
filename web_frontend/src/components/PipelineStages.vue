<script setup>
import { computed, ref } from 'vue'
import { useTaskStore } from '@/stores/taskStore'

const task = useTaskStore()
const selectedStage = ref(0)
const logFilter = ref('all')

const visibleLogs = computed(() => {
  const logs = task.logs.length
    ? task.logs
    : [
        { time: '09:41:02', line: '[SYSTEM] AutoTrain 4.0 工作区已就绪', stage: 'global' },
        { time: '09:41:03', line: '[INFO] 等待新的自然语言训练指令…', stage: 'global' },
      ]
  if (logFilter.value === 'error') return logs.filter((log) => /ERROR|失败/.test(log.line))
  if (logFilter.value === 'warn') return logs.filter((log) => /WARN|警告/.test(log.line))
  return logs
})

function levelClass(line) {
  if (/ERROR|失败/.test(line)) return 'error'
  if (/WARN|警告/.test(line)) return 'warn'
  if (/DONE|完成|成功/.test(line)) return 'success'
  if (/\[新运行\]|\[开始\]|={5,}/.test(line)) return 'meta'
  return 'info'
}
</script>

<template>
  <section class="card execution-card">
    <div class="execution-head">
      <div>
        <span class="eyebrow">LIVE EXECUTION</span>
        <h3>执行控制台</h3>
        <p>任务阶段与原始日志保持同步。</p>
      </div>
      <div class="console-actions">
        <span :class="['status', task.isRunning ? 'running' : 'waiting']">
          {{ task.isRunning ? '实时连接' : '待命' }}
        </span>
        <button type="button" class="export-btn">导出日志</button>
      </div>
    </div>

    <div class="execution-layout">
      <nav class="stage-nav" aria-label="Execution flow">
        <div class="stage-nav-label">EXECUTION FLOW</div>
        <button
          v-for="(stage, index) in task.stageStates"
          :key="stage.key"
          type="button"
          :class="['stage-item', stage.status, { active: selectedStage === index }]"
          @click="selectedStage = index"
        >
          <span class="stage-node">
            {{ stage.status === 'done' ? '✓' : stage.status === 'error' ? '×' : String(index + 1).padStart(2, '0') }}
          </span>
          <span class="stage-copy">
            <strong>{{ stage.title }}</strong>
            <small>{{ stage.desc }}</small>
          </span>
        </button>
      </nav>

      <div class="console-panel">
        <div class="console-topbar">
          <div class="console-title-block">
            <span class="console-eyebrow">LIVE OUTPUT</span>
            <strong>运行日志</strong>
          </div>
          <div class="filters" role="tablist" aria-label="日志筛选">
            <button
              type="button"
              :class="{ active: logFilter === 'all' }"
              @click="logFilter = 'all'"
            >全部</button>
            <button
              type="button"
              :class="{ active: logFilter === 'warn' }"
              @click="logFilter = 'warn'"
            >警告</button>
            <button
              type="button"
              :class="{ active: logFilter === 'error' }"
              @click="logFilter = 'error'"
            >错误</button>
          </div>
          <span :class="['live-pill', task.isRunning ? 'on' : 'off']">
            {{ task.isRunning ? '● LIVE' : '● 待命' }}
          </span>
        </div>

        <div class="console-body">
          <div
            v-for="(log, index) in visibleLogs"
            :key="index"
            :class="['log-line', levelClass(log.line)]"
          >
            <span class="line-number">{{ String(index + 1).padStart(2, '0') }}</span>
            <time>{{ log.time }}</time>
            <code>{{ log.line }}</code>
          </div>
          <div v-if="!visibleLogs.length" class="no-log">当前筛选条件下没有日志</div>
        </div>

        <div class="console-footer">
          <span class="stream">
            <i :class="{ on: task.isRunning }" />
            {{ task.isRunning ? 'stream connected' : 'stream idle' }}
          </span>
          <span>{{ task.logs.length }} lines</span>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.execution-card {
  padding: 0;
  overflow: hidden;
}

.execution-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 20px 21px 16px;
}

.execution-head h3 {
  margin: 0;
  color: var(--text-strong);
  font-size: 15px;
}

.execution-head p {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 12px;
}

.console-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.export-btn {
  padding: 7px 11px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #fff;
  color: #535e73;
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
}

.export-btn:hover {
  background: var(--panel-2);
}

.execution-layout {
  display: grid;
  grid-template-columns: 190px minmax(0, 1fr);
  min-height: 354px;
  border-top: 1px solid var(--line);
}

/* 左侧 flow list — 保留并微调 */
.stage-nav {
  padding: 12px 9px;
  border-right: 1px solid var(--line);
  background: #f7f8fc;
}

.stage-nav-label {
  padding: 4px 8px 10px;
  color: var(--primary);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.12em;
}

.stage-item {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr);
  gap: 9px;
  width: 100%;
  margin-bottom: 4px;
  padding: 9px 8px;
  border: 1px solid transparent;
  border-radius: 10px;
  background: transparent;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}

.stage-item:hover {
  background: #fff;
  border-color: var(--line);
}

.stage-item.active {
  background: #fff;
  border-color: #e0dbff;
  box-shadow: 0 5px 15px rgba(31, 40, 69, 0.06);
}

.stage-node {
  width: 27px;
  height: 27px;
  display: grid;
  place-items: center;
  border: 1px solid #dfe3eb;
  border-radius: 9px;
  background: #fff;
  color: #8992a5;
  font-size: 10px;
  font-weight: 850;
}

.stage-item.running .stage-node {
  border-color: var(--primary);
  background: var(--primary);
  color: #fff;
  box-shadow: 0 5px 13px rgba(98, 85, 231, 0.25);
}

.stage-item.done .stage-node {
  border-color: #ccecdf;
  background: #e8f7f1;
  color: var(--success);
}

.stage-item.error .stage-node {
  border-color: #f5c2c9;
  background: #fdecef;
  color: var(--danger);
}

.stage-copy {
  min-width: 0;
  padding-top: 1px;
}

.stage-copy strong,
.stage-copy small {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.stage-copy strong {
  color: #414a5e;
  font-size: 12px;
}

.stage-item.active .stage-copy strong {
  color: #6655d9;
}

.stage-copy small {
  margin-top: 3px;
  color: #969daf;
  font-size: 11px;
}

/* 右侧日志 — 浅色结构化面板，贴合整体 UI */
.console-panel {
  min-width: 0;
  display: flex;
  flex-direction: column;
  background: #fff;
  color: var(--text);
}

.console-topbar {
  display: flex;
  min-height: 52px;
  align-items: center;
  gap: 12px;
  padding: 0 14px;
  border-bottom: 1px solid var(--line);
  background: #fafbfe;
}

.console-title-block {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.console-eyebrow {
  color: var(--primary);
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.12em;
}

.console-title-block strong {
  color: var(--text-strong);
  font-size: 13px;
  font-weight: 800;
}

.filters {
  display: flex;
  gap: 2px;
  margin-left: auto;
  padding: 3px;
  border-radius: 9px;
  background: #eef0f6;
}

.filters button {
  padding: 5px 10px;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: #6b7280;
  cursor: pointer;
  font-size: 11px;
  font-weight: 700;
}

.filters button:hover {
  color: #414b5f;
  background: rgba(255, 255, 255, 0.65);
}

.filters button.active {
  background: #fff;
  color: #6655d9;
  box-shadow: 0 1px 3px rgba(31, 40, 69, 0.06);
}

.live-pill {
  flex-shrink: 0;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 800;
}

.live-pill.on {
  color: var(--success);
  background: #e9f8f2;
  border: 1px solid #ccecdf;
}

.live-pill.off {
  color: #7b8499;
  background: #eef0f6;
  border: 1px solid var(--line);
}

.console-body {
  flex: 1;
  max-height: 286px;
  overflow: auto;
  padding: 12px 0;
  font-family: "Cascadia Code", Consolas, "SFMono-Regular", monospace;
  font-size: 12px;
  line-height: 1.75;
  background: #fff;
}

.log-line {
  display: grid;
  grid-template-columns: 28px 64px minmax(0, 1fr);
  gap: 8px;
  padding: 2px 14px;
}

.log-line:hover {
  background: #f7f8fc;
}

.line-number {
  color: #b0b7c8;
  text-align: right;
  user-select: none;
}

.log-line time {
  color: #8b93a7;
}

.log-line code {
  color: #4a5568;
  white-space: pre-wrap;
  word-break: break-word;
}

.log-line.error code {
  color: var(--danger);
  font-weight: 650;
}

.log-line.warn code {
  color: #c47d12;
  font-weight: 650;
}

.log-line.success code {
  color: var(--success);
  font-weight: 650;
}

.log-line.meta code {
  color: #6655d9;
}

.no-log {
  display: grid;
  height: 180px;
  place-items: center;
  color: #9aa3b5;
  font-size: 13px;
}

.console-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 14px;
  border-top: 1px solid var(--line);
  background: #fafbfe;
  color: #8b93a7;
  font-family: Consolas, monospace;
  font-size: 11px;
  font-weight: 600;
}

.stream {
  display: flex;
  align-items: center;
  gap: 7px;
}

.stream i {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #c5cad6;
}

.stream i.on {
  background: var(--success);
  box-shadow: 0 0 0 3px rgba(21, 150, 109, 0.15);
}

@media (max-width: 700px) {
  .execution-layout {
    grid-template-columns: 1fr;
  }

  .stage-nav {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }

  .stage-nav-label {
    grid-column: 1 / -1;
  }

  .execution-head {
    flex-direction: column;
  }

  .console-topbar {
    flex-wrap: wrap;
    padding: 10px 14px;
    min-height: auto;
  }

  .filters {
    margin-left: 0;
  }
}
</style>
