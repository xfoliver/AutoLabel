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
  return 'info'
}
</script>

<template>
  <section class="card execution-card">
    <div class="execution-head">
      <div><span class="eyebrow">LIVE EXECUTION</span><h3>执行控制台</h3><p>任务阶段与原始日志保持同步。</p></div>
      <div class="console-actions">
        <span :class="['status', task.isRunning ? 'running' : 'waiting']">{{ task.isRunning ? '实时连接' : '待命' }}</span>
        <button>导出日志</button>
      </div>
    </div>

    <div class="execution-layout">
      <nav class="stage-nav">
        <button
          v-for="(stage, index) in task.stageStates"
          :key="stage.key"
          :class="['stage-item', stage.status, { active: selectedStage === index }]"
          @click="selectedStage = index"
        >
          <span class="stage-node">{{ stage.status === 'done' ? '✓' : String(index + 1).padStart(2, '0') }}</span>
          <span class="stage-copy"><strong>{{ stage.title }}</strong><small>{{ stage.desc }}</small></span>
          <i />
        </button>
      </nav>

      <div class="console-panel">
        <div class="console-topbar">
          <div class="terminal-dots"><i /><i /><i /></div>
          <span>autotrain://live-output</span>
          <div class="filters">
            <button :class="{ active: logFilter === 'all' }" @click="logFilter = 'all'">全部</button>
            <button :class="{ active: logFilter === 'warn' }" @click="logFilter = 'warn'">警告</button>
            <button :class="{ active: logFilter === 'error' }" @click="logFilter = 'error'">错误</button>
          </div>
        </div>
        <div class="console-body">
          <div v-for="(log, index) in visibleLogs" :key="index" :class="['log-line', levelClass(log.line)]">
            <span class="line-number">{{ String(index + 1).padStart(2, '0') }}</span>
            <time>{{ log.time }}</time>
            <code>{{ log.line }}</code>
          </div>
          <div v-if="!visibleLogs.length" class="no-log">当前筛选条件下没有日志</div>
        </div>
        <div class="console-footer">
          <span><i /> stream connected</span>
          <span>{{ task.logs.length }} lines</span>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.execution-card { padding: 0; overflow: hidden; }
.execution-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; padding: 20px 21px 16px; }
.execution-head h3 { margin: 0; color: var(--text-strong); font-size: 15px; }
.execution-head p { margin: 5px 0 0; color: var(--muted); font-size: 10px; }
.console-actions { display: flex; align-items: center; gap: 7px; }
.console-actions button { padding: 7px 9px; border: 1px solid var(--line); border-radius: 8px; background: #fff; color: var(--muted); cursor: pointer; font-size: 8px; }
.execution-layout { display: grid; grid-template-columns: 190px minmax(0,1fr); min-height: 354px; border-top: 1px solid var(--line); }
.stage-nav { padding: 12px 9px; border-right: 1px solid var(--line); background: #fafbfc; }
.stage-item { position: relative; display: grid; grid-template-columns: 28px minmax(0,1fr); gap: 9px; width: 100%; padding: 9px 8px; border: 0; border-radius: 10px; background: transparent; cursor: pointer; text-align: left; }
.stage-item:hover, .stage-item.active { background: #fff; box-shadow: 0 5px 15px rgba(31,40,69,.06); }
.stage-node { position: relative; z-index: 1; width: 27px; height: 27px; display: grid; place-items: center; border: 1px solid #dfe3eb; border-radius: 9px; background: #fff; color: #8992a5; font-size: 8px; font-weight: 850; }
.stage-item.running .stage-node { border-color: var(--primary); background: var(--primary); color: #fff; box-shadow: 0 5px 13px rgba(98,85,231,.25); }
.stage-item.done .stage-node { border-color: #ccecdf; background: #e8f7f1; color: var(--success); }
.stage-copy { min-width: 0; padding-top: 1px; }
.stage-copy strong, .stage-copy small { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stage-copy strong { color: #414a5e; font-size: 9px; }
.stage-copy small { margin-top: 3px; color: #969daf; font-size: 7px; }
.console-panel { min-width: 0; display: flex; flex-direction: column; background: #101524; color: #cbd3e4; }
.console-topbar { display: flex; min-height: 39px; align-items: center; gap: 11px; padding: 0 13px; border-bottom: 1px solid rgba(255,255,255,.07); background: #151b2c; color: #68738e; font-family: Consolas, monospace; font-size: 8px; }
.terminal-dots { display: flex; gap: 4px; }
.terminal-dots i { width: 6px; height: 6px; border-radius: 50%; background: #49536a; }
.terminal-dots i:first-child { background: #df6d7c; }
.terminal-dots i:nth-child(2) { background: #d7a849; }
.terminal-dots i:last-child { background: #49b88f; }
.filters { display: flex; gap: 3px; margin-left: auto; }
.filters button { padding: 4px 6px; border: 0; border-radius: 5px; background: transparent; color: #67718a; cursor: pointer; font-size: 7px; }
.filters button.active { background: #252d43; color: #d8ddec; }
.console-body { flex: 1; max-height: 286px; overflow: auto; padding: 12px 0; font-family: "Cascadia Code", Consolas, monospace; font-size: 9px; line-height: 1.75; }
.log-line { display: grid; grid-template-columns: 25px 59px minmax(0,1fr); gap: 8px; padding: 2px 13px; }
.log-line:hover { background: rgba(255,255,255,.025); }
.line-number { color: #414b63; text-align: right; user-select: none; }
.log-line time { color: #5f6b84; }
.log-line code { color: #aeb9cd; white-space: pre-wrap; word-break: break-word; }
.log-line.error code { color: #f08d99; }
.log-line.warn code { color: #e3bd72; }
.log-line.success code { color: #75d5af; }
.no-log { display: grid; height: 180px; place-items: center; color: #4f5970; }
.console-footer { display: flex; justify-content: space-between; padding: 8px 13px; border-top: 1px solid rgba(255,255,255,.06); color: #56617a; font-family: Consolas, monospace; font-size: 7px; }
.console-footer span:first-child { display: flex; align-items: center; gap: 6px; }
.console-footer i { width: 5px; height: 5px; border-radius: 50%; background: #48bc91; }

@media (max-width: 700px) {
  .execution-layout { grid-template-columns: 1fr; }
  .stage-nav { display: grid; grid-template-columns: 1fr 1fr; border-right: 0; border-bottom: 1px solid var(--line); }
  .execution-head { flex-direction: column; }
}
</style>
