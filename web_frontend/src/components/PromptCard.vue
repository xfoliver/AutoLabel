<script setup>
import { ref } from 'vue'
import { useTaskStore } from '@/stores/taskStore'

const task = useTaskStore()
const focused = ref(false)
const mode = ref('single')
const suggestions = ref([
  '启用随机翻转',
  '固定角度旋转',
  '训练后自动测试',
  '对比 Baseline',
  'Epoch 设为 20',
])

function appendSuggestion(text) {
  const current = task.prompt.trim().replace(/[，。]+$/, '')
  task.prompt = current ? `${current}，${text}` : text
}

function rotateSuggestions() {
  suggestions.value = [...suggestions.value.slice(2), ...suggestions.value.slice(0, 2)]
}
</script>

<template>
  <section class="card prompt-card">
    <div class="composer-head">
      <div>
        <span class="eyebrow">AI COMMAND</span>
        <h2>描述你想完成的训练</h2>
        <p>系统会自动拆解意图、匹配数据集、配置参数并执行训练。</p>
      </div>
      <div class="mode-switch" aria-label="任务模式">
        <button :class="{ active: mode === 'single' }" @click="mode = 'single'">单任务</button>
        <button :class="{ active: mode === 'batch' }" @click="mode = 'batch'">批量编排</button>
      </div>
    </div>

    <div class="prompt-box" :class="{ focused }">
      <div class="prompt-icon">⌘</div>
      <textarea
        v-model="task.prompt"
        :placeholder="mode === 'single'
          ? '例如：训练一个 CLS 模型识别三种药材，启用随机固定角度旋转'
          : '例如：先训练 CLS，再训练 OCR，最后训练 SSEG 并启用随机翻转'"
        @focus="focused = true"
        @blur="focused = false"
      />
      <div class="prompt-meta">
        <div class="capabilities">
          <span>OCR</span><span>CLS</span><span>DET</span><span>DETR</span><span>ISEG</span><span>SSEG</span>
        </div>
        <span class="word-count">{{ task.prompt.length }} / 500</span>
      </div>
    </div>

    <div class="suggestion-row">
      <div class="suggestion-title">
        <span>智能补全</span>
        <button @click="rotateSuggestions">↻ 换一组</button>
      </div>
      <div class="bubbles">
        <button v-for="suggestion in suggestions" :key="suggestion" class="bubble" @click="appendSuggestion(suggestion)">
          ＋ {{ suggestion }}
        </button>
      </div>
    </div>

    <div class="action-row">
      <button class="run-button" :disabled="task.isRunning || !task.prompt.trim()" @click="task.startRun">
        <span class="play">{{ task.isRunning ? '···' : '▶' }}</span>
        <span>
          <strong>{{ task.isRunning ? '训练任务执行中' : '开始自动训练' }}</strong>
          <small>{{ mode === 'batch' ? '按顺序执行多个模型任务' : '生成配置并启动训练平台' }}</small>
        </span>
      </button>
      <button class="minor-action" @click="task.prompt = ''"><span>×</span> 清空</button>
      <button class="minor-action"><span>☆</span> 保存模板</button>
    </div>
  </section>
</template>

<style scoped>
.prompt-card { padding: 22px; overflow: hidden; }
.composer-head { display: flex; justify-content: space-between; gap: 24px; margin-bottom: 17px; }
.composer-head h2 { margin: 0; color: var(--text-strong); font-size: 20px; letter-spacing: -0.025em; }
.composer-head p { margin: 6px 0 0; color: var(--muted); font-size: 11px; }

.mode-switch { display: flex; height: 34px; padding: 3px; border: 1px solid var(--line); border-radius: 10px; background: var(--panel-2); }
.mode-switch button { padding: 0 11px; border: 0; border-radius: 7px; background: transparent; color: var(--muted); cursor: pointer; font-size: 10px; font-weight: 700; }
.mode-switch button.active { background: #fff; color: var(--primary); box-shadow: 0 3px 10px rgba(32,42,74,.08); }

.prompt-box { position: relative; padding: 15px 15px 10px 48px; border: 1px solid var(--line-strong); border-radius: 15px; background: #fbfcfe; transition: .2s ease; }
.prompt-box.focused { border-color: #9d94f5; box-shadow: 0 0 0 4px rgba(98,85,231,.09); background: #fff; }
.prompt-icon { position: absolute; top: 16px; left: 15px; width: 25px; height: 25px; display: grid; place-items: center; border-radius: 8px; background: var(--primary-soft); color: var(--primary); font-size: 11px; font-weight: 800; }
textarea { display: block; width: 100%; min-height: 103px; padding: 0; resize: vertical; border: 0; outline: 0; background: transparent; color: var(--text-strong); font-size: 14px; line-height: 1.7; }
textarea::placeholder { color: #a7adbc; }
.prompt-meta { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding-top: 9px; border-top: 1px solid #eceef3; }
.capabilities { display: flex; flex-wrap: wrap; gap: 5px; }
.capabilities span { padding: 3px 6px; border-radius: 5px; background: #f0f2f6; color: #778196; font-size: 8px; font-weight: 800; }
.word-count { color: var(--muted-2); font-size: 9px; }

.suggestion-row { display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center; gap: 14px; margin-top: 15px; }
.suggestion-title { display: flex; flex-direction: column; gap: 3px; }
.suggestion-title span { color: #303a50; font-size: 10px; font-weight: 800; white-space: nowrap; }
.suggestion-title button { padding: 0; border: 0; background: transparent; color: var(--primary); cursor: pointer; font-size: 9px; text-align: left; }
.bubbles { display: flex; flex-wrap: wrap; gap: 7px; }

.action-row { display: grid; grid-template-columns: minmax(230px, 1fr) auto auto; gap: 9px; margin-top: 17px; padding-top: 17px; border-top: 1px solid var(--line); }
.run-button { display: flex; min-height: 52px; align-items: center; justify-content: center; gap: 11px; padding: 8px 18px; border: 0; border-radius: 12px; background: linear-gradient(135deg, #6255e7, #7467ed); color: #fff; cursor: pointer; box-shadow: 0 12px 25px rgba(98,85,231,.24); text-align: left; transition: .2s ease; }
.run-button:hover { transform: translateY(-1px); box-shadow: 0 15px 28px rgba(98,85,231,.3); }
.run-button:disabled { opacity: .6; cursor: not-allowed; transform: none; }
.play { width: 27px; height: 27px; display: grid; place-items: center; border-radius: 9px; background: rgba(255,255,255,.16); font-size: 10px; }
.run-button strong, .run-button small { display: block; }
.run-button strong { font-size: 12px; }
.run-button small { margin-top: 3px; color: rgba(255,255,255,.68); font-size: 8px; }
.minor-action { min-width: 92px; padding: 0 13px; border: 1px solid var(--line); border-radius: 12px; background: #fff; color: #5d687d; cursor: pointer; font-size: 10px; font-weight: 700; }
.minor-action span { margin-right: 4px; color: var(--primary); font-size: 13px; }

@media (max-width: 700px) {
  .composer-head { flex-direction: column; }
  .mode-switch { align-self: flex-start; }
  .suggestion-row { grid-template-columns: 1fr; }
  .suggestion-title { flex-direction: row; justify-content: space-between; }
  .action-row { grid-template-columns: 1fr 1fr; }
  .run-button { grid-column: 1 / -1; }
}
</style>
