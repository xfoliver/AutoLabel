import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import { createRunStream, api } from '@/api'

const STAGES = [
  { key: 'parse', title: '任务解析', desc: '解析用户意图' },
  { key: 'match', title: '数据集匹配', desc: '语义匹配数据集' },
  { key: 'judge', title: '参数决策', desc: 'LLM 推荐超参' },
  { key: 'prepare', title: '配置与服务准备', desc: '生成配置并启动服务' },
  { key: 'ui', title: 'UI 自动化', desc: 'WebBridge 设置参数' },
  { key: 'monitor', title: '训练监控', desc: '实时跟踪训练进度' },
]

export const useTaskStore = defineStore('task', () => {
  const prompt = ref('帮我训练一个 OCR 模型，不使用数据增强，并在训练完成后生成推理报告。')
  const isRunning = ref(false)
  const logs = ref([])
  const metrics = ref({ taskType: 'OCR', dataset: '1-OCR', estimate: '约 12 分钟' })
  const progress = ref(0)
  const currentStage = ref(-1)
  const abortFn = ref(null)
  const recentTasks = ref([])
  const currentTask = ref(null)
  const assistantReply = ref('')

  const stageStates = computed(() =>
    STAGES.map((stage, index) => ({
      ...stage,
      index: index + 1,
      status: index < currentStage.value
        ? 'done'
        : index === currentStage.value
          ? 'running'
          : 'waiting',
      logs: logs.value.filter((l) => l.stage === stage.key),
    }))
  )

  function resetRun() {
    logs.value = []
    progress.value = 0
    currentStage.value = -1
    inferPrompt(prompt.value)
  }

  function inferPrompt(value) {
    const text = value.toUpperCase()
    const types = ['DETR', 'OCR', 'CLS', 'DET', 'ISEG', 'SSEG']
    const matched = types.find((type) => text.includes(type))
    const folderMap = { OCR: '1-OCR', CLS: '2-CLS', DET: '3-DET', DETR: '4-DETR', ISEG: '5-ISEG', SSEG: '6-SSEG' }
    metrics.value = {
      taskType: matched || '待识别',
      dataset: matched ? folderMap[matched] : '等待语义匹配',
      estimate: /批量|依次|再训练/.test(value) ? '约 35 分钟' : '约 12 分钟',
    }
  }

  function parseLine(line) {
    if (line.includes('task_type=')) {
      const m = line.match(/task_type=([A-Z]+)/)
      if (m) metrics.value.taskType = m[1]
    }
    if (line.includes('dataset_dir=')) {
      const m = line.match(/dataset_dir=.+?\\([^\\]+)\\训练集/)
      if (m) metrics.value.dataset = m[1]
    }
    if (line.includes('total_score=')) {
      const m = line.match(/total_score=([\d.]+)/)
      if (m) metrics.value.estimate = `${Math.round(parseFloat(m[1]) * 100)}% 匹配`
    }
    if (line.includes('训练进度')) {
      const m = line.match(/(\d+(?:\.\d+)?)%/)
      if (m) progress.value = parseFloat(m[1])
    }
    if (line.includes('解析到') || line.includes('task_type=')) currentStage.value = Math.max(currentStage.value, 0)
    if (line.includes('dataset_dir=')) currentStage.value = Math.max(currentStage.value, 1)
    if (line.includes('LLM 顾问推荐超参')) currentStage.value = Math.max(currentStage.value, 2)
    if (line.includes('frontend ready') || line.includes('配置文件已生成')) currentStage.value = Math.max(currentStage.value, 3)
    if (line.includes('点击训练参数') || line.includes('WebBridge')) currentStage.value = Math.max(currentStage.value, 4)
    if (line.includes('训练进度')) currentStage.value = Math.max(currentStage.value, 5)
  }

  function appendLog(line) {
    const chatMarker = '[CHAT_RESPONSE_JSON] '
    if (line.startsWith(chatMarker)) {
      try {
        assistantReply.value = JSON.parse(line.slice(chatMarker.length))
      } catch {
        assistantReply.value = line.slice(chatMarker.length)
      }
      return
    }
    const stage = STAGES.find((s) =>
      logs.value.some((l) => l.stage === s.key) && currentStage.value === STAGES.indexOf(s)
    )?.key || (currentStage.value >= 0 ? STAGES[currentStage.value].key : 'global')
    const entry = { time: new Date().toLocaleTimeString(), line, stage }
    logs.value.push(entry)
    parseLine(line)
  }

  function startRun() {
    if (!prompt.value.trim() || isRunning.value) return
    resetRun()
    isRunning.value = true
    currentTask.value = { prompt: prompt.value, startAt: new Date(), status: 'running' }

    abortFn.value = createRunStream(prompt.value, (data) => {
      if (data.type === 'start') {
        appendLog(`[INFO] 用户输入: ${data.prompt}`)
      } else if (data.type === 'log') {
        appendLog(data.line)
      } else if (data.type === 'done') {
        isRunning.value = false
        appendLog(`[INFO] 任务结束，退出码 ${data.code}，耗时 ${data.elapsed}s`)
        currentTask.value.status = data.code === 0 ? 'done' : 'error'
        currentTask.value.elapsed = data.elapsed
        loadRecentTasks()
      } else if (data.type === 'error') {
        isRunning.value = false
        appendLog(`[ERROR] ${data.message}`)
        currentTask.value.status = 'error'
      }
    })
  }

  function stopRun() {
    if (abortFn.value) {
      abortFn.value()
      abortFn.value = null
    }
    isRunning.value = false
    currentTask.value = null
  }

  async function loadRecentTasks() {
    try {
      recentTasks.value = await api.tasks()
    } catch (e) {
      console.error('加载最近任务失败', e)
    }
  }

  watch(prompt, (value) => {
    if (!isRunning.value) inferPrompt(value)
  })

  return {
    prompt,
    isRunning,
    logs,
    metrics,
    progress,
    currentStage,
    stageStates,
    recentTasks,
    currentTask,
    assistantReply,
    startRun,
    stopRun,
    loadRecentTasks,
    appendLog,
  }
})
