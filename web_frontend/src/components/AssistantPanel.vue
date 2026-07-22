<script setup>
import { computed } from 'vue'
import { useTaskStore } from '@/stores/taskStore'

const task = useTaskStore()
const reply = computed(() => task.assistantReply || '我可以帮你分析训练需求、解释失败原因，或根据最近的运行记录给出下一步建议。')
</script>

<template>
  <section class="card assistant-card">
    <div class="assistant-head">
      <div class="assistant-title">
        <span class="assistant-avatar">✦</span>
        <div><span class="eyebrow">COPILOT</span><h3>训练助手</h3></div>
      </div>
      <span class="online"><i />在线</span>
    </div>

    <div class="assistant-message">
      <p>{{ reply }}</p>
      <div class="message-meta"><span>AutoTrain Assistant</span><span>刚刚</span></div>
    </div>

    <div class="assistant-actions">
      <button @click="task.prompt = '分析最近一次训练失败的原因'">分析失败任务</button>
      <button @click="task.prompt = '告诉我最新模型和 Baseline 的差异'">对比模型</button>
      <button @click="task.prompt = '解释当前训练配置中的关键参数'">解释配置</button>
    </div>
    <div class="assistant-input">
      <span>输入问题后在上方工作区提交</span>
      <button>↗</button>
    </div>
  </section>
</template>

<style scoped>
.assistant-card { padding: 18px; background: linear-gradient(155deg, #fff 0%, #fbfaff 100%); }
.assistant-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 13px; }
.assistant-title { display: flex; align-items: center; gap: 9px; }
.assistant-title h3 { margin: 0; color: var(--text-strong); font-size: 13px; }
.assistant-title .eyebrow { margin-bottom: 2px; font-size: 7px; }
.assistant-avatar { width: 32px; height: 32px; display: grid; place-items: center; border-radius: 10px; background: linear-gradient(135deg, #171c31, #2b3150); color: #a99ffc; font-size: 13px; box-shadow: 0 7px 16px rgba(20,26,47,.16); }
.online { display: flex; align-items: center; gap: 5px; color: var(--success); font-size: 8px; font-weight: 700; }
.online i { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.assistant-message { position: relative; padding: 13px 14px; border: 1px solid #e3dfff; border-radius: 13px 13px 13px 3px; background: #f7f5ff; }
.assistant-message p { max-height: 260px; overflow-y: auto; margin: 0; padding-right: 5px; color: #4d456e; font-size: 9px; line-height: 1.65; white-space: pre-wrap; overflow-wrap: anywhere; }
.message-meta { display: flex; justify-content: space-between; margin-top: 10px; color: #9a94b2; font-size: 7px; }
.assistant-actions { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
.assistant-actions button { padding: 6px 8px; border: 1px solid var(--line); border-radius: 7px; background: #fff; color: #687187; cursor: pointer; font-size: 7px; }
.assistant-actions button:hover { border-color: #cfc9ff; color: var(--primary); }
.assistant-input { display: flex; align-items: center; justify-content: space-between; margin-top: 10px; padding: 8px 8px 8px 10px; border: 1px solid var(--line); border-radius: 10px; background: #fff; color: #a0a7b7; font-size: 7px; }
.assistant-input button { width: 24px; height: 24px; border: 0; border-radius: 7px; background: var(--primary); color: #fff; cursor: pointer; }
</style>
