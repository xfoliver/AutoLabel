<script setup>
import { ref } from 'vue'
import { useTaskStore } from '@/stores/taskStore'

const task = useTaskStore()
const active = ref('all')
const templates = [
  { code: 'OCR', title: 'OCR 基础训练', desc: '标准参数 · 无增强', color: 'violet', prompt: '训练一个 OCR 模型，不使用数据增强' },
  { code: 'DET', title: '目标检测快测', desc: '低 Epoch · 快速验证', color: 'cyan', prompt: '训练一个 DET 目标检测模型，Epoch 设为 5' },
  { code: 'BL', title: 'Baseline 对比', desc: '训练后自动评估', color: 'amber', prompt: '训练一个 CLS 模型并与 Baseline 对比' },
  { code: 'BAT', title: '多模型队列', desc: 'CLS → OCR → SSEG', color: 'green', prompt: '依次训练 CLS、OCR 和 SSEG 三个模型' },
]

function applyTemplate(template) {
  task.prompt = template.prompt
}
</script>

<template>
  <section class="card template-card">
    <div class="card-head">
      <div><span class="eyebrow">QUICK START</span><h3>快捷模板</h3></div>
      <button class="manage">管理 →</button>
    </div>
    <div class="template-tabs">
      <button :class="{ active: active === 'all' }" @click="active = 'all'">全部</button>
      <button :class="{ active: active === 'mine' }" @click="active = 'mine'">我的模板</button>
    </div>
    <div class="template-list">
      <button v-for="template in templates" :key="template.title" class="template" @click="applyTemplate(template)">
        <span :class="['template-icon', template.color]">{{ template.code }}</span>
        <span class="template-copy"><strong>{{ template.title }}</strong><small>{{ template.desc }}</small></span>
        <span class="arrow">›</span>
      </button>
    </div>
  </section>
</template>

<style scoped>
.template-card { padding: 18px; }
.manage { padding: 4px 0; border: 0; background: transparent; color: var(--primary); cursor: pointer; font-size: 9px; font-weight: 700; }
.template-tabs { display: flex; gap: 16px; margin: -3px 0 11px; border-bottom: 1px solid var(--line); }
.template-tabs button { position: relative; padding: 0 0 9px; border: 0; background: transparent; color: var(--muted); cursor: pointer; font-size: 9px; font-weight: 700; }
.template-tabs button.active { color: var(--text-strong); }
.template-tabs button.active::after { position: absolute; right: 0; bottom: -1px; left: 0; height: 2px; border-radius: 2px; background: var(--primary); content: ''; }
.template-list { display: flex; flex-direction: column; gap: 7px; }
.template { display: grid; grid-template-columns: 34px minmax(0,1fr) auto; align-items: center; gap: 10px; width: 100%; padding: 9px; border: 1px solid transparent; border-radius: 11px; background: #f8f9fb; cursor: pointer; text-align: left; transition: .17s ease; }
.template:hover { border-color: #dedafc; background: #fff; transform: translateX(2px); }
.template-icon { width: 34px; height: 34px; display: grid; place-items: center; border-radius: 9px; font-size: 8px; font-weight: 850; }
.violet { color: var(--primary); background: var(--primary-soft); }
.cyan { color: #178c87; background: var(--cyan-soft); }
.amber { color: #b97112; background: #fff2df; }
.green { color: var(--success); background: #e8f7f1; }
.template-copy { min-width: 0; }
.template-copy strong, .template-copy small { display: block; }
.template-copy strong { color: #30394c; font-size: 10px; }
.template-copy small { overflow: hidden; margin-top: 3px; color: var(--muted); font-size: 8px; text-overflow: ellipsis; white-space: nowrap; }
.arrow { color: #a4abba; font-size: 17px; }
</style>
