<script setup>
import { computed, nextTick, ref } from 'vue'
import {
  highlightPython,
  scrollCodeLineIntoView
} from '../../shared/code-focus-utils.js'
import grpoCode from '../snippets/grpo-code-map.py?raw'

const props = defineProps({
  focus: {
    type: String,
    default: 'overview'
  },
  title: {
    type: String,
    default: ''
  }
})

const lines = grpoCode.trimEnd().split('\n')
const hovered = ref(false)
const pinned = ref(false)
const suppressHover = ref(false)
const codePre = ref(null)

const segments = [
  { id: 'A', label: 'Group sampling', range: [13, 56] },
  { id: 'B', label: 'Rule-based reward', range: [59, 77] },
  { id: 'C', label: 'Group-relative advantage', range: [80, 92] },
  { id: 'D', label: 'Per-token logprob', range: [95, 113] },
  { id: 'E', label: 'Per-token ratio / clip', range: [116, 131] },
  { id: 'F', label: 'KL and length normalization', range: [132, 194] },
  { id: 'G', label: 'Training step', range: [197, 225] },
  { id: 'H', label: 'Training loop', range: [228, 245] }
]

const focusMap = {
  overview: {
    title: 'Full GRPO Code Map',
    active: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    compactRanges: [
      [13, 56],
      [80, 113],
      [116, 194],
      [197, 245]
    ],
    highlight: [
      15, 16, 27, 36, 37, 38, 42, 47, 52, 53, 54, 56, 82, 83, 84, 86, 92, 96,
      101, 102, 106, 109, 113, 126, 127, 128, 129, 130, 133, 134, 136, 137, 141,
      143, 147, 167, 175, 176, 179, 186, 207, 213, 219, 221, 223, 224, 232
    ]
  },
  sampling: {
    title: 'Sample multiple responses per prompt',
    active: ['A'],
    compactRanges: [[13, 56]],
    highlight: [15, 16, 27, 36, 37, 38, 42, 47, 52, 53, 54, 56]
  },
  reward: {
    title: 'Rule-based reward: correct answer and format',
    active: ['B'],
    compactRanges: [[59, 77]],
    highlight: [62, 64, 65, 66, 67, 75, 76, 77]
  },
  advantages: {
    title: 'Group-normalized advantages',
    active: ['C'],
    compactRanges: [[80, 92]],
    highlight: [82, 83, 84, 86, 87, 88, 92]
  },
  logprob: {
    title: 'Per-token log probability of the response',
    active: ['D'],
    compactRanges: [[95, 113]],
    highlight: [96, 98, 99, 101, 102, 106, 109, 113]
  },
  clip: {
    title: 'GRPO per-token ratio and PPO-style clipping',
    active: ['E'],
    compactRanges: [[116, 154]],
    highlight: [126, 127, 128, 129, 130, 136, 137, 141]
  },
  kl: {
    title: 'Per-token KL, response-length normalization, and total loss',
    active: ['E', 'F'],
    compactRanges: [[126, 154]],
    highlight: [133, 134, 136, 137, 141, 143, 147]
  },
  train: {
    title: 'GRPO training step and online loop',
    active: ['G', 'H'],
    compactRanges: [[197, 245]],
    highlight: [207, 213, 219, 221, 222, 223, 224, 232, 240, 244]
  }
}

const config = computed(() => focusMap[props.focus] || focusMap.overview)
const isExpanded = computed(
  () => pinned.value || (hovered.value && !suppressHover.value)
)
const activeTitle = computed(() => props.title || config.value.title)
const toggleLabel = computed(() => {
  if (pinned.value) return 'Collapse full code'
  return isExpanded.value ? 'Pin full code' : 'Expand full code'
})

const highlighted = computed(() => new Set(config.value.highlight))
const activeSegments = computed(() => new Set(config.value.active))
const primaryFocusLine = computed(() => {
  if (config.value.highlight?.length) return config.value.highlight[0]

  const [firstRange] = normalizeRanges(config.value.compactRanges)
  return firstRange?.[0] || 1
})

function normalizeRanges(ranges) {
  return ranges
    .map(([start, end]) => [Math.max(1, start), Math.min(lines.length, end)])
    .filter(([start, end]) => start <= end)
    .sort((a, b) => a[0] - b[0])
}

const visibleRows = computed(() => {
  const ranges = isExpanded.value
    ? [[1, lines.length]]
    : normalizeRanges(config.value.compactRanges)
  const rows = []
  let previousEnd = 0

  for (const [start, end] of ranges) {
    if (previousEnd && start > previousEnd + 1) {
      rows.push({ type: 'gap', id: `${previousEnd}-${start}` })
    }

    for (let number = start; number <= end; number += 1) {
      rows.push({
        type: 'code',
        number,
        text: lines[number - 1],
        html: highlightPython(lines[number - 1]),
        isHighlight: highlighted.value.has(number),
        isMarker: lines[number - 1].trimStart().startsWith('# [')
      })
    }

    previousEnd = end
  }

  return rows
})

function togglePinned() {
  if (pinned.value) {
    pinned.value = false
    hovered.value = false
    suppressHover.value = true
    return
  }

  pinned.value = true
  suppressHover.value = false
  scrollToLine(primaryFocusLine.value)
}

function scrollToLine(lineNumber, behavior = 'smooth') {
  nextTick(() => {
    scrollCodeLineIntoView(codePre.value, lineNumber, { behavior })
  })
}

function expandAndPin(lineNumber = primaryFocusLine.value) {
  pinned.value = true
  suppressHover.value = false
  scrollToLine(lineNumber)
}

function handleMouseEnter() {
  if (!suppressHover.value) {
    const wasExpanded = isExpanded.value
    hovered.value = true
    if (!wasExpanded) scrollToLine(primaryFocusLine.value, 'auto')
  }
}

function handleMouseLeave() {
  hovered.value = false
  suppressHover.value = false
}
</script>

<template>
  <section
    class="grpo-code-focus"
    :class="{ 'is-expanded': isExpanded }"
    @mouseenter="handleMouseEnter"
    @mouseleave="handleMouseLeave"
  >
    <button
      class="grpo-code-focus-header"
      type="button"
      :aria-expanded="isExpanded"
      @click="togglePinned"
    >
      <div class="grpo-code-focus-title">
        <span class="grpo-code-focus-kicker">GRPO code lens</span>
        <strong>{{ activeTitle }}</strong>
      </div>
      <span class="grpo-code-focus-toggle">
        {{ toggleLabel }}
      </span>
    </button>

    <div class="grpo-code-focus-segments" aria-label="GRPO code structure">
      <button
        v-for="segment in segments"
        :key="segment.id"
        class="grpo-code-focus-segment"
        :class="{ 'is-active': activeSegments.has(segment.id) }"
        type="button"
        @click="expandAndPin(segment.range[0])"
      >
        <b>[{{ segment.id }}]</b>
        {{ segment.label }}
      </button>
    </div>

    <button class="grpo-code-focus-status" type="button" @click="togglePinned">
      <span>{{ isExpanded ? 'Full code view' : 'Focused view' }}</span>
      <span>
        {{
          pinned ? 'Click to return to focused view' : 'Click to pin full code'
        }}
      </span>
    </button>

    <pre
      ref="codePre"
      class="grpo-code-focus-pre"
      tabindex="0"
      data-lang="python"
    ><code class="language-python"><template
      v-for="row in visibleRows"
      :key="row.type === 'gap' ? row.id : row.number"
    ><span v-if="row.type === 'gap'" class="grpo-code-focus-gap">        ⋮
</span><span
      v-else
      class="grpo-code-focus-line"
      :class="{
        'is-highlight': row.isHighlight,
        'is-marker': row.isMarker
      }"
      :data-line-number="row.number"
    ><span class="grpo-code-focus-number">{{ String(row.number).padStart(3, ' ') }}</span><span class="grpo-code-focus-text" v-html="row.html"></span>
</span></template></code></pre>
  </section>
</template>

<style scoped>
.grpo-code-focus {
  margin: 18px 0 26px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
  background: var(--vp-code-block-bg);
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
}

.grpo-code-focus-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
  padding: 12px 14px;
  border-bottom: 1px solid var(--vp-c-divider);
  border-top: 0;
  border-right: 0;
  border-left: 0;
  background: var(--vp-c-bg-soft);
  font: inherit;
  cursor: pointer;
  text-align: left;
}

.grpo-code-focus-title {
  min-width: 0;
  display: grid;
  gap: 2px;
}

.grpo-code-focus-title strong {
  font-size: 14px;
  line-height: 1.35;
  color: var(--vp-c-text-1);
}

.grpo-code-focus-kicker {
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  line-height: 1.2;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
}

.grpo-code-focus-toggle {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 0 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-bg);
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
}

.grpo-code-focus-toggle:hover,
.grpo-code-focus-toggle:focus-visible {
  border-color: var(--vp-c-brand-1);
  outline: none;
}

.grpo-code-focus-segments {
  display: flex;
  gap: 6px;
  overflow-x: auto;
  padding: 10px 14px;
  border-bottom: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
}

.grpo-code-focus-segment {
  flex: 0 0 auto;
  padding: 4px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  font: inherit;
  font-size: 12px;
  line-height: 1.3;
  cursor: pointer;
}

.grpo-code-focus-segment.is-active {
  border-color: rgba(63, 81, 181, 0.38);
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  font-weight: 700;
}

.grpo-code-focus-status {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
  padding: 7px 14px;
  border-top: 0;
  border-right: 0;
  border-bottom: 1px solid var(--vp-c-divider);
  border-left: 0;
  color: var(--vp-c-text-3);
  background: var(--vp-code-block-bg);
  font: inherit;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
}

.grpo-code-focus-pre {
  position: relative;
  max-height: 460px;
  margin: 0;
  padding: 10px 0;
  overflow: auto;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  line-height: 1.55;
  background: var(--vp-code-block-bg);
}

.grpo-code-focus-pre::before {
  content: attr(data-lang);
  position: sticky;
  top: 0;
  float: right;
  padding: 0 12px 4px;
  color: var(--vp-c-text-3);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  user-select: none;
}

.grpo-code-focus.is-expanded .grpo-code-focus-pre {
  max-height: 720px;
}

.grpo-code-focus-pre code {
  display: block;
  min-width: max-content;
  color: var(--vp-code-block-color);
}

.grpo-code-focus-line,
.grpo-code-focus-gap {
  display: grid;
  grid-template-columns: 48px minmax(0, 1fr);
  min-height: 20px;
  padding: 0 14px 0 0;
}

.grpo-code-focus-gap {
  display: block;
  padding-left: 54px;
  color: var(--vp-c-text-3);
  user-select: none;
}

.grpo-code-focus-number {
  padding-right: 12px;
  color: var(--vp-c-text-3);
  text-align: right;
  user-select: none;
}

.grpo-code-focus-text {
  white-space: pre;
}

.grpo-code-focus-text :deep(.py-keyword) {
  color: #0000ff;
  font-weight: 500;
}

.grpo-code-focus-text :deep(.py-builtin) {
  color: #795e26;
}

.grpo-code-focus-text :deep(.py-class) {
  color: #267f99;
}

.grpo-code-focus-text :deep(.py-function) {
  color: #795e26;
}

.grpo-code-focus-text :deep(.py-module),
.grpo-code-focus-text :deep(.py-variable),
.grpo-code-focus-text :deep(.py-property),
.grpo-code-focus-text :deep(.py-self) {
  color: #001080;
}

.grpo-code-focus-text :deep(.py-constant) {
  color: #0000ff;
}

.grpo-code-focus-text :deep(.py-string) {
  color: #a31515;
}

.grpo-code-focus-text :deep(.py-number) {
  color: #098658;
}

.grpo-code-focus-text :deep(.py-comment) {
  color: #008000;
  font-style: italic;
}

.grpo-code-focus-text :deep(.py-decorator) {
  color: #af00db;
}

.grpo-code-focus-text :deep(.py-operator) {
  color: #000000;
}

.grpo-code-focus-line.is-marker {
  color: var(--vp-c-brand-1);
  font-weight: 700;
}

.grpo-code-focus-line.is-highlight {
  box-shadow: inset 3px 0 0 var(--vp-c-brand-1);
  background: rgba(63, 81, 181, 0.12);
  font-weight: 700;
}

.dark .grpo-code-focus {
  box-shadow: none;
}

.dark .grpo-code-focus-line.is-highlight {
  background: rgba(129, 140, 248, 0.16);
}

.dark .grpo-code-focus-text :deep(.py-keyword) {
  color: #569cd6;
}

.dark .grpo-code-focus-text :deep(.py-builtin) {
  color: #dcdcaa;
}

.dark .grpo-code-focus-text :deep(.py-class) {
  color: #4ec9b0;
}

.dark .grpo-code-focus-text :deep(.py-function) {
  color: #dcdcaa;
}

.dark .grpo-code-focus-text :deep(.py-module),
.dark .grpo-code-focus-text :deep(.py-variable),
.dark .grpo-code-focus-text :deep(.py-property),
.dark .grpo-code-focus-text :deep(.py-self) {
  color: #9cdcfe;
}

.dark .grpo-code-focus-text :deep(.py-constant) {
  color: #569cd6;
}

.dark .grpo-code-focus-text :deep(.py-string) {
  color: #ce9178;
}

.dark .grpo-code-focus-text :deep(.py-number) {
  color: #b5cea8;
}

.dark .grpo-code-focus-text :deep(.py-comment) {
  color: #6a9955;
}

.dark .grpo-code-focus-text :deep(.py-decorator) {
  color: #c586c0;
}

.dark .grpo-code-focus-text :deep(.py-operator) {
  color: #d4d4d4;
}

@media (max-width: 640px) {
  .grpo-code-focus-header,
  .grpo-code-focus-status {
    align-items: flex-start;
    flex-direction: column;
  }

  .grpo-code-focus-toggle {
    width: 100%;
  }

  .grpo-code-focus-pre {
    font-size: 12px;
  }
}
</style>
