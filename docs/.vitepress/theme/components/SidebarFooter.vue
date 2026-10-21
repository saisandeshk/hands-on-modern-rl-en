<script setup>
import { useData } from 'vitepress'
import { computed } from 'vue'
import { Github, Moon, Sun } from 'lucide-vue-next'

const { isDark, theme } = useData()

const appearanceLabel = computed(() =>
  isDark.value ? 'Switch to light mode' : 'Switch to dark mode'
)

function toggleAppearance() {
  isDark.value = !isDark.value
}

const githubUrl = computed(() => {
  const social = theme.value.socialLinks?.find((l) => l.icon === 'github')
  if (social?.link) return social.link
  return 'https://github.com/saisandeshk/hands-on-modern-rl-en'
})
</script>

<template>
  <div class="ct-sidebar-footer">
    <div class="ct-sidebar-footer-divider" />
    <div class="ct-sidebar-footer-row">
      <div class="ct-sidebar-footer-actions">
        <button
          class="ct-sidebar-footer-btn"
          type="button"
          :title="appearanceLabel"
          :aria-label="appearanceLabel"
          @click="toggleAppearance"
        >
          <Sun v-if="isDark" :size="16" :stroke-width="2" aria-hidden="true" />
          <Moon v-else :size="16" :stroke-width="2" aria-hidden="true" />
        </button>
        <slot name="settings" />
      </div>
      <a
        class="ct-sidebar-footer-link"
        :href="githubUrl"
        target="_blank"
        rel="noopener noreferrer"
        title="GitHub"
        aria-label="GitHub"
      >
        <Github :size="16" :stroke-width="2" aria-hidden="true" />
      </a>
    </div>
  </div>
</template>
