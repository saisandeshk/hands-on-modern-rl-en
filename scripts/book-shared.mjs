/* global process */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __filename = fileURLToPath(import.meta.url)

export const rootDir = path.resolve(path.dirname(__filename), '..')
export const docsDir = path.join(rootDir, 'docs')
export const distDir = path.join(docsDir, '.vitepress', 'dist')
export const configPath = path.join(docsDir, '.vitepress', 'config.mjs')
export const packageJsonPath = path.join(rootDir, 'package.json')
export const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
export const logoPath = '../public/readme/readmelogo.png'

// --- Book metadata ---

const sanbuGithubUrl =
  process.env.PDF_SANBU_GITHUB_URL || 'https://github.com/sanbuphy'

// Fork that publishes this English-only edition.
export const forkGithubUrl =
  'https://github.com/saisandeshk/hands-on-modern-rl-en'

export const bookVersion = process.env.PDF_VERSION || '0.1.0'

export const bookAuthor =
  process.env.PDF_AUTHORS ||
  process.env.PDF_AUTHOR ||
  packageJson.author ||
  `WalkingLabs; sanbuphy (${sanbuGithubUrl})`

export const bookAdaptedBy = process.env.PDF_ADAPTED_BY || 'saisandeshk'

export const bookTitle = 'Hands-On Modern RL'

export const bookSubtitle = 'From Code to Principles'

export const bookTagline =
  'A practical guide to modern reinforcement learning: classical control, LLM post-training, RLVR and multimodal agents'

export const bookLicense = 'CC-BY-NC-SA-4.0'

export const bookBuildDate =
  process.env.PDF_BUILD_DATE ||
  new Intl.DateTimeFormat('en-US', {
    dateStyle: 'long',
    timeZone: 'UTC'
  }).format(new Date())

export function repositoryUrl() {
  if (process.env.PDF_GITHUB_URL) return process.env.PDF_GITHUB_URL

  const githubRepository = process.env.GITHUB_REPOSITORY
  if (githubRepository && githubRepository.includes('/')) {
    return `https://github.com/${githubRepository.replace(/\.git$/, '')}`
  }

  const repository =
    typeof packageJson.repository === 'string'
      ? packageJson.repository
      : packageJson.repository?.url || ''

  const sshMatch = repository.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
  if (sshMatch) {
    return `https://github.com/${sshMatch[1]}/${sshMatch[2]}`
  }

  const normalized = repository
    .replace(/^git\+/, '')
    .replace(/^https?:\/\/github\.com\//, '')
    .replace(/^git@github\.com:/, '')
    .replace(/\.git$/, '')

  if (normalized.includes('/')) {
    return `https://github.com/${normalized}`
  }

  return forkGithubUrl
}

export const bookGithubUrl = repositoryUrl()

// Original project (attribution only).
export const originalGithubUrl =
  'https://github.com/walkinglabs/hands-on-modern-rl'

// --- Utility functions ---

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

export function stripFrontmatter(value) {
  if (!value.startsWith('---\n')) return value
  const end = value.indexOf('\n---\n', 4)
  if (end === -1) return value
  return value.slice(end + 5)
}

export function stripScriptSetup(value) {
  return value.replace(/^\s*<script\b[\s\S]*?<\/script>\s*/i, '')
}

export function flattenSidebar(items, pages = [], seen = new Set()) {
  for (const item of items || []) {
    if (item.link && !item.link.includes('#') && !seen.has(item.link)) {
      seen.add(item.link)
      pages.push({ title: item.text || item.link, link: item.link })
    }
    if (item.items) {
      flattenSidebar(item.items, pages, seen)
    }
  }
  return pages
}

export function splitSuffix(value) {
  const hashIndex = value.indexOf('#')
  const queryIndex = value.indexOf('?')
  const suffixIndexes = [hashIndex, queryIndex].filter((index) => index >= 0)
  const suffixIndex = suffixIndexes.length ? Math.min(...suffixIndexes) : -1

  if (suffixIndex === -1) {
    return { clean: value, suffix: '' }
  }

  return {
    clean: value.slice(0, suffixIndex),
    suffix: value.slice(suffixIndex)
  }
}

export function pagePathForLink(link) {
  const { clean } = splitSuffix(link.replace(/^\//, ''))
  const normalized = clean.replace(/\/$/, '')
  const candidates = [
    path.join(docsDir, `${normalized}.md`),
    path.join(docsDir, normalized, 'index.md')
  ]

  return candidates.find((candidate) => fs.existsSync(candidate)) || null
}

export async function loadVitePressConfig() {
  const module = await import(pathToFileURL(configPath).href)
  return module.default
}

export function loadSidebar(config) {
  return config.themeConfig?.sidebar?.['/'] || []
}

// --- Front matter content (structured data) ---

export const frontMatter = {
  cover: {
    kicker: 'Open Textbook · Modern Reinforcement Learning',
    titleEn: bookTitle,
    subtitle: bookSubtitle,
    description: `${bookTagline}. Starting from runnable code, training curves and failure cases, the book works step by step through MDPs, policy gradients, PPO, DPO, GRPO, RLVR, agentic RL and multimodal reinforcement learning.`,
    publisher:
      'Original work by WalkingLabs · English edition adapted by saisandeshk'
  },
  intro: {
    title: 'About This Book',
    paragraphs: [
      '<strong>Hands-On Modern RL</strong> is an open e-book on the practice of modern reinforcement learning. It is not a web site mechanically printed to PDF: the course content has been arranged into a book that can be read offline, cited, and iterated on over time.',
      'The book follows a "practice first" path. It starts from runnable code, training curves, metric diagnostics and failure modes, so that readers see how an agent improves its behavior by trial and error against rewards, and only then returns to the core concepts: states, value functions, the Bellman equation, policy gradients, reward modeling and credit assignment.',
      'The content moves from CartPole, DQN, Actor-Critic and PPO into LLM post-training, and goes on to cover RLHF, DPO, GRPO, RLVR, tool-using agents, Deep Research, VLM reinforcement learning, embodied intelligence and multi-agent self-play.'
    ],
    audience: {
      title: 'Who This Book Is For',
      items: [
        'Machine learning engineers moving from supervised learning to reinforcement learning;',
        'Researchers and students preparing to read papers on modern RL, LLM alignment and agentic RL;',
        'LLM practitioners who want to understand RLHF, DPO, GRPO, RLVR and post-training systems;',
        'Self-directed learners who prefer to start from code, experiments and visualizations before the derivations.'
      ]
    },
    howToRead: {
      title: 'How to Read This Book',
      paragraphs: [
        'If you are new to reinforcement learning, read the chapters in order. If you already know classical RL, you can start from the PPO, DPO, GRPO and agentic RL chapters and return to the earlier math and algorithm chapters whenever you need to fill in a concept.'
      ]
    }
  },
  publicationNote: {
    title: 'Edition Notes',
    attribution: `Hands-On Modern RL is an original work by WalkingLabs (${originalGithubUrl}), licensed under CC BY-NC-SA 4.0. This English-only edition is adapted by saisandeshk (${forkGithubUrl}) under the same license.`,
    iterationNote:
      'This is an e-book that keeps evolving with the open-source course. Version v0.1.0 already contains the full table of contents, chapter opener pages, per-page references, PDF bookmarks, a cover and front matter; later versions will add finer typesetting, an index, a glossary, exercise answers and print-quality layout details.',
    disclaimer:
      'If a chapter is still changing quickly, please refer to the latest GitHub repository. Corrections, extra examples and better exercises are welcome through issues, pull requests or the course discussion channels.',
    licenseSection: {
      title: 'License and Citation',
      text: 'This book is released under CC BY-NC-SA 4.0, as stated in the repository, and is suitable for study, teaching and non-commercial sharing. When citing it, please give the title, version, authors and the GitHub repository address.'
    }
  }
}
