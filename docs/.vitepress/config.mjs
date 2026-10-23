/* global process */
import { defineConfig } from 'vitepress'
import { createLogger } from 'vite'
import { MermaidMarkdown } from 'vitepress-plugin-mermaid'
import { createRequire } from 'module'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const require = createRequire(import.meta.url)
const markdownItFootnote = require('markdown-it-footnote')
const markdownItContainer = require('markdown-it-container')
const katex = require('katex')

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const packageJsonPath = path.resolve(__dirname, '../../package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
const docsRoot = path.resolve(__dirname, '..')
const assetManifestPath = path.resolve(
  docsRoot,
  'public/optimized/asset-manifest.json'
)

function loadAssetManifest() {
  if (!fs.existsSync(assetManifestPath)) return { assets: {} }

  try {
    return JSON.parse(fs.readFileSync(assetManifestPath, 'utf8'))
  } catch {
    return { assets: {} }
  }
}

const assetManifest = loadAssetManifest()

const isVercel = process.env.VERCEL === '1' || !!process.env.VERCEL_URL

function parseRepository() {
  const repositoryUrl =
    process.env.GITHUB_REPOSITORY ||
    packageJson.repository?.url ||
    packageJson.repository ||
    ''

  if (repositoryUrl.includes('/')) {
    if (repositoryUrl.includes(':')) {
      const sshMatch = repositoryUrl.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
      if (sshMatch) {
        return { owner: sshMatch[1], repo: sshMatch[2] }
      }
    }

    const normalized = repositoryUrl
      .replace(/^https?:\/\/github\.com\//, '')
      .replace(/^git@github\.com:/, '')
      .replace(/\.git$/, '')

    if (normalized.includes('/')) {
      const [owner, repo] = normalized.split('/')
      return { owner, repo }
    }
  }

  return { owner: 'walkinglabs', repo: packageJson.name || 'course-template' }
}

const { owner, repo } = parseRepository()
const base = process.env.BASE || (isVercel ? '/' : `/${repo}/`)
const siteUrl = process.env.SITE_URL || `https://${owner}.github.io/${repo}`
const editLinkPattern = `https://github.com/${owner}/${repo}/edit/main/docs/:path`
const enableLocalSearch = process.env.LOCAL_SEARCH !== '0'
const mermaidConfig = {
  securityLevel: 'loose',
  startOnLoad: false
}

function mermaidConfigPlugin() {
  const virtualModuleId = 'virtual:mermaid-config'
  const resolvedVirtualModuleId = `\0${virtualModuleId}`

  return {
    name: 'local-mermaid-config',
    resolveId(id) {
      if (id === virtualModuleId) {
        return resolvedVirtualModuleId
      }
    },
    load(id) {
      if (id === resolvedVirtualModuleId) {
        return `export default ${JSON.stringify(mermaidConfig)}`
      }
    }
  }
}

function normalizeBrokenDocPathPlugin() {
  const canonicalSegments = ['appendix_math', 'linear-algebra-basics']

  return {
    name: 'normalize-broken-doc-path',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next()

        const url = new URL(req.url, 'http://localhost')
        if (!url.pathname.includes('appendix')) return next()

        const decodedPathname = decodeURIComponent(url.pathname)
        const normalizedDecodedPathname = decodedPathname
          .replace(/\s+/g, '')
          .replace(/appendix_m+ath/gi, canonicalSegments[0])
          .replace(/linea+r-algebra-basics/gi, canonicalSegments[1])

        if (normalizedDecodedPathname !== decodedPathname) {
          const normalizedPathname = normalizedDecodedPathname
            .split('/')
            .map((segment) => encodeURIComponent(segment))
            .join('/')
          const redirectTarget = `${normalizedPathname}${url.search}`
          res.statusCode = 302
          res.setHeader('Location', redirectTarget)
          res.end()
          return
        }

        next()
      })
    }
  }
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function slugifySearchHeading(value) {
  return value
    .normalize('NFKD')
    .replace(/[\u0300-\u036F]/g, '')
    .replace(/[\x00-\x1f]/g, '')
    .replace(/[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1')
    .toLowerCase()
}

function stripMarkdown(value) {
  return value
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_~>#-]/g, '')
    .trim()
}

function renderSearchMarkdown(src) {
  const html = []
  const slugCounts = new Map()
  let inFence = false

  for (const rawLine of src.replace(/^---[\s\S]*?---\n/, '').split('\n')) {
    const line = rawLine.trim()

    if (line.startsWith('```')) {
      inFence = !inFence
      continue
    }

    if (!line || inFence || line.startsWith(':::')) {
      continue
    }

    const heading = /^(#{1,6})\s+(.+)$/.exec(line)
    if (heading) {
      const level = heading[1].length
      const title = stripMarkdown(heading[2])
      const escapedTitle = escapeHtml(title)
      const baseSlug = slugifySearchHeading(title)
      const count = slugCounts.get(baseSlug) || 0
      slugCounts.set(baseSlug, count + 1)
      const slug = count === 0 ? baseSlug : `${baseSlug}-${count}`
      html.push(
        `<h${level}>${escapedTitle}<a class="header-anchor" href="#${slug}"></a></h${level}>`
      )
      continue
    }

    html.push(`<p>${escapeHtml(stripMarkdown(line))}</p>`)
  }

  return html.join('\n')
}

function isValidMathDelimiter(state, pos) {
  const max = state.posMax
  const prevChar = pos > 0 ? state.src.charCodeAt(pos - 1) : -1
  const nextChar = pos + 1 < max ? state.src.charCodeAt(pos + 1) : -1

  return {
    canOpen: nextChar !== 0x20 && nextChar !== 0x09,
    canClose:
      prevChar !== 0x20 &&
      prevChar !== 0x09 &&
      (nextChar < 0x30 || nextChar > 0x39)
  }
}

function mathInline(state, silent) {
  if (state.src[state.pos] !== '$') return false

  // Display math inline: $$...$$ inside a paragraph/list/blockquote line.
  // Must be tried before the single-$ branch so the closing $$ isn't read as
  // two consecutive empty inline-math delimiters.
  if (state.pos + 1 < state.posMax && state.src[state.pos + 1] === '$') {
    const start = state.pos + 2
    const close = state.src.indexOf('$$', start)
    if (close !== -1 && close < state.posMax) {
      if (!silent) {
        const token = state.push('math_inline', 'math', 0)
        token.markup = '$$'
        token.content = state.src.slice(start, close)
        token.displayMode = true
      }
      state.pos = close + 2
      return true
    }
  }

  let delimiter = isValidMathDelimiter(state, state.pos)
  if (!delimiter.canOpen) {
    if (!silent) state.pending += '$'
    state.pos += 1
    return true
  }

  const start = state.pos + 1
  const max = state.posMax
  let match = start
  while ((match = state.src.indexOf('$', match)) !== -1) {
    if (match >= max) {
      match = -1
      break
    }
    let pos = match - 1
    while (state.src[pos] === '\\') pos -= 1
    if ((match - pos) % 2 === 1) break
    match += 1
  }

  if (match === -1) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (match - start === 0) {
    if (!silent) state.pending += '$$'
    state.pos = start + 1
    return true
  }

  delimiter = isValidMathDelimiter(state, match)
  if (!delimiter.canClose) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (!silent) {
    const token = state.push('math_inline', 'math', 0)
    token.markup = '$'
    token.content = state.src.slice(start, match)
  }

  state.pos = match + 1
  return true
}

function mathBlock(state, start, end, silent) {
  let pos = state.bMarks[start] + state.tShift[start]
  const max = state.eMarks[start]

  if (pos + 2 > max) return false
  if (state.src.slice(pos, pos + 2) !== '$$') return false

  pos += 2
  let firstLine = state.src.slice(pos, max)
  let lastLine = ''
  let found = false
  let next = start

  if (silent) return true
  if (firstLine.trim().slice(-2) === '$$') {
    firstLine = firstLine.trim().slice(0, -2)
    found = true
  }

  while (!found) {
    next++
    if (next >= end) break

    pos = state.bMarks[next] + state.tShift[next]
    const lineMax = state.eMarks[next]
    if (pos < lineMax && state.tShift[next] < state.blkIndent) break

    if (state.src.slice(pos, lineMax).trim().slice(-2) === '$$') {
      const lastPos = state.src.slice(0, lineMax).lastIndexOf('$$')
      lastLine = state.src.slice(pos, lastPos)
      found = true
    }
  }

  state.line = next + 1
  const token = state.push('math_block', 'math', 0)
  token.block = true
  token.content =
    (firstLine && firstLine.trim() ? `${firstLine}\n` : '') +
    state.getLines(start + 1, next, state.tShift[start], true) +
    (lastLine && lastLine.trim() ? lastLine : '')
  token.map = [start, state.line]
  token.markup = '$$'
  return true
}

function renderKatex(content, displayMode) {
  try {
    return katex.renderToString(content, {
      displayMode,
      output: 'html',
      throwOnError: true, // Enable error throwing for debugging
      strict: false,
      trust: true
    })
  } catch (error) {
    // Log detailed error information with file context
    const fs = require('fs')
    const path = require('path')
    const markdownFile = path.join(
      process.cwd(),
      'docs/chapter10_ppo/ppo-math.md'
    )

    console.error('\n' + '='.repeat(80))
    console.error('❌ KaTeX Rendering Error')
    console.error('='.repeat(80))
    console.error(
      `Mode: ${displayMode ? 'Display Math ($$...$$)' : 'Inline Math ($...$)'}`
    )
    console.error(
      `Expression: ${content.substring(0, 200)}${content.length > 200 ? '...' : ''}`
    )
    console.error(`Error: ${error.message}`)
    if (error.position !== undefined) {
      console.error(`Position in expression: ${error.position}`)
      const start = Math.max(0, error.position - 50)
      const end = Math.min(content.length, error.position + 50)
      console.error(`Context: ...${content.substring(start, end)}...`)
    }

    // Try to find this expression in the markdown file
    try {
      const mdContent = fs.readFileSync(markdownFile, 'utf8')
      const searchStr = content.substring(0, Math.min(100, content.length))
      const index = mdContent.indexOf(searchStr)
      if (index !== -1) {
        const lineNum = mdContent.substring(0, index).split('\n').length
        console.error(
          `Location: ${markdownFile}, approximately line ${lineNum}`
        )

        // Show surrounding lines
        const lines = mdContent.split('\n')
        const startLine = Math.max(0, lineNum - 3)
        const endLine = Math.min(lines.length, lineNum + 2)
        console.error('\nSurrounding context:')
        for (let i = startLine; i < endLine; i++) {
          const marker = i + 1 === lineNum ? '>>> ' : '    '
          console.error(`${marker}${i + 1}: ${lines[i].substring(0, 100)}`)
        }
      }
    } catch (e) {
      // Ignore file reading errors
    }

    console.error('='.repeat(80) + '\n')

    // Return error HTML for visibility in the page
    return `<div style="background: #fee; border: 2px solid #c00; padding: 10px; margin: 10px 0; border-radius: 4px;">
      <strong style="color: #c00;">KaTeX Rendering Error</strong>
      <p style="margin: 5px 0;"><strong>Mode:</strong> ${displayMode ? 'Display' : 'Inline'}</p>
      <p style="margin: 5px 0;"><strong>Expression:</strong> <code>${content.substring(0, 150)}${content.length > 150 ? '...' : ''}</code></p>
      <p style="margin: 5px 0;"><strong>Error:</strong> ${error.message}</p>
      <p style="margin: 5px 0; font-size: 0.9em; color: #666;">Check browser console for detailed stack trace and file location.</p>
    </div>`
  }
}

function rescueMathInInline(md) {
  // Phase 1 (core, before inline): protect $...$ from emphasis/strong rules.
  // Replace inline-math in raw token content with unique placeholders so that
  // markdown's emphasis rule never sees the underscores inside formulas.
  let mathCounter = 0
  const mathStore = new Map()

  md.core.ruler.before('inline', 'math_inline_protect', function (state) {
    for (const token of state.tokens) {
      if (token.type !== 'inline') continue
      if (!token.content.includes('$')) continue

      token.content = token.content.replace(
        /\$([^\$]+)\$/g,
        function (match, formula) {
          const key = '\x01MATH' + mathCounter++ + '\x01'
          mathStore.set(key, formula)
          return key
        }
      )
    }
  })

  // Phase 2 (core, after inline): restore placeholders as math_inline tokens.
  md.core.ruler.after('inline', 'math_inline_restore', function (state) {
    for (const token of state.tokens) {
      if (token.type !== 'inline' || !token.children) continue

      let hasPlaceholder = false
      for (const child of token.children) {
        if (child.type === 'text' && child.content.includes('\x01')) {
          hasPlaceholder = true
          break
        }
      }
      if (!hasPlaceholder) continue

      const newChildren = []
      for (const child of token.children) {
        if (child.type === 'text' && child.content.includes('\x01')) {
          const parts = child.content.split(/(\x01MATH\d+\x01)/)
          for (const part of parts) {
            if (!part) continue
            if (part.startsWith('\x01MATH') && mathStore.has(part)) {
              const t = new state.Token('math_inline', 'math', 0)
              t.content = mathStore.get(part)
              t.markup = '$'
              newChildren.push(t)
            } else if (part.trim()) {
              const t = new state.Token('text', '', 0)
              t.content = part
              newChildren.push(t)
            }
          }
        } else {
          newChildren.push(child)
        }
      }
      token.children = newChildren
    }
  })
}

function katexMarkdown(md) {
  md.inline.ruler.push('math_inline', mathInline)
  md.block.ruler.after('blockquote', 'math_block', mathBlock, {
    alt: ['paragraph', 'reference', 'blockquote', 'list']
  })
  md.renderer.rules.math_inline = (tokens, idx) =>
    renderKatex(tokens[idx].content, tokens[idx].displayMode || false)
  md.renderer.rules.math_block = (tokens, idx) =>
    `<p>${renderKatex(tokens[idx].content, true)}</p>\n`
}

function footnoteTitlePlugin(md) {
  md.renderer.rules.footnote_block_open = (tokens, idx, options, env) => {
    const previousContentToken = [...tokens]
      .slice(0, idx)
      .reverse()
      .find((token) => token.type === 'inline' && token.content?.trim())
    const previousContent = (previousContentToken?.content || '')
      .replace(/[*_`#]/g, '')
      .trim()
    const hasManualTitle = /^References:?$/.test(previousContent)
    const title = 'References'
    const heading = hasManualTitle
      ? ''
      : `<div class="footnotes-title">${title}</div>\n`

    const separator = options.xhtmlOut
      ? '<hr class="footnotes-sep" />\n'
      : '<hr class="footnotes-sep">\n'

    return `${separator}<section class="footnotes">\n${heading}<ol class="footnotes-list">\n`
  }
}

function safeHeadingAttrs(md) {
  md.core.ruler.before('linkify', 'safe_heading_attrs', (state) => {
    for (let idx = 0; idx < state.tokens.length - 1; idx += 1) {
      const headingOpen = state.tokens[idx]
      const inline = state.tokens[idx + 1]

      if (headingOpen.type !== 'heading_open' || inline.type !== 'inline') {
        continue
      }

      const children = inline.children || []
      const lastText = [...children]
        .reverse()
        .find((token) => token.type === 'text')
      if (!lastText) continue

      const match = lastText.content.match(
        /\s*\{((?:[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)(?:\s+[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)*)\}$/
      )
      if (!match) continue

      const attrs = match[1].trim().split(/\s+/)
      const classes = []

      for (const attr of attrs) {
        if (attr.startsWith('#')) {
          headingOpen.attrSet('id', attr.slice(1))
        } else if (attr.startsWith('.')) {
          classes.push(attr.slice(1))
        }
      }

      if (classes.length) {
        headingOpen.attrJoin('class', classes.join(' '))
      }

      lastText.content = lastText.content.slice(0, match.index)
      inline.content = inline.content.replace(match[0], '')
    }
  })
}

function isExternalAsset(value) {
  return (
    /^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(value) || value.startsWith('data:')
  )
}

function resolveMarkdownAsset(relativePagePath, src) {
  if (!src || isExternalAsset(src) || src.startsWith('/')) return null

  const hashIndex = src.indexOf('#')
  const queryIndex = src.indexOf('?')
  const suffixIndexCandidates = [hashIndex, queryIndex].filter(
    (idx) => idx >= 0
  )
  const suffixIndex = suffixIndexCandidates.length
    ? Math.min(...suffixIndexCandidates)
    : -1
  const cleanSrc = suffixIndex >= 0 ? src.slice(0, suffixIndex) : src
  const suffix = suffixIndex >= 0 ? src.slice(suffixIndex) : ''
  const pageDir = path.posix.dirname(relativePagePath || '')
  const sourcePath = path.posix
    .normalize(path.posix.join(pageDir, cleanSrc))
    .replace(/^\.\//, '')

  return {
    sourcePath,
    suffix
  }
}

function optimizedImagesPlugin(md) {
  const imageRule = md.renderer.rules.image

  md.renderer.rules.image = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const src = token.attrGet('src')
    const resolved = resolveMarkdownAsset(env.relativePath, src)
    const asset = resolved && assetManifest.assets?.[resolved.sourcePath]

    if (asset?.status === 'optimized' && asset.optimized) {
      token.attrSet('src', `${asset.optimized}${resolved.suffix}`)
      token.attrSet('data-source-src', `/src/${resolved.sourcePath}`)
    }

    if (!token.attrGet('loading')) {
      token.attrSet('loading', 'lazy')
    }

    if (!token.attrGet('decoding')) {
      token.attrSet('decoding', 'async')
    }

    return imageRule(tokens, idx, options, env, self)
  }
}

function mermaidBlockIndex(tokens, idx) {
  let index = 0

  for (let i = 0; i <= idx; i += 1) {
    const info = tokens[i].info?.trim()
    if (info === 'mermaid' || info === 'mmd') {
      index += 1
    }
  }

  return index
}

function buildMermaidManifestMap() {
  return new Map(
    (assetManifest.mermaid || [])
      .filter((block) => block.status === 'optimized' && block.optimized)
      .map((block) => [`${block.page}:${block.index}`, block])
  )
}

const mermaidManifest = buildMermaidManifestMap()

function optimizedMermaidPlugin(md) {
  const fenceRule = md.renderer.rules.fence

  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const info = token.info.trim()

    if (info === 'mermaid' || info === 'mmd') {
      const index = mermaidBlockIndex(tokens, idx)
      const block = mermaidManifest.get(`${env.relativePath}:${index}`)

      if (block?.optimized) {
        const src = md.utils.escapeHtml(block.optimized)
        const source = md.utils.escapeHtml(block.source)
        const page = md.utils.escapeHtml(block.page)

        return `<p class="mermaid-static"><img src="${src}" alt="Mermaid diagram" data-mermaid-viewer="true" data-source-src="/src/${source}" data-source-page="/src/${page}" data-source-index="${block.index}" loading="lazy" decoding="async"></p>\n`
      }
    }

    return fenceRule(tokens, idx, options, env, self)
  }
}

const nav = [
  { text: 'Preface', link: '/preface/introduction' },
  { text: 'Fundamentals', link: '/chapter01_cartpole/principles' },
  { text: 'Deep RL', link: '/chapter05_dqn/from-q-to-dqn' },
  { text: 'LLM Alignment', link: '/chapter13_rlhf/base-model-to-assistant' },
  { text: 'Agentic RL', link: '/chapter19_agentic/overview' },
  {
    text: 'Safety & Frontiers',
    link: '/chapter25_alignment_failures/classical-failures'
  },
  {
    text: 'Appendices',
    link: '/appendix_a_engineering/training-debugging'
  }
]

const sidebar = {
  '/': [
    {
      text: 'Preface · Introduction',
      collapsed: false,
      items: [
        {
          text: 'Introduction to RL',
          link: '/preface/introduction'
        },
        {
          text: 'Brief History of RL',
          link: '/preface/brief-history/'
        },
        {
          text: 'Environment Setup',
          link: '/preface/env-setup'
        }
      ]
    },
    {
      text: 'Fundamentals & Classical RL',
      collapsed: false,
      items: [
        {
          text: '1. CartPole',
          collapsed: false,
          items: [
            {
              text: '1.1 Run CartPole',
              link: '/chapter01_cartpole/principles'
            },
            {
              text: '1.2 CartPole Principles',
              link: '/chapter01_cartpole/metrics'
            },
            {
              text: '1.3 Hands-on: PPO Training Visualization',
              link: '/chapter01_cartpole/training'
            },
            {
              text: '1.4 Summary and Exercises',
              link: '/chapter01_cartpole/review'
            }
          ]
        },
        {
          text: '2. RL Problems and Basic Definitions',
          collapsed: false,
          items: [
            {
              text: '2.1 Exploration and Exploitation',
              link: '/chapter02_rl_basics/bandit'
            },
            {
              text: '2.2 Markov Decision Processes',
              link: '/chapter02_rl_basics/mdp'
            },
            {
              text: '2.3 Policy, Value and Return',
              link: '/chapter02_rl_basics/policy-value'
            },
            {
              text: '2.4 Summary and Exercises',
              link: '/chapter02_rl_basics/review'
            }
          ]
        },
        {
          text: '3. Value Functions & Bellman Equations',
          collapsed: false,
          items: [
            {
              text: '3.1 State Values and Bellman Expectation',
              link: '/chapter03_value_functions/value-bellman'
            },
            {
              text: '3.2 Action Values and Bellman Optimality',
              link: '/chapter03_value_functions/value-q'
            },
            {
              text: '3.3 Hands-on: Value Function Experiments',
              link: '/chapter03_value_functions/value-experiment'
            },
            {
              text: '3.4 Summary and Exercises',
              link: '/chapter03_value_functions/review'
            }
          ]
        },
        {
          text: '4. Classical RL Methods',
          collapsed: false,
          items: [
            {
              text: '4.1 Dynamic Programming, Monte Carlo and TD',
              link: '/chapter04_classical_methods/dp-mc-td'
            },
            {
              text: '4.2 Policy Sampling and Data Sources',
              link: '/chapter04_classical_methods/algorithm-taxonomy'
            },
            {
              text: '4.3 Reward Design',
              link: '/chapter04_classical_methods/reward-design'
            },
            {
              text: '4.4 Lab: Tabular Methods Against Exact Oracles',
              link: '/chapter04_classical_methods/lab'
            },
            {
              text: '4.5 Summary and Exercises',
              link: '/chapter04_classical_methods/review'
            }
          ]
        }
      ]
    },
    {
      text: 'Deep Reinforcement Learning',
      collapsed: false,
      items: [
        {
          text: '5. Deep Q-Networks',
          collapsed: false,
          items: [
            {
              text: '5.1 From Q-Learning to DQN',
              link: '/chapter05_dqn/from-q-to-dqn'
            },
            {
              text: '5.2 DQN Architecture and Implementation',
              link: '/chapter05_dqn/dqn-components'
            },
            {
              text: '5.3 Hands-on: LunarLander',
              link: '/chapter05_dqn/lunar-lander'
            },
            {
              text: '5.4 DQN Improvements',
              link: '/chapter05_dqn/dqn-family'
            },
            {
              text: '5.5 Hands-on: Visual Game Projects',
              link: '/chapter05_dqn/visual-game-projects'
            },
            {
              text: '5.6 Summary and Exercises',
              link: '/chapter05_dqn/review'
            }
          ]
        },
        {
          text: '6. Policy Gradient Methods',
          collapsed: false,
          items: [
            {
              text: '6.1 Policy Gradient and REINFORCE',
              link: '/chapter06_policy_gradient/reinforce'
            },
            {
              text: '6.2 Policy Gradient Improvements',
              link: '/chapter06_policy_gradient/pg-improvements'
            },
            {
              text: '6.3 Hands-on: Two-Armed Dice-Game Bandit',
              link: '/chapter06_policy_gradient/dice-game'
            },
            {
              text: '6.4 Hands-on: REINFORCE on CartPole',
              link: '/chapter06_policy_gradient/cartpole'
            },
            {
              text: '6.5 Hands-on: Value Baseline on CartPole',
              link: '/chapter06_policy_gradient/baseline-experiment'
            },
            {
              text: '6.6 Hands-on: Policy Gradient with a Value Baseline',
              link: '/chapter06_policy_gradient/cartpole-baseline'
            },
            {
              text: '6.7 Summary and Exercises',
              link: '/chapter06_policy_gradient/review'
            }
          ]
        },
        {
          text: '7. Actor-Critic Methods',
          collapsed: false,
          items: [
            {
              text: '7.1 Advantage Function',
              link: '/chapter07_actor_critic/advantage-function'
            },
            {
              text: '7.2 Actor-Critic Updates',
              link: '/chapter07_actor_critic/actor-critic'
            },
            {
              text: '7.3 Critic Training Details',
              link: '/chapter07_actor_critic/critic-training'
            },
            {
              text: '7.4 Hands-on: Pendulum Continuous Control',
              link: '/chapter07_actor_critic/pendulum'
            },
            {
              text: '7.5 Hands-on: Reproducing AlphaGo',
              link: '/chapter07_actor_critic/alphago'
            },
            {
              text: '7.6 Hands-on: BipedalWalker',
              link: '/chapter07_actor_critic/bipedalwalker'
            },
            {
              text: '7.7 Hands-on: Actor-Critic Frontier Applications',
              link: '/chapter07_actor_critic/ac-frontier'
            },
            {
              text: '7.8 Summary and Exercises',
              link: '/chapter07_actor_critic/review'
            }
          ]
        },
        {
          text: '8. PPO: Stable Actor-Critic',
          collapsed: false,
          items: [
            {
              text: '8.1 Hands-on: PPO on BipedalWalker',
              link: '/chapter08_ppo/ppo-bipedal-walker'
            },
            {
              text: '8.2 Trust Regions and PPO-Clip',
              link: '/chapter08_ppo/trust-region-clipping'
            },
            {
              text: '8.3 GAE',
              link: '/chapter08_ppo/gae-reward-model'
            },
            {
              text: '8.4 Hands-on: PPO Mathematical Derivation',
              link: '/chapter08_ppo/ppo-math'
            },
            {
              text: '8.5 Hands-on: PPO Game Projects',
              link: '/chapter08_ppo/ppo-game-benchmark'
            },
            {
              text: '8.6 Hands-on: Long-Horizon Tasks and Planning',
              link: '/chapter08_ppo/rl-long-horizon-planning'
            },
            {
              text: '8.7 Summary and Exercises',
              link: '/chapter08_ppo/review'
            }
          ]
        },
        {
          text: '9. Continuous Control and Model-Based RL',
          collapsed: false,
          items: [
            {
              text: '9.1 Deterministic Policy Gradients and DDPG',
              link: '/chapter09_continuous_control/ddpg'
            },
            {
              text: '9.2 TD3 and SAC',
              link: '/chapter09_continuous_control/td3-sac'
            },
            {
              text: '9.3 Model-Based RL',
              link: '/chapter09_continuous_control/model-based'
            },
            {
              text: '9.4 Search and World Models',
              link: '/chapter09_continuous_control/search-world-models'
            },
            {
              text: '9.5 Summary and Exercises',
              link: '/chapter09_continuous_control/review'
            }
          ]
        }
      ]
    },
    {
      text: 'Advanced RL Methods (Elective)',
      collapsed: false,
      items: [
        {
          text: '10. Offline Reinforcement Learning (Elective)',
          collapsed: false,
          items: [
            {
              text: '10.1 Offline Data and Distribution Shift',
              link: '/chapter10_offline_rl/offline-data-distribution-shift'
            },
            {
              text: '10.2 Sequence Modeling for Offline RL',
              link: '/chapter10_offline_rl/sequence-modeling'
            },
            {
              text: '10.3 Offline RL and Preference Data',
              link: '/chapter10_offline_rl/experiments'
            },
            {
              text: '10.4 Lab: Offline RL from Fixed Datasets',
              link: '/chapter10_offline_rl/lab'
            },
            {
              text: '10.5 Summary and Exercises',
              link: '/chapter10_offline_rl/review'
            }
          ]
        },
        {
          text: '11. Imitation, Inverse and Meta Reinforcement Learning (Elective)',
          collapsed: false,
          items: [
            {
              text: '11.1 Behavioral Cloning and Interactive Imitation',
              link: '/chapter11_imitation_meta_rl/bc-dagger'
            },
            {
              text: '11.2 Inverse RL and GAIL',
              link: '/chapter11_imitation_meta_rl/irl-gail'
            },
            {
              text: '11.3 Meta-RL and In-Context Adaptation',
              link: '/chapter11_imitation_meta_rl/meta-rl'
            },
            {
              text: '11.4 Lab: Behavior Cloning versus DAgger',
              link: '/chapter11_imitation_meta_rl/lab'
            },
            {
              text: '11.5 Summary and Exercises',
              link: '/chapter11_imitation_meta_rl/review'
            }
          ]
        },
        {
          text: '12. Exploration, Multi-Agent and Hierarchical RL (Elective)',
          collapsed: false,
          items: [
            {
              text: '12.1 Intrinsic Motivation and Exploration',
              link: '/chapter12_exploration_marl_hierarchical/intrinsic-motivation-exploration'
            },
            {
              text: '12.2 Multi-Agent Reinforcement Learning',
              link: '/chapter12_exploration_marl_hierarchical/marl'
            },
            {
              text: '12.3 Hierarchical RL and World Models',
              link: '/chapter12_exploration_marl_hierarchical/hierarchical'
            },
            {
              text: '12.4 Summary and Exercises',
              link: '/chapter12_exploration_marl_hierarchical/review'
            }
          ]
        }
      ]
    },
    {
      text: 'LLM Alignment & Post-Training',
      collapsed: false,
      items: [
        {
          text: '13. RLHF Pipeline',
          collapsed: false,
          items: [
            {
              text: '13.1 Base Model to Instruction Alignment',
              link: '/chapter13_rlhf/base-model-to-assistant'
            },
            {
              text: '13.2 Supervised Fine-Tuning',
              link: '/chapter13_rlhf/imitation-learning-pipeline'
            },
            {
              text: '13.3 AI Feedback and Safety Principles',
              link: '/chapter13_rlhf/hhh-practice'
            },
            {
              text: '13.4 Reinforcement Learning Fine-Tuning',
              link: '/chapter13_rlhf/standard-rlhf-pipeline'
            },
            {
              text: '13.5 Large-Scale Training Engineering',
              link: '/chapter13_rlhf/scaling-to-large-models'
            },
            {
              text: '13.6 Alignment Evaluation',
              link: '/chapter13_rlhf/evaluation'
            },
            {
              text: '13.7 Extended Practice: Reward Hacking and the Data Flywheel',
              link: '/chapter13_rlhf/extended-practice'
            },
            {
              text: '13.8 Hands-on: veRL PPO on GSM8K',
              link: '/chapter13_rlhf/verl-ppo-gsm8k'
            },
            {
              text: '13.9 Supplement: The PPO-RLHF Training Loop',
              link: '/chapter13_rlhf/ppo-rlhf-loop'
            },
            {
              text: '13.10 Summary and Exercises',
              link: '/chapter13_rlhf/review'
            }
          ]
        },
        {
          text: '14. Preference Optimization and DPO',
          collapsed: false,
          items: [
            {
              text: '14.1 DPO Objective and Derivation',
              link: '/chapter14_dpo/dpo-objective-derivation'
            },
            {
              text: '14.2 DPO Training and Evaluation Metrics',
              link: '/chapter14_dpo/metrics'
            },
            {
              text: '14.3 DPO Improvements',
              link: '/chapter14_dpo/dpo-theory-and-family'
            },
            {
              text: '14.4 Hands-on: A DPO Alignment Experiment',
              link: '/chapter14_dpo/dpo-hands-on'
            },
            {
              text: '14.5 Summary and Exercises',
              link: '/chapter14_dpo/review'
            }
          ]
        },
        {
          text: '15. GRPO and Verifiable Rewards',
          collapsed: false,
          items: [
            {
              text: '15.1 GRPO Training Mechanism',
              link: '/chapter15_grpo/grpo-practice-and-mechanism'
            },
            {
              text: '15.2 DeepSeek-R1-Zero and DAPO',
              link: '/chapter15_grpo/deepseek-dapo'
            },
            {
              text: '15.3 Hands-on: Building RLVR Rewards',
              link: '/chapter15_grpo/rlvr'
            },
            {
              text: '15.4 GRPO Improvements',
              link: '/chapter15_grpo/grpo-family'
            },
            {
              text: '15.5 RL Environments and Verifiers',
              link: '/chapter15_grpo/rl-environments'
            },
            {
              text: '15.6 Hands-on: GRPO for Financial Tool Calling',
              link: '/chapter15_grpo/financial-tool-calling-grpo'
            },
            {
              text: '15.7 On-Policy Distillation',
              link: '/chapter15_grpo/on-policy-distillation'
            },
            {
              text: '15.8 Hands-on: Code Generation with veRL',
              link: '/chapter15_grpo/verl-code-sandbox'
            },
            {
              text: '15.9 Summary and Exercises',
              link: '/chapter15_grpo/review'
            }
          ]
        },
        {
          text: '16. Reasoning Models and Test-Time Compute',
          collapsed: false,
          items: [
            {
              text: '16.1 From Language Models to Reasoning Models',
              link: '/chapter16_reasoning/emergence-and-o1'
            },
            {
              text: '16.2 R1-Zero Pure RL Reasoning',
              link: '/chapter16_reasoning/r1-zero-pure-rl-reasoning'
            },
            {
              text: '16.3 Test-Time Scaling',
              link: '/chapter16_reasoning/test-time-scaling'
            },
            {
              text: '16.4 Hybrid Thinking and Budget Control',
              link: '/chapter16_reasoning/hybrid-thinking'
            },
            {
              text: '16.5 Adaptive Thinking',
              link: '/chapter16_reasoning/adaptive-thinking'
            },
            {
              text: '16.6 Reasoning Trace Presentation and Alignment',
              link: '/chapter16_reasoning/cot-visibility-alignment'
            },
            {
              text: '16.7 Summary and Exercises',
              link: '/chapter16_reasoning/review'
            }
          ]
        },
        {
          text: '17. Process Rewards and Inference-Time Search',
          collapsed: false,
          items: [
            {
              text: '17.1 Outcome and Process Rewards',
              link: '/chapter17_prm_search/outcome-vs-process'
            },
            {
              text: '17.2 Discriminative PRMs',
              link: '/chapter17_prm_search/discriminative-prm'
            },
            {
              text: '17.3 Generative PRMs',
              link: '/chapter17_prm_search/generative-prm'
            },
            {
              text: '17.4 Formal Verifiers',
              link: '/chapter17_prm_search/formal-prm'
            },
            {
              text: '17.5 Inference-Time Search',
              link: '/chapter17_prm_search/inference-time-search'
            },
            {
              text: '17.6 Parallel Reasoning and Answer Aggregation',
              link: '/chapter17_prm_search/parallel-reasoning-and-summary'
            },
            {
              text: '17.7 Summary and Exercises',
              link: '/chapter17_prm_search/review'
            }
          ]
        },
        {
          text: '18. Industrial LLM RL Engineering',
          collapsed: false,
          items: [
            {
              text: '18.1 From Single-Machine Experiments to Industrial Training',
              link: '/chapter18_llm_rl_industrial/single-machine-to-industrial'
            },
            {
              text: '18.2 Industrial Post-Training Pipeline',
              link: '/chapter18_llm_rl_industrial/industrial-post-training'
            },
            {
              text: '18.3 Training Stability',
              link: '/chapter18_llm_rl_industrial/modern-industrial-practice'
            },
            {
              text: '18.4 Distributed RL Training',
              link: '/chapter18_llm_rl_industrial/distributed-sync'
            },
            {
              text: '18.5 Data Engineering for Large-Scale RL',
              link: '/chapter18_llm_rl_industrial/data-engineering'
            },
            {
              text: '18.6 Summary and Exercises',
              link: '/chapter18_llm_rl_industrial/review'
            }
          ]
        }
      ]
    },
    {
      text: 'Agentic RL',
      collapsed: false,
      items: [
        {
          text: '19. Agentic RL Systems',
          collapsed: false,
          items: [
            {
              text: '19.1 Agentic RL Foundations',
              link: '/chapter19_agentic/overview'
            },
            {
              text: '19.2 Multi-Turn Reinforcement Learning',
              link: '/chapter19_agentic/formulation'
            },
            {
              text: '19.3 Trajectory Credit Assignment',
              link: '/chapter19_agentic/credit-assignment'
            },
            {
              text: '19.4 Tool Use and Trajectory Generation',
              link: '/chapter19_agentic/tool-use-and-trajectory'
            },
            {
              text: '19.5 Search-Augmented RL',
              link: '/chapter19_agentic/tool-use-agents'
            },
            {
              text: '19.6 Code Interpreter RL',
              link: '/chapter19_agentic/industrial-practice'
            },
            {
              text: '19.7 Multi-Agent Collaboration',
              link: '/chapter19_agentic/multi-agent-swarm'
            },
            {
              text: '19.8 Hands-on: Training a DeepCoder Agent with rLLM',
              link: '/chapter19_agentic/rllm-deepcoder-lab'
            },
            {
              text: '19.9 Hands-on: Training a Financial Analysis Agent with rLLM',
              link: '/chapter19_agentic/rllm-finqa-lab'
            },
            {
              text: '19.10 Hands-on: Building an Agentic RL Training System from Scratch',
              link: '/chapter19_agentic/build-agentic-training-system'
            },
            {
              text: '19.11 Summary and Exercises',
              link: '/chapter19_agentic/review'
            }
          ]
        },
        {
          text: '20. RL for Code Agents',
          collapsed: false,
          items: [
            {
              text: '20.1 SWE-RL Basics',
              link: '/chapter20_swe_agents/swe-bench-and-rlvr'
            },
            {
              text: '20.2 Code World Model and DeepSWE',
              link: '/chapter20_swe_agents/world-model-and-deep-swe'
            },
            {
              text: '20.3 Self-Play SWE-RL',
              link: '/chapter20_swe_agents/self-play-ssr-and-summary'
            },
            {
              text: '20.4 Summary and Exercises',
              link: '/chapter20_swe_agents/review'
            }
          ]
        },
        {
          text: '21. Deep Research and Browser Agents',
          collapsed: false,
          items: [
            {
              text: '21.1 Browser RL Harness',
              link: '/chapter21_deep_research/browser-rl-harness'
            },
            {
              text: '21.2 Evaluation Benchmarks and Open-Source Projects',
              link: '/chapter21_deep_research/deep-research-eval'
            },
            {
              text: '21.3 Summary and Exercises',
              link: '/chapter21_deep_research/review'
            }
          ]
        },
        {
          text: '22. Computer Use and GUI Agents',
          collapsed: false,
          items: [
            {
              text: '22.1 GUI Agent Training',
              link: '/chapter22_computer_use/training'
            },
            {
              text: '22.2 Prompt Injection and Instruction Hierarchy',
              link: '/chapter22_computer_use/safety-swarm'
            },
            {
              text: '22.3 Summary and Exercises',
              link: '/chapter22_computer_use/review'
            }
          ]
        }
      ]
    },
    {
      text: 'Multimodal Reinforcement Learning (Elective)',
      collapsed: false,
      items: [
        {
          text: '23. Vision-Language Model RL (Elective)',
          collapsed: false,
          items: [
            {
              text: '23.1 Visual Reward Design',
              link: '/chapter23_vlm/vlm-challenges'
            },
            {
              text: '23.2 Visual Reflection RL',
              link: '/chapter23_vlm/qwen3-vl-reflection'
            },
            {
              text: '23.3 Hands-on: Multimodal GRPO',
              link: '/chapter23_vlm/vlm-grpo-hands-on'
            },
            {
              text: '23.4 Hands-on: GeoQA Geometric Reasoning',
              link: '/chapter23_vlm/easyr1-geoqa'
            },
            {
              text: '23.5 Summary and Exercises',
              link: '/chapter23_vlm/review'
            }
          ]
        },
        {
          text: '24. Frontier Multimodal RL (Elective)',
          collapsed: false,
          items: [
            {
              text: '24.1 Audio Reward Design',
              link: '/chapter24_multimodal_frontier/audio-reward-design'
            },
            {
              text: '24.2 Multimodal Audio Agents',
              link: '/chapter24_multimodal_frontier/audio-agents'
            },
            {
              text: '24.3 VLA Models',
              link: '/chapter24_multimodal_frontier/embodied-intelligence/'
            },
            {
              text: '24.4 How RL Aligns Image Generation',
              link: '/chapter24_multimodal_frontier/visual-generation-dancegrpo'
            },
            {
              text: '24.5 Why Videos Contradict Themselves',
              link: '/chapter24_multimodal_frontier/video-generation-modern'
            },
            {
              text: '24.6 Summary and Exercises',
              link: '/chapter24_multimodal_frontier/review'
            }
          ]
        }
      ]
    },
    {
      text: 'Safety, Evaluation and Research Frontiers (Elective)',
      collapsed: false,
      items: [
        {
          text: '25. Reward Hacking and RL Evaluation (Elective)',
          collapsed: false,
          items: [
            {
              text: '25.1 Why Higher Reward Can Make the Task Worse',
              link: '/chapter25_alignment_failures/classical-failures'
            },
            {
              text: '25.2 How to Verify RLVR Gains',
              link: '/chapter25_alignment_failures/modern-incidents'
            },
            {
              text: '25.3 Why Models Switch Behavior',
              link: '/chapter25_alignment_failures/sleeper-and-faking'
            },
            {
              text: '25.4 How to Prevent Reward Exploitation',
              link: '/chapter25_alignment_failures/scaling-and-defenses'
            },
            {
              text: '25.5 How to Evaluate RL Models Reliably',
              link: '/chapter25_alignment_failures/rl-evaluation'
            },
            {
              text: '25.6 Summary and Exercises',
              link: '/chapter25_alignment_failures/review'
            }
          ]
        },
        {
          text: '26. Self-Play, Scaling and Research Frontiers (Elective)',
          collapsed: false,
          items: [
            {
              text: '26.1 How Models Generate Training Data',
              link: '/chapter26_selfplay_frontiers/self-play-outlook/'
            },
            {
              text: '26.2 Where Additional Compute Should Go',
              link: '/chapter26_selfplay_frontiers/rl-scaling-outlook'
            },
            {
              text: '26.3 How Multiple LLMs Learn Together',
              link: '/chapter26_selfplay_frontiers/llm-multi-agent-rl/'
            },
            {
              text: '26.4 How LLMs Search for New Algorithms',
              link: '/chapter26_selfplay_frontiers/alphaevolve/'
            },
            {
              text: '26.5 Summary and Exercises',
              link: '/chapter26_selfplay_frontiers/review'
            }
          ]
        }
      ]
    },
    {
      text: 'Appendices',
      collapsed: false,
      items: [
        {
          text: 'A. Training Debugging and Engineering Practice',
          collapsed: false,
          items: [
            {
              text: 'A.1 Training Debugging Guide',
              link: '/appendix_a_engineering/training-debugging'
            },
            {
              text: 'A.2 Training Infrastructure',
              link: '/appendix_a_engineering/rl-infrastructure'
            },
            {
              text: 'A.3 Agent Sandbox',
              link: '/appendix_a_engineering/agentic-rl-infra'
            },
            {
              text: 'A.4 Evaluation Benchmarks',
              link: '/appendix_a_engineering/evaluation-badcase'
            }
          ]
        },
        {
          text: 'B. Core Algorithm Implementations',
          collapsed: false,
          items: [
            {
              text: 'B.1 SFT and KL Divergence',
              link: '/appendix_b_algorithms/sft-kl'
            },
            {
              text: 'B.2 PPO and GAE',
              link: '/appendix_b_algorithms/ppo-gae'
            },
            {
              text: 'B.3 DPO Methods',
              link: '/appendix_b_algorithms/dpo-family'
            },
            {
              text: 'B.4 GRPO and Reward Models',
              link: '/appendix_b_algorithms/grpo-rlvr'
            },
            {
              text: 'B.5 DAPO',
              link: '/appendix_b_algorithms/dapo'
            },
            {
              text: 'B.6 Softmax and Cross-Entropy',
              link: '/appendix_b_algorithms/softmax-ce'
            },
            {
              text: 'B.7 Sampling Methods',
              link: '/appendix_b_algorithms/top-k-top-p'
            },
            {
              text: 'B.8 Attention Mechanism',
              link: '/appendix_b_algorithms/attention-mha'
            }
          ]
        },
        {
          text: 'C. Learning Resources and Reference Materials',
          collapsed: false,
          items: [
            {
              text: 'C.1 Learning Resources and Project Roadmap',
              link: '/appendix_c_resources/learning-resources'
            },
            {
              text: 'C.2 GPU Hours Estimation Table',
              link: '/appendix_c_resources/gpu-hours-estimation'
            },
            {
              text: 'C.3 Metrics Glossary',
              link: '/appendix_c_resources/metrics-glossary'
            },
            {
              text: 'C.4 Industrial Practice Exercises',
              link: '/appendix_c_resources/industrial-exercises'
            }
          ]
        },
        {
          text: 'D. Math Foundations',
          collapsed: false,
          items: [
            {
              text: 'D.1 Linear Algebra',
              collapsed: false,
              items: [
                {
                  text: 'D.1.1 Vectors and Matrices',
                  link: '/appendix_d_math/linear-algebra-basics'
                },
                {
                  text: 'D.1.2 Matrix Form of the Bellman Equation',
                  link: '/appendix_d_math/linear-algebra-bellman'
                },
                {
                  text: 'D.1.3 Dot Products, Norms, and Function Approximation',
                  link: '/appendix_d_math/linear-algebra-function-approx'
                },
                {
                  text: 'D.1.4 Convergence, Eigenvalues, and Trust Regions',
                  link: '/appendix_d_math/linear-algebra-advanced'
                },
                {
                  text: 'D.1.5 Formula Review and Exercises',
                  link: '/appendix_d_math/linear-algebra-formulas-exercises'
                }
              ]
            },
            {
              text: 'D.2 Probability, Expectation, and Random Estimation',
              collapsed: false,
              items: [
                {
                  text: 'D.2.1 Probability, Conditional Probability, and Expectation',
                  link: '/appendix_d_math/probability-basics'
                },
                {
                  text: 'D.2.2 From Random Trajectories to State Values',
                  link: '/appendix_d_math/probability-value'
                },
                {
                  text: 'D.2.3 Monte Carlo, Incremental Averages, and Importance Sampling',
                  link: '/appendix_d_math/probability-sampling'
                },
                {
                  text: 'D.2.4 Trajectory Probability, Baselines, and GAE',
                  link: '/appendix_d_math/probability-trajectory-td'
                },
                {
                  text: 'D.2.5 Bellman Expectation Equation and Action Values',
                  link: '/appendix_d_math/probability-bellman-advanced'
                },
                {
                  text: 'D.2.6 Probability and Statistics Formula Reference and Exercises',
                  link: '/appendix_d_math/probability-formulas-exercises'
                }
              ]
            },
            {
              text: 'D.3 Calculus and Optimization',
              collapsed: false,
              items: [
                {
                  text: 'D.3.1 Derivatives, Gradients, and the Chain Rule',
                  link: '/appendix_d_math/calculus-basics'
                },
                {
                  text: 'D.3.2 Policy Gradients and Advantage Functions',
                  link: '/appendix_d_math/calculus-policy-gradient'
                },
                {
                  text: 'D.3.3 PPO Clipping, Adam, and Update Stability',
                  link: '/appendix_d_math/calculus-ppo'
                },
                {
                  text: 'D.3.4 Policy Gradient, Taylor, and GRPO Derivations',
                  link: '/appendix_d_math/calculus-derivations'
                },
                {
                  text: 'D.3.5 Complete Formulas for PG, DQN, GAE, PPO, and GRPO',
                  link: '/appendix_d_math/calculus-advanced-formulas'
                },
                {
                  text: 'D.3.6 Calculus and Optimization Formula Reference and Exercises',
                  link: '/appendix_d_math/calculus-formulas-exercises'
                }
              ]
            },
            {
              text: 'D.4 Information Theory and Distribution Distance',
              collapsed: false,
              items: [
                {
                  text: 'D.4.1 Self-Information, Entropy, and Exploration',
                  link: '/appendix_d_math/information-basics'
                },
                {
                  text: 'D.4.2 Cross-Entropy and KL Divergence',
                  link: '/appendix_d_math/information-cross-entropy-kl'
                },
                {
                  text: 'D.4.3 Information Theory in PPO, RLHF, and DPO',
                  link: '/appendix_d_math/information-rlhf-dpo'
                },
                {
                  text: 'D.4.4 Mutual Information and Representation Learning',
                  link: '/appendix_d_math/information-mutual-info'
                },
                {
                  text: 'D.4.5 Complete Formulas for KL, RLHF, DPO, and Mutual Information',
                  link: '/appendix_d_math/information-advanced-formulas'
                },
                {
                  text: 'D.4.6 Information Theory Formula Reference and Exercises',
                  link: '/appendix_d_math/information-formulas-exercises'
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

const logger = createLogger()
const originalWarn = logger.warn
logger.warn = (msg, options) => {
  if (msg.includes('Failed to resolve "/@siteData"')) return
  originalWarn(msg, options)
}

export default defineConfig({
  lang: 'en-US',
  title: 'Hands-on Modern RL',
  description:
    'Hands-on modern reinforcement learning: classic control, LLM post-training, RLVR, and multimodal agents',
  base,
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    image: {
      lazyLoading: true
    },
    attrs: {
      disable: true
    },
    config: (md) => {
      safeHeadingAttrs(md)
      optimizedImagesPlugin(md)
      md.use(markdownItFootnote)
      footnoteTitlePlugin(md)
      katexMarkdown(md)
      MermaidMarkdown(md)
      optimizedMermaidPlugin(md)
      // Custom "output" container for displaying code running results
      md.use(markdownItContainer, 'output', {
        render: function (tokens, idx) {
          if (tokens[idx].nesting === 1) {
            const title = tokens[idx].info.trim().slice(6).trim() || 'Output'
            return `<div class="custom-block output"><p class="custom-block-title">${title}</p>\n`
          }
          return '</div>\n'
        }
      })
    }
  },
  vite: {
    customLogger: logger,
    server: {
      watch: {
        ignored: ['**/.vitepress/dist/**']
      }
    },
    plugins: [mermaidConfigPlugin(), normalizeBrokenDocPathPlugin()],
    optimizeDeps: {
      include: [
        '@braintree/sanitize-url',
        'cytoscape',
        'cytoscape-cose-bilkent',
        'dayjs',
        'debug'
      ]
    },
    resolve: {
      alias: {
        'dayjs/plugin/advancedFormat.js': 'dayjs/esm/plugin/advancedFormat',
        'dayjs/plugin/customParseFormat.js':
          'dayjs/esm/plugin/customParseFormat',
        'dayjs/plugin/isoWeek.js': 'dayjs/esm/plugin/isoWeek',
        'cytoscape/dist/cytoscape.umd.js': 'cytoscape/dist/cytoscape.esm.js'
      }
    }
  },
  ignoreDeadLinks: false,
  head: [
    ['link', { rel: 'icon', href: `${base}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#3f51b5' }],
    [
      'meta',
      { name: 'viewport', content: 'width=device-width, initial-scale=1.0' }
    ],
    ['meta', { name: 'author', content: 'WalkingLabs' }],
    ['meta', { name: 'robots', content: 'index,follow' }],
    ['meta', { property: 'og:title', content: 'Hands-on Modern RL' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          'Hands-on modern reinforcement learning: classic control, LLM post-training, RLVR, and multimodal agents'
      }
    ],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:url', content: siteUrl }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }]
  ],
  themeConfig: {
    logo: '/readme/logo-symbol.svg',
    siteTitle: 'Hands on Modern RL',
    nav,
    sidebar,
    socialLinks: [
      { icon: 'github', link: `https://github.com/${owner}/${repo}` }
    ],
    search: enableLocalSearch
      ? {
          provider: 'local',
          options: {
            _render: renderSearchMarkdown
          }
        }
      : undefined,
    editLink: {
      pattern: editLinkPattern,
      text: 'Edit this page on GitHub'
    },
    footer: {
      message: 'Hands-on Modern Reinforcement Learning',
      copyright: 'Based on Hands-On Modern RL by WalkingLabs (CC BY-NC-SA 4.0)'
    },
    outline: {
      level: [2, 3],
      label: 'On this page'
    },
    lastUpdated: {
      text: 'Last updated'
    },
    docFooter: {
      prev: 'Previous page',
      next: 'Next page'
    },
    notFound: {
      title: 'Page not found',
      quote: 'This page does not exist. Try the table of contents instead.',
      link: '/',
      linkText: 'Back to the start',
      linkLabel: 'Back to the start'
    }
  }
})
