/**
 * Write section numbers back into Markdown from the single English VitePress
 * sidebar (docs/.vitepress/config.mjs), keeping the two in sync.
 *
 * Each numbered sidebar entry (`text: '1.1 Title'` + `link: '...'`, including
 * appendix numbers such as `A.1` or `D.1.1`) is compared against the linked
 * page's front-matter `title:` and first `# H1`:
 *
 * - front-matter mismatch: replace the old number prefix in the `title:` line
 *   with the sidebar's number.
 * - front-matter missing: prepend the sidebar's number to the `title:` value.
 * - H1 mismatch: replace the old section number in the body with the
 *   sidebar's, and update every heading line starting with the old number
 *   (e.g. `### 1.3.1` → `### 1.1.1`), so sub-headings don't keep the old
 *   number.
 * - H1 missing: rewrite the body's first H1 as `# X.Y` + the sidebar title
 *   (matching the text after the number in `text`).
 *
 * Usage:
 *   npm run fix:headings
 *   node scripts/fix-heading-numbers.mjs --dry-run   # check mode, writes nothing
 *   node scripts/fix-heading-numbers.mjs --check      # same as --dry-run
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '..')
const configPath = path.join(repoRoot, 'docs', '.vitepress', 'config.mjs')
const docsRoot = path.join(repoRoot, 'docs')

const dryRun =
  process.argv.includes('--dry-run') || process.argv.includes('--check')

// Numbered entries of the single English sidebar, e.g.
//   text: '1.1 Run CartPole', / text: 'A.1 Training Debugging Guide',
//   text: 'D.1.1 Vectors and Matrices' — each followed by its link line.
const pairRe =
  /text:\s*'((?:[A-Z]\.)?\d+(?:\.\d+)*)\s+([^']+)'\s*,\s*\r?\n\s*link:\s*'([^']+)'/g

const numRe = /^(?:[A-Z]\.)?\d+(?:\.\d+)*\b/
const h1NumRe = /^#\s*((?:[A-Z]\.)?\d+(?:\.\d+)*)\b/
const h1LineRe = /^#\s+/

const src = fs.readFileSync(configPath, 'utf8')

function resolveDocMd(link) {
  const clean = link.replace(/^\//, '')
  if (clean.endsWith('/')) {
    return path.join(docsRoot, clean.replace(/\/$/, ''), 'index.md')
  }
  return path.join(docsRoot, `${clean}.md`)
}

function stripFrontmatterLines(lines) {
  if (lines[0] !== '---') return lines
  const end = lines.indexOf('---', 1)
  if (end === -1) return lines
  return lines.slice(end + 1)
}

function frontmatterSplit(lines) {
  if (lines[0] !== '---') {
    return { before: [], body: lines, afterFmLine: 0 }
  }
  const end = lines.indexOf('---', 1)
  if (end === -1) {
    return { before: lines, body: [], afterFmLine: lines.length }
  }
  return {
    before: lines.slice(0, end + 1),
    body: lines.slice(end + 1),
    afterFmLine: end + 1
  }
}

function frontmatterTitleInfo(beforeLines) {
  const index = beforeLines.findIndex((line) => /^\s*title:\s*/.test(line))
  if (index === -1) return { line: null, index: -1, number: null }
  const raw = beforeLines[index].replace(/^\s*title:\s*/, '').trim()
  const unquoted =
    raw.length >= 2 &&
    ((raw.startsWith("'") && raw.endsWith("'")) ||
      (raw.startsWith('"') && raw.endsWith('"')))
      ? raw.slice(1, -1).trim()
      : raw
  const match = unquoted.match(numRe)
  return {
    line: beforeLines[index],
    index,
    number: match ? match[0] : null,
    quoted: unquoted !== raw
  }
}

function applyFrontmatterFix(beforeLines, info, newNum) {
  const next = beforeLines.slice()
  const prefix = info.line.slice(0, info.line.indexOf('title:') + 6)
  let rest = info.line.slice(info.line.indexOf('title:') + 6).trim()
  let quote = ''
  if (
    rest.length >= 2 &&
    ((rest.startsWith("'") && rest.endsWith("'")) ||
      (rest.startsWith('"') && rest.endsWith('"')))
  ) {
    quote = rest[0]
    rest = rest.slice(1, -1).trim()
  }
  rest = rest.replace(numRe, '').trim()
  const value = rest ? `${newNum} ${rest}` : newNum
  next[info.index] = `${prefix} ${quote}${value}${quote}`
  return next
}

function firstNumberedH1InBody(bodyLines) {
  const idx = bodyLines.findIndex((line) => h1NumRe.test(line))
  if (idx === -1) return { line: null, index: -1 }
  return { line: bodyLines[idx], index: idx }
}

function firstH1InBody(bodyLines) {
  const idx = bodyLines.findIndex((line) => h1LineRe.test(line))
  if (idx === -1) return { line: null, index: -1 }
  return { line: bodyLines[idx], index: idx }
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function remapHeadingLine(line, oldNum, newNum) {
  const re = new RegExp(`^(#{1,6}\\s+)${escapeRegExp(oldNum)}(?=\\.|\\s|$)`)
  return line.replace(re, `$1${newNum}`)
}

function applyMismatchFix(bodyLines, oldNum, newNum) {
  return bodyLines.map((line) => remapHeadingLine(line, oldNum, newNum))
}

function applyMissingFix(bodyLines, newNum, sidebarTitleRest) {
  const h1 = firstH1InBody(bodyLines)
  const newLine = `# ${newNum} ${sidebarTitleRest.trim()}`
  if (h1.index === -1) {
    if (
      bodyLines.length === 0 ||
      (bodyLines.length === 1 && bodyLines[0] === '')
    ) {
      return [newLine, '']
    }
    return [newLine, ...bodyLines]
  }
  const next = bodyLines.slice()
  next[h1.index] = newLine
  return next
}

function joinWithEol(lines, eol) {
  if (lines.length === 0) return ''
  return lines.join(eol) + (lines[lines.length - 1] === '' ? '' : eol)
}

const entries = []
let m
while ((m = pairRe.exec(src)) !== null) {
  entries.push({
    number: m[1],
    titleRest: m[2],
    link: m[3].replace(/^\//, '')
  })
}

let changedFiles = 0
let totalEdits = 0

for (const ent of entries) {
  const mdPath = resolveDocMd(ent.link)
  if (!fs.existsSync(mdPath)) {
    console.warn(`[skip] file not found: ${path.relative(repoRoot, mdPath)}`)
    continue
  }

  const raw = fs.readFileSync(mdPath, 'utf8')
  const eol = raw.includes('\r\n') ? '\r\n' : '\n'
  const lines = raw.split(/\r?\n/)
  const { before, body } = frontmatterSplit(lines)

  let newBefore = null
  let fmReason = ''
  const fm = frontmatterTitleInfo(before)
  if (fm.line) {
    if (fm.number !== ent.number) {
      newBefore = applyFrontmatterFix(before, fm, ent.number)
      fmReason =
        fm.number == null
          ? `front-matter title missing number, set to ${ent.number}`
          : `front-matter title ${fm.number} → ${ent.number}`
    }
  }

  let newBody = null
  let bodyReason = ''
  const numH1 = firstNumberedH1InBody(body)
  if (numH1.line) {
    const got = numH1.line.match(h1NumRe)[1]
    if (got !== ent.number) {
      newBody = applyMismatchFix(body, got, ent.number)
      bodyReason = `H1 mismatch ${got} → ${ent.number}`
    }
  } else {
    newBody = applyMissingFix(body, ent.number, ent.titleRest)
    bodyReason = `missing # X.Y, set to ${ent.number}`
  }

  if (!newBefore && !newBody) continue

  const mergedBefore = newBefore || before
  const mergedBody = newBody || body
  const newRaw = joinWithEol(
    mergedBefore.length ? [...mergedBefore, ...mergedBody] : mergedBody,
    eol
  )

  if (newRaw !== raw) {
    changedFiles += 1
    totalEdits += 1
    const rel = path.relative(repoRoot, mdPath)
    const reason = [fmReason, bodyReason].filter(Boolean).join('; ')
    console.log(`${dryRun ? '[dry-run] ' : ''}${reason}: ${rel}`)
    if (!dryRun) {
      fs.writeFileSync(mdPath, newRaw, 'utf8')
    }
  }
}

console.log(
  `\n${dryRun ? '(not written) ' : 'wrote '} ${changedFiles} file(s) (sidebar entries with numbers: ${entries.length})`
)

// dry-run only prints the plan and exits 0, for local preview
