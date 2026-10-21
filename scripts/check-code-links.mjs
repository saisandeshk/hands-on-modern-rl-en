import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'

// Every link from the course pages into this repository (GitHub blob/tree URLs,
// or relative links that climb out of docs/) must point at a file that exists.

const root = process.cwd()
const docsRoot = path.join(root, 'docs')
const repoLink =
  /https:\/\/github\.com\/[\w.-]+\/hands-on-modern-rl(?:-en)?\/(?:blob|tree)\/[\w.-]+\/([^)\s"'#?]+)/g
const relativeLink = /\]\(((?:\.\.\/)+(?:code|notebooks|scripts)\/[^)\s#?]+)\)/g

function collectMarkdown(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const absolute = path.join(directory, entry.name)
    if (entry.isDirectory()) {
      if (entry.name === '.vitepress' || entry.name === 'public') return []
      return collectMarkdown(absolute)
    }
    return entry.isFile() && entry.name.endsWith('.md') ? [absolute] : []
  })
}

const missing = []
let checked = 0

for (const file of collectMarkdown(docsRoot)) {
  const source = fs.readFileSync(file, 'utf8')
  const lines = source.split('\n')
  lines.forEach((line, index) => {
    for (const match of line.matchAll(repoLink)) {
      checked += 1
      const target = decodeURIComponent(match[1]).replace(/\/$/, '')
      if (!fs.existsSync(path.join(root, target))) {
        missing.push(`${path.relative(root, file)}:${index + 1} -> ${target}`)
      }
    }
    for (const match of line.matchAll(relativeLink)) {
      checked += 1
      const target = path.resolve(path.dirname(file), match[1])
      if (!fs.existsSync(target)) {
        missing.push(
          `${path.relative(root, file)}:${index + 1} -> ${path.relative(root, target)}`
        )
      }
    }
  })
}

if (missing.length > 0) {
  console.error(`Broken repository links (${missing.length} of ${checked}):`)
  for (const entry of missing) console.error(`  ${entry}`)
  process.exit(1)
}

console.log(`Repository links OK (${checked} checked).`)
