#!/bin/sh
set -eu

echo "Checking formatting..."
npm run format:check

echo "Checking lint..."
npm run lint

echo "Checking English publication quality..."
npm run verify:english

echo "Checking repository code links..."
npm run verify:code-links

echo "Building site..."
npm run build

echo "Checking build output..."
test -f docs/.vitepress/dist/preface/introduction.html
test -f docs/.vitepress/dist/sitemap.xml
test -f docs/.vitepress/dist/robots.txt

echo "Verification passed"
