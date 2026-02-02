# Copilot Instructions for this Hugo Blog

This repository is a personal blog built with Hugo and hosted on GitHub Pages. Follow these rules when making changes.

## Project Context
- Static site built with Hugo.
- Content lives in Markdown files under `_posts/` and possibly other content directories.
- Static assets live in `data/`, `notebooks/`, and other folders as needed.

## General Rules
- Prefer minimal, targeted edits.
- Preserve existing front matter keys and formatting.
- Do not reformat unrelated content.
- Keep links relative when appropriate.
- Avoid introducing new dependencies unless requested.

## Content Editing
- Blog posts are Markdown files in `_posts/`.
- Preserve YAML/TOML front matter and its field order.
- Maintain the author’s voice; avoid rewriting unless requested.
- For dates, use ISO format: `YYYY-MM-DD`.

## Hugo Conventions
- Do not rename files or change slugs unless explicitly asked.
- If adding new content, match the naming convention in `_posts/`.
- Keep shortcodes intact.

## Images and Assets
- Place new images in the appropriate existing folder if one is specified; otherwise, use `static/` if it exists.
- Use relative paths in Markdown.

## Safety
- Do not modify `.github/` or CI unless explicitly requested.
- Do not delete content unless asked.

## When Uncertain
- Search the workspace for existing patterns and follow them.
- Ask a brief clarification only if required to proceed.
