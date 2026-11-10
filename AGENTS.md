# Working on Shivam Mehta’s website

## Project
- Personal AI and speech research blog at https://shivammehta.me.
- Jekyll with the gem-based Chirpy theme; GitHub Pages deploys through GitHub Actions.
- `_posts/`: articles; `_tabs/`: navigation pages; `posts/`: article media; `_data/`: theme data; `_config.yml`: site settings.
- Preserve published post URLs, front matter, media, identity, and existing content unless the requested change requires edits.

## Theme and comments
- Follow https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Upgrade-Guide for upgrades.
- The `chirpy` remote tracks the official Starter repository. Merge a stable release on a separate branch, resolve customizations carefully, and update Gemfile.lock.
- Prefer theme extension points over copying entire layouts. Local `_data` files override the gem; review them during upgrades.
- Giscus comments use GitHub Discussions in this repository. Preserve repository/category IDs and pathname mapping so existing discussions remain attached to posts. Never expose tokens.
- Keep the canonical URL https://shivammehta.me and an empty baseurl. Do not change DNS, email, domain, or GitHub permissions as part of visual work.

## Development and validation
- Use Ruby 3.4, then `bundle install`.
- Preview with `bundle exec jekyll serve`; production build: `JEKYLL_ENV=production bundle exec jekyll build`.
- Post-relative images use `media_subpath: /posts/<published-slug>/` in front matter with Chirpy 7.
- One historical machinelearning.ru PDF has an explicit link-check exception because its HTTPS certificate fails validation. Keep exceptions narrow.
- Run `bundle exec htmlproofer _site --disable-external` and investigate failures. Distinguish pre-existing content issues from regressions.
- Inspect the homepage, a representative article, navigation/search, mobile layout, light/dark modes, and Giscus configuration for theme changes.
- Do not edit generated `_site/` or commit local dependency caches. Exclude agent notes from the public site.
- Keep the GitHub Actions build/test commands aligned with local validation.

## Collaboration
- This site is being gradually revamped. Make focused, reviewable changes and explain material design choices.
- Prepare and test changes locally. Ask before pushing, merging, deploying, or making consequential external changes. A push to main publishes the site.
