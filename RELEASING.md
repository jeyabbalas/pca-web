# Releasing

`pca-web` is published to npm via **Trusted Publishing** (OIDC) from GitHub
Actions — there is no long-lived `NPM_TOKEN`. Releases are triggered by pushing a
`vx.y.z` git tag, and the publish step is gated behind the `npm-publish`
GitHub Environment.

## Recurring release flow

1. **Update `CHANGELOG.md`:** move items from `[Unreleased]` into a new
   `## [x.y.z] - YYYY-MM-DD` section, and update the compare links at the bottom
   of the file.
2. **Commit** the changelog on `main`.
3. **Bump + tag:** `npm version <patch|minor|major>` — edits `package.json`,
   creates the `vx.y.z` commit and matching tag.
4. **Push:** `git push --follow-tags`.
5. The **Release** workflow runs the test gate, then pauses on the `npm-publish`
   environment. **Approve** it from the GitHub Actions run.
6. **Confirm:** `npm view pca-web version` shows the new version, and the npm
   package page shows the provenance ("Built and signed on GitHub Actions")
   badge.

The workflow guards against a mistyped tag: if the tag number disagrees with
`package.json`, the publish job fails before publishing anything.

## One-time setup (already done, kept here for reference)

- **First publish was manual** — Trusted Publishing only works for packages that
  already exist on the registry, so `v0.1.0` was published with `npm publish`
  from an authenticated CLI session (no provenance on that first release).
- **GitHub Environment** `npm-publish` — created under
  *Settings → Environments*, with the maintainer as a required reviewer.
- **npm trusted publisher** — configured on the package's npm settings:
  org/user `jeyabbalas`, repository `pca-web`, workflow filename `release.yml`,
  environment `npm-publish`, allowed action `npm publish`.
