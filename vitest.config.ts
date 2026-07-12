import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    exclude: ['tests/browser/**'],
    testTimeout: 120_000,
    hookTimeout: 120_000,
    // Report-only (`npm run test:coverage`); no thresholds are enforced.
    coverage: {
      provider: 'v8',
      include: ['src/**'],
      reporter: ['text', 'lcov'],
    },
  },
});
