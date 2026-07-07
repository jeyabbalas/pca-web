End-to-end build a production-grade, open-source TypeScript library. Read this whole brief first, ask me any clarifying questions you still have after reading it, then scope the work and execute autonomously. If you ask me clarifying questions, for each question, always provide me with your best expert recommendation.

<context>
The library is a client-side, zero-runtime-dependency reimplementation of scikit-learn's `sklearn.decomposition.PCA` (targeting the current 1.9 API) plus an `IncrementalPCA`-equivalent. The primary target is the web browser, for in-browser dimensionality reduction: data never leaves the user's machine. It must also work in Node.js. The two audiences are (1) app developers doing classic small-to-medium PCA in the browser or Node, and (2) developers running PCA on large datasets — large in both sample count and feature dimension — who need memory-efficient and GPU-accelerated paths.

This will be a max effort, complex task running over a long horizon. Treat sklearn as the ground-truth specification: where behavior is ambiguous, sklearn's actual output is the tiebreaker.
</context>

<goal>
Ship a single ESM package, `pca-web` (will also be the npm package name), that a developer can install and use in both Node and the browser, that numerically matches scikit-learn's PCA across its full API surface, and that provides memory-efficient and WebGPU-accelerated paths for large data. "Matches scikit-learn" is defined concretely below by a parity test suite run against fixtures generated from real scikit-learn.
</goal>

<scope>
Full API parity with `sklearn.decomposition.PCA` (1.9), including:
- `n_components` as int, as float in (0,1) for variance-ratio selection (full solver), as `'mle'` (Minka's MLE), and `None`.
- All `svd_solver` modes: `'full'`, `'covariance_eigh'`, `'randomized'`, `'arpack'`, and `'auto'` (with auto's selection heuristic matching sklearn's shape/component-count logic).
- `whiten`, `copy`, `tol`, `iterated_power`, `n_oversamples`, `power_iteration_normalizer`, `random_state` (seedable, reproducible).
- Methods: `fit`, `transform`, `fit_transform`, `inverse_transform`, `score`, `score_samples` (the Tipping–Bishop probabilistic PCA model), `get_covariance`, `get_precision`, `get_feature_names_out`.
- All fitted attributes with sklearn's exact semantics and sign conventions: `components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `mean_`, `n_components_`, `n_samples_`, `n_features_in_`, `noise_variance_`. Match sklearn's deterministic sign convention (svd_flip) so components are reproducible.

Plus a sibling `IncrementalPCA`-equivalent (matching `sklearn.decomposition.IncrementalPCA`): `partial_fit`, `batch_size`, running mean/variance updates, and the same fitted attributes, so that data never needs to fully reside in memory.

Memory efficiency: support both Float64 and Float32 typed-array storage, operate in place where the `copy` flag allows, and avoid unnecessary matrix copies. The incremental path is the primary answer for very large sample counts.

WebGPU acceleration: accelerate the expensive kernels — Gram/covariance formation (XᵀX / XXᵀX), and the matmuls in the randomized solver's range-finder and power iterations — with a feature-detected WebGPU backend and a correct CPU fallback that produces numerically equivalent results (within floating-point tolerance). Capability detection must degrade gracefully when WebGPU is absent (e.g. Node without a GPU adapter, or an unsupported browser). You will be developing on an M5 Apple Silicon chip.

Distribution: single package, ESM-only, shipped with `.d.ts` types and an `exports` map. Expose the WebGPU backend behind a subpath export (e.g. `pca-web/webgpu`) so the core stays tree-shakeable and GPU code isn't pulled in unless imported. Node ≥ 20 and current evergreen browsers.
</scope>

<correctness_and_verification>
This is the heart of the project, so establish it early, before building out solvers:

1. Generate a numerical reference oracle. The container has network access to PyPI. Install scikit-learn and numpy, and write a Python script that fits sklearn's PCA and IncrementalPCA across a matrix of cases — varied shapes (wide, tall, square; from tiny up to sizes large enough to exercise the large-data paths), ranks, dtypes, every solver, every `n_components` mode, whiten on/off, and known edge cases (n_samples < n_features, constant/zero-variance features, single component, requesting all components). Serialize inputs and all resulting attributes/outputs (`transform`, `inverse_transform`, `score`, `score_samples`, `get_covariance`, `get_precision`) to committed fixture files. This script is part of the deliverable and must be re-runnable.

2. Build a parity test suite that loads those fixtures and asserts the TypeScript implementation matches, at a documented floating-point tolerance appropriate to each dtype, accounting for the legitimate sign ambiguity of singular vectors. Randomized/arpack solvers are approximate — assert against sklearn with tolerances consistent with the approximation, and test determinism under a fixed seed.

3. Test the WebGPU backend against the CPU backend for numerical equivalence. Since this container has no GPU, WebGPU kernels can be written and type-checked here but not executed here. Set up a headless-browser test harness (real browser, real GPU adapter — e.g. Playwright against a WebGPU-capable browser) so the GPU path *can* be run and verified, and be explicit in your reporting about which GPU tests you were actually able to execute versus which remain unverified in this environment. Do not report GPU code as working on the basis of type-checking alone.

Establish a method for checking your own work as you build. Run your parity suite continuously, and use fresh-context verifier subagents to check implementations against this specification at sensible intervals rather than relying on self-review.

Before reporting progress, audit each claim against a tool result from this session. Only report work you can point to evidence for; if something is not yet verified, say so explicitly. Report outcomes faithfully: if tests fail, say so with the output; if a step was skipped, say that; when something is done and verified, state it plainly without hedging.
</correctness_and_verification>

<deliverables>
- The library source, the fixture-generation script, the committed fixtures, and the full test suite (parity + WebGPU-equivalence + headless-browser harness).
- A README documenting the API, the Node and browser usage paths, the memory-efficient and WebGPU paths, solver selection, and the measured parity tolerances.
- A benchmark script comparing CPU vs WebGPU on large inputs and reporting throughput/timing. (Performance is a goal but not a correctness gate — report the numbers you actually measure; don't tune thresholds to pass.)
- Working, typechecked, linted, building code with green tests.
- Git commit your changes in this repository as you work. Do not push anything.
</deliverables>

<boundaries>
Zero runtime dependencies. Dev dependencies (scikit-learn/numpy for fixtures, a test runner, a bundler, Playwright, typecheck/lint tooling) are expected and fine.

Don't add features, refactor, or introduce abstractions beyond what the task requires. A one-shot operation usually doesn't need a helper. Don't design for hypothetical future requirements: do the simplest thing that works well. Avoid premature abstraction and half-finished implementations. Don't add error handling, fallbacks, or validation for scenarios that cannot happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, the public API surface). Don't scaffold sparse-input support, kernel PCA, or other sklearn decomposition classes I haven't asked for.

When you have enough information to act, act. If you're weighing a design choice, give a recommendation and proceed, rather than surveying every option. This doesn't apply to your private reasoning.

You are operating largely autonomously. I am not watching in real time. For reversible actions that follow from this brief, proceed without asking. Pause and end the turn only when the work genuinely requires me: a real scope change, an irreversible or destructive action, or a decision only I can make. If you hit one of those, ask and end the turn rather than ending on a promise of work not yet done. Before ending any turn, check your last paragraph — if it's a plan, a question, or an "I'll now do X," do that work now with tool calls instead. End your turn only when the deliverables above are met and verified, or you're blocked on input only I can provide.

Keep a Markdown memory file for lessons learned across this build: store one lesson per entry with a one-line summary, record both corrections and confirmed approaches and why they mattered, update rather than duplicate, and delete anything that turns out to be wrong.
</boundaries>

<communication>
Lead with the outcome. When you report back, your first sentence should answer "what happened" — the TLDR I'd ask for. Supporting detail comes after. If you've been working a while without me watching, write your summary as a re-grounding for someone who didn't see the working thread: complete sentences, terms spelled out, no arrow-chains or invented shorthand, each file or identifier in its own plain clause. Readability matters more than brevity.
</communication>

Start by scoping the project and asking any remaining clarifying questions. Then build it.
