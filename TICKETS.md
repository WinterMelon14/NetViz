# NetViz Tickets

This backlog turns the compatibility, structured-input, visualization, and profiling roadmap into implementation-ready work.

## Guiding principles

- Report static-analysis results honestly. A preflight can identify likely blockers, but it cannot guarantee that symbolic tracing will succeed.
- Keep ordinary tracing separate from profiling. Profiling performs repeated execution and has different latency, side-effect, synchronization, and reproducibility concerns.
- Version request and trace schemas deliberately. Do not silently reinterpret older payloads.
- Bound all user-controlled structures, tensor allocations, source inputs, diagnostics, and serialized trace data.
- Preserve the existing trusted-code model: static inspection does not make user code safe to execute.
- Prefer explicit status and capability fields over inferred frontend behavior.

## Status vocabulary

Compatibility findings use exactly these statuses:

- `supported`: NetViz can configure and execute the item with the current runtime.
- `configuration_required`: NetViz can support the item after the user supplies required information.
- `unsupported`: The current runtime deliberately cannot represent or execute the item.
- `unknown`: Static inspection cannot determine compatibility reliably.

## Milestones

### R1 — Compatibility and diagnostics

Produce an honest compatibility report, support structured forward arguments, group the resulting trace by module, and expose suspicious runtime values.

### R2 — Project compatibility

Support predictable execution of models spanning multiple local modules and explicit resources.

### R3 — CPU profiling

Add a separate, repeatable CPU profiling workflow with aggregate timing and critical-path analysis.

### Later

CUDA profiling, peak CUDA memory, FLOP estimates, `torch.compile` comparison, and richer large-graph navigation.

---

# R1 — Compatibility and diagnostics

## NV-001 — Define and version the compatibility-report schema

**Goal:** Establish a stable contract shared by static inspection, the desktop bridge, and the frontend.

**Work:**

- Add a versioned compatibility-report type to Python and TypeScript.
- Model findings for the class, constructor, `forward()`, each input, imports, resources, runtime constraints, and likely FX blockers.
- Give every finding one of the four canonical statuses.
- Include a machine-readable code, short title, explanation, evidence, and remediation where applicable.
- Distinguish facts derived from source from conservative heuristics.
- Reject malformed or unsupported report versions at the bridge boundary.

**Acceptance criteria:**

- Python and TypeScript agree on one documented fixture.
- Every finding has a status and stable code.
- The schema can represent unresolved shape and dtype independently.
- Unknown findings are not presented as failures.
- Contract tests cover valid, malformed, and unsupported-version payloads.

**Dependencies:** None.

---

## NV-002 — Inspect constructor and forward signatures comprehensively

**Goal:** Report how each callable parameter can be configured before user code executes.

**Work:**

- Inspect positional-only, positional-or-keyword, keyword-only, optional/defaulted, `*args`, and `**kwargs` parameters.
- Record annotations and statically resolvable defaults without evaluating source.
- Report whether optional parameters may be omitted.
- Identify unsupported or ambiguous signatures explicitly.
- Preserve declared parameter order.

**Acceptance criteria:**

- Fixtures cover every Python parameter category and combinations of them.
- Each parameter receives an independent compatibility status.
- Optional parameters with safe literal defaults are represented as omittable.
- Variadic parameters are never silently treated as ordinary positional inputs.
- No source import or execution occurs during inspection.

**Dependencies:** NV-001.

---

## NV-003 — Report shapes, dtypes, imports, resources, and runtime constraints

**Goal:** Make preflight output useful beyond callable signatures.

**Work:**

- Reuse and extend current AST input suggestions.
- Report known, unresolved, and user-configurable dimensions and dtypes separately.
- Inventory local, standard-library, and external imports conservatively.
- Detect obvious referenced paths and resource-loading calls without claiming completeness.
- Report effective constraints such as CPU execution, evaluation mode, allocation limits, timeout, and supported dtypes.
- Include evidence locations where source line information is available.

**Acceptance criteria:**

- A report can show a partially known tensor shape.
- Local and external imports are distinguished when statically resolvable.
- Runtime constraints come from shared constants rather than duplicated UI strings.
- Resource findings are labeled as likely/unknown unless certainty is justified.
- Tests cover aliased imports and common checkpoint/config loading patterns.

**Dependencies:** NV-001, NV-002.

---

## NV-004 — Detect likely dynamic-control-flow and FX blockers

**Goal:** Warn about constructs that commonly prevent or distort symbolic tracing.

**Work:**

- Detect likely tensor-dependent branches, iteration over tensor-derived values, data extraction such as `.item()`, and other known FX hazards.
- Detect dynamic module selection and mutation where feasible.
- Separate definite unsupported constructs from heuristic warnings.
- Attach source line evidence.
- State explicitly that a clean report is not a tracing guarantee.

**Acceptance criteria:**

- Known blocker fixtures produce stable finding codes.
- Heuristic findings use `unknown` or `configuration_required`, not false certainty.
- Findings include actionable explanations.
- False-positive-prone checks are documented and tested.
- The report includes a global “symbolic tracing outcome unknown until execution” statement.

**Dependencies:** NV-001.

---

## NV-005 — Build the compatibility preflight UI

**Goal:** Present inspection results before the trusted-code execution confirmation.

**Work:**

- Add a compatibility summary with counts by status.
- Show sections for signatures, inputs, imports/resources, runtime constraints, and FX findings.
- Link findings to the relevant configuration control when possible.
- Keep unknown findings visually distinct from unsupported findings.
- Prevent execution only for blockers that are definitively unsupported or missing required configuration.
- Make the report keyboard accessible.

**Acceptance criteria:**

- Users can identify exactly why the run button is disabled.
- Unknown warnings do not disable execution.
- Unsupported and configuration-required states have distinct remediation.
- Long reports remain usable without hiding critical blockers.
- Component tests cover all statuses and mixed reports.

**Dependencies:** NV-001, NV-002, NV-003, NV-004.

---

## NV-006 — Design structured input schema v2

**Goal:** Represent positional and keyword arguments plus bounded nested values deliberately.

**Work:**

- Add a new request schema version rather than extending the existing tensor-list payload ambiguously.
- Represent positional arguments and string-keyed keyword arguments.
- Support tensors, finite numeric scalars, booleans, `None`, strings where explicitly permitted, tuples, lists, and string-keyed dictionaries.
- Define omission separately from an explicit `None`.
- Define tensor dtype, shape, generator, bounds, and other generator-specific configuration.
- Specify maximum nesting depth, container size, string length, tensor count, per-tensor allocation, and total allocation.
- Decide and document how non-finite floating-point scalar values are handled.

**Acceptance criteria:**

- The schema distinguishes omitted, `None`, positional, and keyword values.
- Tuple and list semantics survive serialization.
- Dictionary keys are strings and duplicate-key ambiguity is impossible.
- All recursive structures have enforceable limits.
- Schema v1 remains explicitly supported or fails with a clear version error.
- Python and TypeScript share passing and failing fixtures.

**Dependencies:** NV-001.

---

## NV-007 — Validate and construct structured inputs in the worker

**Goal:** Safely convert schema v2 values into `(*args, **kwargs)`.

**Work:**

- Validate the complete request before allocating tensors.
- Recursively construct supported bounded values.
- Support float and integer tensors suitable for masks, token IDs, and indices.
- Apply user-confirmed dtype, shape, and generation strategy.
- Enforce aggregate allocation limits across nested inputs.
- Invoke the model as `model(*args, **kwargs)`.
- Return path-aware validation errors such as `kwargs.attention_mask.shape[1]`.

**Acceptance criteria:**

- Tests cover every supported leaf and container type.
- Optional arguments can be omitted.
- Keyword-only arguments execute correctly.
- Nested invalid values identify their exact request path.
- Allocation limits cannot be bypassed through multiple nested tensors.
- No partially constructed request reaches model execution after validation failure.

**Dependencies:** NV-006.

---

## NV-008 — Extend or replace the example-input provider contract

**Goal:** Let trusted model-provided examples use the same structured calling convention.

**Work:**

- Define a provider return contract that can express positional and keyword arguments.
- Validate provider output with the same recursive limits as schema v2.
- Preserve a compatibility path for the current tuple/list-of-tensors provider if desired.
- Surface provider contract errors separately from model execution errors.

**Acceptance criteria:**

- Providers can return both `args` and `kwargs`.
- Provider values cannot bypass input limits.
- Legacy provider behavior is documented and tested if retained.
- Error messages identify whether provider validation or model execution failed.

**Dependencies:** NV-006, NV-007.

---

## NV-009 — Build the structured input editor

**Goal:** Configure schema v2 without requiring users to edit JSON.

**Work:**

- Choose positional versus keyword placement from the inspected signature.
- Add editors for supported leaf values.
- Add bounded tuple, list, and dictionary editors.
- Allow optional arguments to be omitted or explicitly set to `None`.
- Support integer masks/tensors, dtype selection, dimensions, and generation strategy.
- Show total estimated allocation across the full input tree.
- Preserve AST-generated suggestions as editable starting points.

**Acceptance criteria:**

- The UI cannot create structurally invalid schema v2 payloads.
- Omitted and explicit-`None` states are visibly different.
- Validation errors point to the relevant editor.
- Aggregate allocation updates as nested values change.
- Keyboard-only configuration is possible for representative fixtures.

**Dependencies:** NV-002, NV-005, NV-006.

---

## NV-010 — Add module hierarchy metadata to trace schema

**Goal:** Preserve enough scope information to group both modules and traced function operations.

**Work:**

- Audit current `module.path` coverage.
- Serialize additional FX scope/stack metadata where function and method operations lack a useful module path.
- Define stable hierarchy identifiers and display labels.
- Handle reused modules without duplicating or mis-parenting nodes.
- Version the trace schema if the contract changes incompatibly.

**Acceptance criteria:**

- Custom nested modules produce a navigable hierarchy.
- Function operations inside custom modules are assigned to the best available scope.
- Reused modules are identified consistently.
- Missing scope metadata has a documented root/fallback group.
- Trace fixtures cover nested, reused, sequential, and functional operations.

**Dependencies:** None.

---

## NV-011 — Implement collapsible module grouping

**Goal:** Make large graphs understandable through progressive disclosure.

**Work:**

- Build a hierarchy from trace scope metadata.
- Collapse a group into a summary node with correct boundary edges.
- Expand groups without losing selection or viewport unnecessarily.
- Summarize node count, parameter count, activation memory, and suspicious-value counts.
- Define behavior for search matches inside collapsed groups.

**Acceptance criteria:**

- Collapsing never changes graph connectivity semantically.
- Boundary edges do not multiply incorrectly.
- Selected descendants remain discoverable after collapse.
- Group summaries update from their descendants.
- Large fixture graphs remain interactive at an agreed performance threshold.

**Dependencies:** NV-010.

---

## NV-012 — Add graph search, filters, and path isolation

**Goal:** Let users locate and focus on relevant operations quickly.

**Work:**

- Search by node label, operation, target, module path, shape, and dtype.
- Filter by operation category, dtype, memory range, and suspicious values.
- Isolate upstream, downstream, or complete dependency paths from a selected node.
- Reveal matching descendants inside collapsed groups.
- Add clear/reset behavior and result counts.

**Acceptance criteria:**

- Search and filters compose predictably.
- Path isolation preserves all connecting nodes and edges required to explain the path.
- Hidden matches are surfaced through their collapsed ancestor.
- Zero-result states explain active constraints.
- Search remains responsive on large fixtures.

**Dependencies:** NV-011.

---

## NV-013 — Add diagnostic color modes and suspicious-value indicators

**Goal:** Turn existing trace summaries into visible diagnostic signals.

**Work:**

- Add color modes for operation category, dtype, parameter memory, and activation memory.
- Add NaN and Inf badges.
- Add sparse-output indicators using a documented threshold or scale.
- Provide legends with accessible labels and non-color cues.
- Define behavior when a metric is absent.
- Aggregate child diagnostics into collapsed module summaries.

**Acceptance criteria:**

- Memory coloring uses a documented scale that handles outliers.
- NaN, Inf, and sparsity remain identifiable without relying on color alone.
- Missing metrics are visually distinct from zero.
- Collapsed groups expose descendant warning counts.
- Tests cover absent, zero, extreme, and invalid numeric values.

**Dependencies:** NV-011.

---

## NV-014 — R1 integration, migration, and end-to-end fixtures

**Goal:** Ship the first cohesive release without silently breaking existing traces.

**Work:**

- Add end-to-end fixtures covering positional/keyword inputs, nested values, masks, optional omission, module grouping, and suspicious outputs.
- Define request and trace schema migration behavior.
- Add a representative-model compatibility matrix.
- Document limitations and trusted-code behavior.
- Exercise cancellation, timeout, large-result transport, and source-change checks through the new flow.

**Acceptance criteria:**

- One end-to-end test produces a compatibility report, configures structured inputs, runs a trace, groups modules, and displays suspicious values.
- Existing supported v1 workflows have a documented outcome.
- Unsupported older payloads fail with actionable version errors.
- Release notes describe supported value types and known FX limitations.

**Dependencies:** NV-005, NV-007, NV-008, NV-009, NV-011, NV-013.

---

# R2 — Project compatibility

## NV-015 — Define project-root and resource request contracts

**Goal:** Represent the model’s execution context explicitly.

**Work:**

- Add an explicit project root/source root concept.
- Define allowed local module resolution.
- Add explicit checkpoint and configuration resource descriptors.
- Define working-directory behavior.
- Hash or otherwise identify inspected source and declared resources for change detection.
- Avoid exposing absolute paths unnecessarily to the frontend.

**Acceptance criteria:**

- The runtime never guesses a project root silently.
- The UI can explain the effective import root and working directory.
- Declared resources are validated before execution.
- Changed or missing resources produce targeted errors.
- The security implications of project access are documented.

**Dependencies:** NV-006.

---

## NV-016 — Execute multi-module projects predictably

**Goal:** Support models whose source spans local modules.

**Work:**

- Replace or extend temporary single-file sanitization with a project-aware execution strategy.
- Configure `sys.path` and working directory explicitly in the worker.
- Preserve source identity checks for the entry file and declared local modules.
- Handle relative imports consistently.
- Keep diagnostic paths redacted or display-name mapped at the bridge boundary.

**Acceptance criteria:**

- Fixtures cover sibling imports, package-relative imports, and nested packages.
- Import resolution does not depend on the process’s launch directory.
- Project files outside the declared root are not treated as local project modules.
- Source changes between inspection and execution are detected.
- Failure diagnostics use user-facing paths without leaking unrelated temporary paths.

**Dependencies:** NV-015.

---

## NV-017 — Add project and resource configuration UI

**Goal:** Let users configure project context explicitly and understand its effects.

**Work:**

- Add project-root selection.
- Show discovered local imports and unresolved external dependencies.
- Add explicit checkpoint/config resource selection.
- Show effective working directory and runtime access summary.
- Require reconfirmation when inspected project content or resource identities change.

**Acceptance criteria:**

- Required missing resources block execution with remediation.
- Users can distinguish project files, external dependencies, and declared resources.
- Changes invalidate stale compatibility reports.
- The trusted-code confirmation describes the project scope being executed.

**Dependencies:** NV-003, NV-005, NV-015, NV-016.

---

# R3 — CPU profiling

## NV-018 — Define profiling request and result schemas

**Goal:** Keep profiling semantics separate from ordinary traces.

**Work:**

- Add an explicit profiling mode and schema version.
- Configure warmup count, measurement count, and safe upper limits.
- Record environment and run metadata needed to interpret results.
- Represent per-node samples and aggregate median/percentile statistics.
- Define how failed or partially observed operations are reported.
- Document that profiling executes the model repeatedly and may consume randomness or trigger side effects.

**Acceptance criteria:**

- A normal trace request cannot accidentally enable repeated execution.
- Warmup and measurement limits are enforced.
- Results preserve sample count and aggregation method.
- Python and TypeScript contract fixtures agree.
- The UI can explain the execution count before confirmation.

**Dependencies:** NV-014.

---

## NV-019 — Instrument CPU duration per node

**Goal:** Collect useful per-operation CPU timing with controlled overhead.

**Work:**

- Add profiling instrumentation to the interpreter/runtime.
- Exclude warmup samples from aggregates.
- Use a monotonic high-resolution clock.
- Preserve node identity across repetitions.
- Record instrumentation and aggregation behavior.
- Handle exceptions and cancellation during warmup or measurement.

**Acceptance criteria:**

- Each observed node reports sample count, median, and configured percentiles.
- Warmup runs are not included.
- Cancellation stops further repetitions promptly.
- Tests use tolerant assertions and a controllable clock where practical.
- Ordinary trace mode performs no profiling repetitions.

**Dependencies:** NV-018.

---

## NV-020 — Add expensive-operations table and timing visualization

**Goal:** Make profiling results actionable.

**Work:**

- Add a sortable table for aggregate duration, self duration where available, percentile, call count, and module path.
- Add a timing color mode to the graph.
- Link table rows and graph selection bidirectionally.
- Aggregate timing into collapsed module groups.
- Clearly label wall-clock, inclusive, and self-time semantics.

**Acceptance criteria:**

- Sorting is stable and handles missing metrics.
- Selecting a table row reveals its graph node or collapsed ancestor.
- Group timing aggregation does not double-count under its documented semantics.
- Timing legends expose units and scale.
- Profiling results are not shown as ordinary trace measurements.

**Dependencies:** NV-011, NV-019.

---

## NV-021 — Compute and visualize a timing-based critical path

**Goal:** Highlight the dependency path with the greatest measured cost.

**Work:**

- Define the critical-path algorithm over the trace DAG.
- Choose and document the timing weight used.
- Handle disconnected components and missing timing values.
- Expose the total path cost and ordered operations.
- Allow users to isolate the path in the graph.

**Acceptance criteria:**

- Deterministic fixtures produce the expected path.
- Missing timing values do not create misleading zero-cost shortcuts.
- Disconnected graphs report their selected/maximum component clearly.
- The path can be inspected through the existing isolation interaction.

**Dependencies:** NV-012, NV-019.

---

## NV-022 — R3 profiling validation and documentation

**Goal:** Establish trustworthy expectations for CPU profiling.

**Work:**

- Add end-to-end profiling fixtures.
- Measure instrumentation overhead on representative small and medium models.
- Document sources of noise, thread settings, warmups, randomness, and side effects.
- Add reproducibility guidance.
- Confirm cancellation, timeout, and transport limits under repeated execution.

**Acceptance criteria:**

- End-to-end profiling returns aggregate timing and an expensive-operations view.
- Documentation avoids implying benchmark-grade precision.
- Profiling errors are distinguishable from ordinary trace errors.
- A performance budget guards against accidental extreme overhead.

**Dependencies:** NV-020, NV-021.

---

# Later tickets

## NV-023 — Minimap and large-graph navigation

Begin after module grouping defines the graph’s collapse/expand interaction model.

## NV-024 — CUDA event timing and synchronization-aware profiling

Use CUDA events and explicit synchronization rules; do not derive CUDA duration from CPU wall-clock timing.

## NV-025 — Peak CUDA memory reporting

Define allocation/reset boundaries carefully and distinguish allocated from reserved memory.

## NV-026 — FLOP estimates with confidence metadata

Report only supported operations and distinguish reliable estimates from unknown or partial totals.

## NV-027 — `torch.compile` comparison mode

Compare eager and compiled execution with explicit compilation warmup and graph-break reporting.

## NV-028 — Trace and profile comparison

Compare two runs by stable node/module identity and show timing, memory, shape, and graph-structure deltas.

---

# Recommended implementation order

1. NV-001 → NV-002 → NV-003/NV-004 → NV-005
2. NV-006 → NV-007 → NV-008/NV-009
3. NV-010 → NV-011 → NV-012/NV-013
4. NV-014
5. NV-015 → NV-016 → NV-017
6. NV-018 → NV-019 → NV-020/NV-021 → NV-022

NV-003 and NV-004 can proceed in parallel after the base report schema. NV-008 and NV-009 can proceed in parallel once schema v2 and worker construction are stable. NV-012 and NV-013 can proceed in parallel after grouping works.

# Definition of done for every ticket

- Behavior is covered at the narrowest practical unit level.
- Cross-language schema changes have shared fixtures or contract tests.
- User-facing failures have stable codes and actionable messages.
- New user-controlled data has explicit limits and boundary validation.
- Accessibility is considered for new interactive UI.
- Documentation and known limitations are updated.
- Existing cancellation, timeout, source-identity, and transport protections continue to work.
