# prompts-readme.md

How to use `prompts.md`, and why the prompts are shaped the way they are. The design draws on two sources: Anthropic's prompting best practices and context-engineering guidance, and Andrej Karpathy's four behavioral rails for agentic coding. This file explains the mapping so you can adapt the prompts instead of copying them blindly.

## The two playbooks behind these files

Anthropic's prompting best practices supply the structure of each prompt: be clear and direct, add context and motivation, use examples, structure with XML tags, give the model a role, and specify the output format. Anthropic's context-engineering guidance supplies the discipline behind CLAUDE.md: treat context as a finite resource and aim for the smallest set of high-signal tokens that gets the outcome, rather than stuffing everything in. Karpathy's rails supply the behavioral constraints that stop a capable agent from doing the wrong helpful thing: think before coding, keep it simple, make surgical changes, and work toward verifiable success criteria.

CLAUDE.md and prompts.md play different roles. CLAUDE.md is the standing constraint layer that loads every session and interrupts default behavior. prompts.md is the per-task instruction layer that tells the agent what to build now and how to know it succeeded. Keep durable rules in CLAUDE.md and keep task specifics in the prompts, so neither bloats the other.

## Anatomy of a prompt in this repo

Every build prompt uses the same skeleton, and each tag earns its place.

The `<role>` tag sets behavior and tone in one or two sentences, which Anthropic notes measurably focuses output. The `<context>` block gives the motivation and the background, because explaining why a thing matters lets the model generalize instead of pattern-matching a generic solution. The `<task>` block is a numbered, sequential list, which Anthropic recommends when order and completeness matter. The `<constraints>` block states limits as things to do rather than only things to avoid, since positive instructions steer more reliably than negative ones. The `<success_criteria>` block is the most important part: it gives the agent a verifiable target to loop toward, which is exactly Karpathy's fourth rail and the single biggest lever on output quality. The `<output_format>` block controls what comes back so sessions stay legible and you can inspect intermediate work.

XML tags are used throughout because they let the model separate instructions from context from inputs without ambiguity, which is Anthropic's recommended way to structure anything beyond a trivial prompt.

## How the prompts enforce Karpathy's four rails

Think before coding shows up as the standing instruction to state assumptions and ask one clarifying question, and as output-format lines like "state your plan before coding." Keep it simple shows up as explicit anti-over-engineering constraints: smallest implementation, no speculative abstractions, no unrequested features. Make surgical changes shows up as scope fences in the constraints ("do not alter Phase 1 code beyond adding the new class") and in the review checklist ("did it touch anything outside the task scope, if so revert"). Work toward success criteria shows up as the dedicated success-criteria block in every prompt plus the rule that the agent does not declare done until those criteria pass when run.

One Anthropic detail worth keeping: with current models, avoid piling on words like CRITICAL and MUST. These newer models follow normal instructions well and can overreact to aggressive phrasing. The prompts use firm, plain language for that reason.

## Running this with Claude Code

Run the phases in order. Phase 0 is deliberately a setup-only session, following Anthropic's advice to use the first context window to build scaffolding and a test harness rather than features. After that, treat one phase as roughly one session.

Commit between phases. Git is the state layer: it gives the agent a clean starting point and gives you a checkpoint to roll back to, and Claude Code is good at reconstructing state from the filesystem and git log at the start of a fresh session. Start each new session from a fresh context rather than a long compacted one, point the agent at PLAN.md and the last results, and let it rediscover state from disk.

Use the reusable review prompt after each phase before committing. The draft, then review, then refine loop is Anthropic's most reliable chaining pattern, and running the review as if a different reviewer wrote the code catches the issues the author would rationalize. Use the reusable debugging prompt whenever something fails, since it forces root-cause analysis over symptom muting.

Let tests be the source of truth. Ask the agent to write tests as part of each phase and treat them as fixed: it is not acceptable to weaken or delete a test to make a run pass, because that hides the very bug the test exists to catch.

## Adapting the prompts

Steal the shape, not the literal text. The skeleton (role, context, task, constraints, success criteria, output format) transfers to any task. When you add a phase, write its success criteria first, because if you cannot state how you will verify the result, the agent cannot either. When the same correction comes up across sessions, move it out of the prompt and into CLAUDE.md so it becomes a standing rail rather than a thing you retype.

## Sources

- Anthropic, Prompting best practices (clarity, context, examples, XML structure, roles, output control, chaining, anti-over-engineering, tool and thinking guidance).
- Anthropic, Effective context engineering for AI agents (context as a finite resource, smallest high-signal token set).
- Andrej Karpathy's observations on agentic-coding failure modes, distilled into four rails: think before coding and state assumptions, keep implementations simple, make surgical changes, and steer with verifiable success criteria.
