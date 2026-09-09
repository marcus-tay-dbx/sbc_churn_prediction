# Databricks notebook source
# MAGIC %md
# MAGIC # 12 - Genie Code Best Practices
# MAGIC
# MAGIC Genie Code writes code, queries, and configuration for you inside Databricks. 
# MAGIC
# MAGIC The habits below fall into these areas — adopt them gradually:
# MAGIC
# MAGIC | Area | Topic |
# MAGIC | --- | --- |
# MAGIC | **1. Context** | 1.1 Write business semantics in Unity Catalog |
# MAGIC | | 1.2 Start the chat in the right editor |
# MAGIC | | 1.3 Leverage Genie Agents |
# MAGIC | **2. Collaboration** | 2.1 Share Genie Code sessions |
# MAGIC | **3. Customization** | 3.1 Leverage skills |
# MAGIC | | 3.2 Leverage MCP |
# MAGIC | | 3.3 Add proper instructions |
# MAGIC | **4. Memory** | 4.1 Maintain project memory |
# MAGIC | | 4.2 Capture session memory |
# MAGIC | **5. Working with Genie Code** | Review before commit · Tell, don't surprise · Use a Git branch |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Context
# MAGIC
# MAGIC **What is the context for Genie Code?** 
# MAGIC - File structure and naming,
# MAGIC - Documentation
# MAGIC - Unity Catalog metadata: table
# MAGIC descriptions, column comments etc
# MAGIC - and many more
# MAGIC
# MAGIC These does more to improve Genie Code's results than prompting
# MAGIC technique alone.
# MAGIC
# MAGIC The mindset shift: **manage your contextual information as part of daily work** —  Context is something
# MAGIC you build up and keep current, not something you assemble at the moment you ask.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="max-width:1000px;margin:0 auto;font-family:sans-serif;color:#0b2026;">
# MAGIC   <div style="font-size:16pt;font-weight:700;margin-bottom:4px;">The everyday workflow</div>
# MAGIC   <div style="font-size:12pt;color:#5E7077;margin-bottom:16px;">An ongoing habit, plus two moves you make together inside a Genie Code session.</div>
# MAGIC   <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:stretch;">
# MAGIC     <div style="flex:0 0 230px;background:#F9F7F4;border-radius:10px;box-shadow:0 2px 8px rgba(27,49,57,0.08);padding:16px 18px;position:relative;overflow:hidden;">
# MAGIC       <div style="position:absolute;top:0;left:0;width:100%;height:8px;background:#1B5162;"></div>
# MAGIC       <div style="display:inline-block;background:#1B5162;color:#fff;font-size:9.5pt;font-weight:700;padding:3px 11px;border-radius:999px;margin-bottom:10px;">ongoing</div>
# MAGIC       <div style="font-size:13pt;font-weight:700;margin-bottom:6px;">Manage UC business semantics</div>
# MAGIC       <div style="font-size:11pt;color:#5E7077;">Keep table &amp; column comments current.</div>
# MAGIC     </div>
# MAGIC     <div style="align-self:center;font-size:22pt;color:#9BB0B6;">&#8594;</div>
# MAGIC     <div style="flex:1 1 470px;border:2px dashed #B7C4C8;border-radius:12px;padding:12px 14px 14px;background:#FCFBFA;">
# MAGIC       <div style="display:inline-block;background:#0b2026;color:#fff;font-size:9.5pt;font-weight:700;padding:3px 11px;border-radius:999px;margin-bottom:12px;">In a Genie Code session</div>
# MAGIC       <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:stretch;">
# MAGIC         <div style="flex:1 1 195px;background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(27,49,57,0.08);padding:14px 16px;position:relative;overflow:hidden;">
# MAGIC           <div style="position:absolute;top:0;left:0;width:100%;height:8px;background:#2F7C8C;"></div>
# MAGIC           <div style="font-size:12.5pt;font-weight:700;margin-bottom:6px;">Start the chat in the right editor</div>
# MAGIC           <div style="font-size:11pt;color:#5E7077;">Notebook &middot; SQL &middot; Git &middot; bundle.</div>
# MAGIC         </div>
# MAGIC         <div style="align-self:center;font-size:20pt;color:#9BB0B6;">&#8594;</div>
# MAGIC         <div style="flex:1 1 195px;background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(27,49,57,0.08);padding:14px 16px;position:relative;overflow:hidden;">
# MAGIC           <div style="position:absolute;top:0;left:0;width:100%;height:8px;background:#FF5F46;"></div>
# MAGIC           <div style="font-size:12.5pt;font-weight:700;margin-bottom:6px;">Write a good prompt</div>
# MAGIC           <div style="font-size:11pt;color:#5E7077;">Name the related assets.</div>
# MAGIC         </div>
# MAGIC       </div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <div style="margin-top:16px;text-align:center;">
# MAGIC     <span style="display:inline-block;background:#2E7D5B;color:#fff;font-size:12pt;font-weight:700;padding:8px 18px;border-radius:999px;">&#10003; Reliable Genie Code output</span>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Write business semantics in Unity Catalog
# MAGIC
# MAGIC Column comments, table descriptions, and governance tags are read automatically as context, so
# MAGIC well-annotated metadata removes the need for exploratory queries.
# MAGIC
# MAGIC - Write a clear description on each table and a comment on each column.
# MAGIC - Use the **AI generate** option to draft the descriptions and comments, then review them.
# MAGIC
# MAGIC
# MAGIC 🎯 Example — comment `tier_rank` as *"customer segment: 0 = Mass, 1 = Mass Affluent, 2 = Affluent, 3 = Private"* so Genie Code reads the meaning instead of guessing.
# MAGIC
# MAGIC 🗺️  In the UI: open **Catalog** (left sidebar) → select the table → edit the **Comment** on the table and on each column.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Start the chat in the right editor
# MAGIC
# MAGIC The editor you start the chat in determines which tools Genie Code can access:
# MAGIC
# MAGIC | Editor | Genie Code has access to | Best for |
# MAGIC | --- | --- | --- |
# MAGIC | Notebook | Cell execution, iterative debugging, DataFrame inspection, library installation | Data exploration, pipeline development, prototyping |
# MAGIC | SQL Editor | SQL warehouse execution, query results, schema browsing | SQL development, query optimization, DDL |
# MAGIC | Git Folder | File creation/editing, git operations (commit, push, branch) | Project scaffolding, multi-file changes, documentation |
# MAGIC | Pipeline Editor | Pipeline-specific tools, dataset inspection, run history | SDP development, pipeline debugging |
# MAGIC | Dashboard | Widget creation, dataset binding, layout tools | Dashboard building and editing |
# MAGIC | Jobs Page | Job configuration, task editing, run history | Job setup, scheduling, debugging failures |
# MAGIC | Apps Page | App scaffolding, deployment, permission management | App development and deployment |
# MAGIC | Bundle Editor | Full DAB topology, resource definitions, variable interpolation, target management, dependency graph awareness | DAB configuration, resource wiring, multi-target setup, architectural decisions |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Leverage Genie Agents
# MAGIC
# MAGIC A Genie Agent answers questions in natural language over a curated set of tables; Genie Code can
# MAGIC call it as a tool while it works, instead of writing SQL by hand.
# MAGIC
# MAGIC - Create a Genie Agent over the tables you want to expose.
# MAGIC - Add **instructions** and **example queries** so it answers consistently.
# MAGIC - 🎯 Example (churn) — instruction: *"`churned` = 1 means the customer has left; tiers 2–3 are high-value; 'at-risk' means a large 60-day outflow"*; example query: *"which Private-tier customers had the largest 60-day outflow last quarter?"*
# MAGIC
# MAGIC 🗺️ In the UI: open **Genie** (left sidebar) → **New** → choose tables → add instructions and example queries.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Collaboration
# MAGIC
# MAGIC ### 2.1 Share Genie Code sessions
# MAGIC **When to Share**
# MAGIC - You've solved a complex problem and want to show the approach
# MAGIC - You're pairing asynchronously — share the session for a colleague to continue
# MAGIC - You want to document a demo flow that others can reproduce
# MAGIC - You've built something and want stakeholder review of the process
# MAGIC
# MAGIC **How to Share**
# MAGIC - Share a read-only view of a conversation; viewers read along but cannot send messages.
# MAGIC - 🗺️ In the UI: the **Share** button in the thread header at the top of the assistant panel.
# MAGIC
# MAGIC ![image_1788793176471.png](./image_1788793176471.png "image_1788793176471.png")
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Customization
# MAGIC
# MAGIC These extend and configure Genie Code, and most live in **Assistant Settings** (the **gear icon** in
# MAGIC the Genie Code panel).
# MAGIC
# MAGIC ![image_1788791830532.png](./image_1788791830532.png "image_1788791830532.png")
# MAGIC
# MAGIC ### 3.1 Leverage skills
# MAGIC - Reusable domain-knowledge notes that Genie Code loads automatically — naming rules, a deployment checklist, a house coding style.
# MAGIC - 🎯 Example — For the notebook/job/table/dasbhoard/model built for the project `customer_churn_model`, tag every asset `sbc` / `churn-prediction`. 
# MAGIC
# MAGIC ### 3.2 Leverage MCP
# MAGIC - Connectors that integrate external services — Slack, GitHub, Google Drive, Jira.
# MAGIC - 🎯 Example — connect **GitHub** so Genie Code can open the churn repo and reuse the `04-Model-Training` code.
# MAGIC
# MAGIC ### 3.3 Add proper instructions
# MAGIC - Personal preferences applied across all your work — e.g., comment style, coding conventions, platform gotchas.
# MAGIC - 🎯 Example — "for imbalanced targets like churn, always report PR-AUC and class-1 recall rather than accuracy" — so every churn model you build is judged the same way.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Memory
# MAGIC
# MAGIC Knowledge you keep as notes in your project's **Workspace** folder so sessions don't start cold.
# MAGIC
# MAGIC ### 4.1 Maintain project memory
# MAGIC
# MAGIC **What is it** — a first-read notes file (`PROJECT_MEMORY.md`) in your project folder capturing the project's identity, architecture, targets and resource IDs, conventions, and open questions.
# MAGIC
# MAGIC **Why it matters** — it eliminates cold starts: a new session reads it and is immediately up to speed instead of re-discovering the project.
# MAGIC
# MAGIC **Template**
# MAGIC ```
# MAGIC # PROJECT_MEMORY.md — [Project Name]
# MAGIC > Last updated: YYYY-MM-DD
# MAGIC > Owner: email@databricks.com
# MAGIC
# MAGIC ## Project Identity
# MAGIC [What is this? Who is it for? What does it do?]
# MAGIC
# MAGIC ## Architecture
# MAGIC [Key decisions, patterns, dependencies]
# MAGIC
# MAGIC ## Targets & Environments
# MAGIC [dev/staging/prod mappings, resource IDs]
# MAGIC
# MAGIC ## Conventions
# MAGIC [Project-specific naming, patterns]
# MAGIC
# MAGIC ## Open Questions
# MAGIC [Things still being decided]
# MAGIC ```
# MAGIC
# MAGIC 🎯 Example (churn) — Project Identity: bank-churn prediction workshop · Architecture: feature-store model + Lakebase online store + serving endpoint · Targets: `solution_builder.sbc_churn_prediction`, model `customer_churn_model` (`dev`/`champion`), endpoint `customer_churn_endpoint`, synced table `customer_churn_features_synced` · Conventions: PR-AUC primary, tag `sbc`/`churn-prediction` · Open Questions: probability gauge deferred.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Capture session memory
# MAGIC
# MAGIC **What is it** — a short, dated summary of each work session, kept in a `sessions` subfolder and named `YYYY-MM-DD_short-description.md`, with an `INDEX.md` listing them newest-first.
# MAGIC
# MAGIC **Why it matters** — the next session reads the latest summary and continues with full continuity; decisions and dead-ends aren't lost between days.
# MAGIC
# MAGIC **Template**
# MAGIC ```
# MAGIC # Session Summary: [Short Description]
# MAGIC **Date:** YYYY-MM-DD
# MAGIC **Branch:** <branch-name>
# MAGIC
# MAGIC ## Problems Encountered
# MAGIC - [What went wrong or was unclear]
# MAGIC
# MAGIC ## Root Causes
# MAGIC - [Why it happened]
# MAGIC
# MAGIC ## Decisions
# MAGIC - [Architectural choices and rationale]
# MAGIC
# MAGIC ## Changes Made
# MAGIC - path/to/file — [what changed and why]
# MAGIC ```
# MAGIC
# MAGIC 🎯 Example (churn) — Date 2026-09-07 · Decisions: set the drift-gate threshold to 0.2 · Changes Made: `06-Real-Time-Inference` deployed `customer_churn_endpoint` v3; `07-Observability` enabled Lakehouse monitoring.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Working with Genie Code: Interaction Patterns
# MAGIC
# MAGIC How you drive Genie Code during a session — stay in control of what it changes.
# MAGIC
# MAGIC - **Review before commit** — Genie Code shows proposed changes for your approval before finalizing; only trivial single-line fixes go straight in.
# MAGIC - **Tell, don't surprise** — ask to see the plan before a large refactor, and capture what changed in a session summary (see 4.2).
# MAGIC - **Work on a Git branch** — keep changes on a feature branch; never let Genie Code edit directly on the `main` / production branch.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where this shows up in this workshop
# MAGIC - Notebooks `01-Overview` and `02-EDA-with-Genie-Code` introduce Genie Code and use it to explore data.
# MAGIC - The Retention Cockpit app's setup step — asking Genie Code, in plain English, to give the app access to the serving endpoint — is the Context principle in action.
# MAGIC - Full reference: `genieCodeWorkshop/docs/conventions/genie-code-best-practices.md`.