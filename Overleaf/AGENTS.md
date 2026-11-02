# Collaborative paper editing

- Always publish this Overleaf folder to `git@github.com:Reminguch/Non-Markovian-Weather.git`, on its `main` branch, with this folder's contents at the repository root.
- Use the dedicated SSH key `~/.ssh/non_markovian_weather` for that repository. Preserve its existing history when syncing. A request to push Overleaf does not authorize pushing the parent Weather_Global repository or its `minimalistic_code` branch.
- The active manuscript is the root `main_v2.tex`, unless the user specifies another file.
- User writing suggestions are comments by default. Record them in blue `\suozhi{...}` annotations near the relevant passage. Do not apply them as prose, title, abstract, equation, or claim changes unless the user explicitly requests that rewrite.
- Preserve the user's original Chinese wording in adjacent LaTeX source comments when the rendered annotation is in English. English annotations keep the existing pdfLaTeX workflow usable.
- Do not attribute assistant-generated suggestions to `suozhi`. Discuss them separately unless the user adopts them.
- Preserve collaborator comments and existing manuscript content. Never mark review comments resolved without an explicit request.
- The user has authorized migration to the official ICLR 2027 template. Keep the official style files unmodified. Formatting and compilation fixes must not silently change scientific claims.
- Read the `write-ml-paper` skill for paper work. Compile and inspect the PDF after LaTeX changes.
