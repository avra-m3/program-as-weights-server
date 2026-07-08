# specs/

Each file here is a natural-language spec passed to `paw.compile(...)`.

Convention: one `.txt` file per task, named after the task, e.g.
`sentiment.txt`, `json-fixer.txt`.

A good spec has a short description plus a handful of `Input: ... Output: ...`
examples pulled from your real data. For example:

```
Classify user intent. Return ONLY one of: search, create, delete, other.

Input: Find the latest report
Output: search

Input: Make a new folder
Output: create

Input: Remove old backups
Output: delete
```

Tips (from the PAW docs):

- State output constraints explicitly ("Return ONLY one of: X, Y, Z").
- Include examples from your actual data — they outperform prose alone.
- Iterate: compile, run against `data/<task>.yaml`, look at specific
  failures, tweak wording, recompile.

No spec files are checked in yet — add one once a task is chosen.
