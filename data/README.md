# data/

Each file here is a YAML list of test cases for one task, consumed by
`paw_re_server.data.load_test_cases`.

Convention: one `.yaml` file per task, matching the spec name, e.g.
`data/sentiment.yaml` for `specs/sentiment.txt`.

Schema — a list of objects with `input` and `expected` (string, exact match),
plus an optional `name` label used in test output:

```yaml
- name: happy path
  input: "I love this!"
  expected: "positive"

- name: negative
  input: "This is terrible."
  expected: "negative"

- input: "It's fine I guess."
  expected: "neutral"
```

No data files are checked in yet — add one once a task is chosen.
