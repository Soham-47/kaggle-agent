# Competition adapters

Create a competition with:

```bash
kaggle-agent init --competition demo --slug demo-contest
```

Edit the generated YAML contract and implement the pipeline files under
`competitions/demo/pipeline/`. The generic scaffold uses `id` and `target`
placeholders only. Replace them with the verified sample-submission contract
for the selected competition.

Competition-specific image or tabular adapters may live alongside the
competition workspace. Shared runtime modules must obtain identifiers, labels,
metrics, and paths from `CompetitionConfig`; they must not encode a contest as
the framework default.
