# Label Provenance

Experiment: `notification_triage_real_v1`

Gold labels are stored in `data/notification/gold_labels.jsonl` and split into:

- `profile_train`: 120 labels
- `patch_validation`: 80 labels
- `heldout_test`: 160 labels
- `adversarial_test`: 40 labels

The labels are synthetic benchmark labels authored as fixed fixture data for this repository. They were not generated from model predictions, not copied from the weak/strong model outputs, and not revised during the v7 experiment run.

`GoldLabel.generated_from_predictions` defaults to `false` when the field is absent in JSONL. In the current dataset, all 400 records omit the field and therefore deserialize to `generated_from_predictions=false`. The evaluator rejects any label with `generated_from_predictions=true`, which is covered by `gold_labels_must_not_be_generated_from_predictions`.

Current limitation: this run does not include a separate human-audit or strong-judge provenance artifact for the gold labels. The evidence here is repository fixture provenance plus evaluator enforcement against prediction-derived labels.
