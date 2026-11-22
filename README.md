# Support Ticket Classification & Routing API

A Python ML service that predicts the team for a customer support ticket and sends low-confidence predictions to manual review. Built with **pandas, scikit-learn, FastAPI, and Docker**.

```text
subject + description → TF-IDF → logistic regression → confidence threshold
                                                    ├─ auto_route → team
                                                    └─ manual_review → no assigned team
```

This is a reproducible portfolio demo. The current dataset covers **Account Access, Billing, Order Management, and Refunds**. Technical Support is outside its label set. Strong results on this generated dataset do not establish reliability on real customer tickets.

## Run locally

Use Python **3.14** to match the evaluated environment. Run commands from the project root.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m ticket_router.train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn ticket_router.api:app --host 127.0.0.1 --port 8000
```

macOS / Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ticket_router.train
python -m pytest -q
python -m uvicorn ticket_router.api:app --host 127.0.0.1 --port 8000
```

The first training run downloads the pinned public CSV (about 19 MB). Later runs verify its SHA-256 and use the local copy. Training saves `artifacts/model.joblib` and evaluation files in `reports/`. The raw data and binary model are excluded from Git; a fresh clone must train before serving or building Docker. Only load trusted joblib artifacts because deserialization executes Python code.

Open [interactive API docs](http://localhost:8000/docs). The service loads the model once during startup and fails with a training instruction if it is absent. `MODEL_PATH` can point to another trusted artifact.

## API examples

`POST /predict` requires string fields `subject` and `description`. One may be empty; both cannot be blank. Subject is limited to 500 characters and description to 10,000. Missing fields, wrong types, extra fields, and invalid text return HTTP 422. Clients cannot override the routing threshold.

PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/health
$ticket = @{ subject = 'Payment issue'; description = 'I cannot complete my payment' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body $ticket
```

curl (bash):

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"subject":"Payment issue","description":"I cannot complete my payment"}'
```

Observed response from the saved model (confidence rounded here):

```json
{
  "predicted_category": "Billing",
  "confidence": 0.991870,
  "routing_decision": "auto_route",
  "assigned_team": "Billing"
}
```

Manual-review example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"subject":"","description":"My bank says it paid for the order but your checkout says it failed."}'
```

```json
{
  "predicted_category": "Account Access",
  "confidence": 0.415410,
  "routing_decision": "manual_review",
  "assigned_team": null
}
```

The predicted category remains available to a reviewer; it is not assigned when routing is uncertain. Manual review is a returned status, with no queue or external ticket-system integration. Confidence is the largest logistic-regression probability, not a calibrated guarantee of correctness.

`GET /health` returns:

```json
{"status":"ok","model_loaded":true,"threshold":0.5}
```

## Dataset and preparation

Source: [Bitext customer-support training dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset), attributed to **Bitext**, pinned to commit `72ea2203180d14a416b579c657033ecd21e59a11`. The dataset is generated with a hybrid NLP/NLG methodology. See [the upstream description](https://github.com/bitext/customer-support-llm-chatbot-training-dataset/blob/72ea2203180d14a416b579c657033ecd21e59a11/README.md) and the preserved [CDLA-Sharing-1.0 data license](data/LICENSE.txt). This license applies to the source data; this project does not redistribute the raw CSV.

CSV SHA-256: `6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434`.

The source has `instruction` and `category`, without a separate subject. We use **an empty subject and the instruction as description**. We never fabricate a subject from the category or use the answer/intent as model input. This keeps label information out of the ticket text. At inference, subject and description are combined using the same normalization.

| Source label | API team | Unique tickets | Train | Validation | Test |
|---|---|---:|---:|---:|---:|
| ACCOUNT | Account Access | 5,358 | 3,750 | 804 | 804 |
| PAYMENT | Billing | 1,986 | 1,390 | 298 | 298 |
| ORDER | Order Management | 3,067 | 2,147 | 460 | 460 |
| REFUND | Refunds | 2,577 | 1,804 | 386 | 387 |
| **Total** | | **12,988** | **9,091** | **1,948** | **1,949** |

Preparation selects 14,964 of 26,872 source rows. It normalizes Unicode and whitespace, removes blank text, discards text with contradictory labels, and removes duplicates using a case-insensitive text key **before splitting**. This run removed 1,976 duplicates, zero blanks, and zero conflicting rows. A stratified 70/15/15 split uses seed 42; counts differ slightly from exact percentages due to integer rounding. `reports/split_manifest.json` contains text hashes to audit split membership without exporting ticket text.

## Model and routing policy

The baseline always predicts the most frequent training label, Account Access. The trained pipeline uses TF-IDF word unigrams/bigrams (`min_df=2`, sublinear term frequency) and logistic regression (`C=2`, `max_iter=1000`). The entire pipeline is fitted on the training split only. There is no refit on validation or test tickets.

Validation compares thresholds from 0.50 to 1.00. The 0.50 floor requires majority probability. Among candidates with at least **98% validation auto-routing accuracy**, selection maximizes coverage, breaking ties by accuracy and then the higher threshold. If no candidate qualifies, threshold 1.00 disables automatic routing. Otherwise confidence at or above the chosen threshold routes automatically. This target is an empirical selection rule, not a statistical accuracy guarantee.

| Validation threshold | Auto-routing coverage | Auto-routing accuracy |
|---:|---:|---:|
| **0.50 (chosen)** | **99.69%** | **99.90%** |
| 0.60 | 99.54% | 99.95% |
| 0.70 | 99.18% | 100.00% |
| 0.80 | 98.15% | 100.00% |
| 0.90 | 94.76% | 100.00% |
| 0.95 | 84.91% | 100.00% |
| 0.99 | 36.70% | 100.00% |
| 1.00 (disabled) | 0.00% | undefined |

All candidate results are in [validation_thresholds.csv](reports/validation_thresholds.csv). You can change the target explicitly and retrain:

```bash
python -m ticket_router.train --target-accuracy 0.999
```

That replaces the model and reports; the tables here describe the default run.

## Actual held-out results

Measured on the 1,949 test tickets after selecting the threshold using validation only:

| Classifier | Macro F1 | Accuracy |
|---|---:|---:|
| Most-frequent baseline | 0.1460 | 41.25% |
| TF-IDF + logistic regression | 1.0000 | 100.00% |

At threshold **0.50**, test **auto-routing coverage is 100.00% (1,949/1,949)** and **auto-routing accuracy is 100.00% (1,949/1,949)**. Coverage means auto-routed tickets divided by all tickets; routing accuracy means correct auto-routes divided by auto-routed tickets. Accuracy is undefined when coverage is zero, represented as `null` in JSON.

Confusion matrix, with actual labels in rows and predicted labels in columns:

| Actual / predicted | Account Access | Billing | Order Management | Refunds |
|---|---:|---:|---:|---:|
| Account Access | 804 | 0 | 0 | 0 |
| Billing | 0 | 298 | 0 | 0 |
| Order Management | 0 | 0 | 460 | 0 |
| Refunds | 0 | 0 | 0 | 387 |

Full metrics, per-class precision/recall/F1, library versions, counts, and error examples are in [metrics.json](reports/metrics.json); the matrix is also saved as [CSV](reports/confusion_matrix.csv).

### Mistakes and limitations

There were **no classifier errors in this particular test split**. A real **validation** mistake was `assistance informing of an error with onlinepayment`: Billing was predicted as Account Access with confidence **0.6919**, and it was automatically routed at the chosen threshold. A plausible cause is word-based TF-IDF failing to associate the concatenated token `onlinepayment` with `payment`, leaving less specific words to drive the prediction. This is an interpretation of the error, not a proven causal explanation. Another validation ticket, `I get an error when I try to check out`, was misclassified as Refunds at **0.3792** and correctly sent to manual review.

A separate, author-written probe, `I was charged twice for my subscription.`, predicted Account Access at **0.7268** instead of the intended Billing team. This probe is **not part of the held-out metrics**. It demonstrates that confidence-based routing can still make confident mistakes on different wording.

The generated dataset contains closely related language templates. Removing exact normalized duplicates does not remove all paraphrase/template overlap, so the perfect random-split test score can overstate generalization. The selected labels do not cover technical problems, shipping-address changes, or every billing issue. The model has no reliable out-of-domain detector, and an unsupported request can still be assigned confidently. Before real deployment, evaluate on independently collected, human-labeled tickets, use a template/group or time-based split, assess probability calibration, and choose the routing policy against actual misrouting costs. Request text is not logged by this application.

## Docker

Train first, then build an image that includes the resulting artifact:

```bash
docker build -t support-ticket-router .
docker run --rm -p 8000:8000 --name support-ticket-router support-ticket-router
```

The image uses Python 3.14, runs as a non-root user, and includes a health check for `/health`. It serves the saved model without downloading data or retraining at startup. Docker was unavailable in the development environment, so container build/runtime verification remains outstanding. The Python API and saved artifact were exercised directly.

## Project files and verification

```text
ticket_router/
  data.py       # pinned download, text normalization, deduplication, splits
  train.py      # baseline, pipeline, validation threshold, held-out evaluation
  routing.py    # artifact loading and routing decision
  api.py        # request validation, startup loading, /predict and /health
tests/          # offline fitted fixtures and focused ML/API tests
data/LICENSE.txt
artifacts/      # locally generated model, excluded from Git
reports/        # actual evaluation outputs and split hashes
Dockerfile
.github/workflows/ci.yml
```

`python -m pytest -q` passed **21 tests** covering valid requests, blank/invalid input, automatic and manual routing, exact threshold boundaries, artifact startup failure, duplicate/conflict removal, disjoint stratified splits, and threshold selection. CI runs these tests without downloading the dataset. A separate smoke test loaded the full trained model through FastAPI and checked health, Billing routing, and manual review. The installed Starlette release emits an `httpx` deprecation warning in tests; it does not affect the passing results.

A GitHub repository shares the source code and runs CI. Hosting the running FastAPI service requires a separate application host; GitHub Pages cannot run this Python API.
