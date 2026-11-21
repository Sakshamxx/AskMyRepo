# GCP setup (Cloud Run)

Every command used to put AskMyRepo on Google Cloud Run. Replace the placeholders; nothing here contains a real key.

```bash
export PROJECT_ID=your-project-id        # the project to deploy into
export REGION=us-central1
export SERVICE=askmyrepo
```

Cloud Run needs an **amd64** image. A Mac (Apple silicon) builds arm64, so the image is built on **Cloud Build** with `--source` instead of locally.

## 1. Log in and pick the project

```bash
gcloud --version
gcloud auth login                              # only if no account is listed
gcloud auth list                               # which account is active
gcloud config list                             # current account, project, region
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION
gcloud billing projects describe $PROJECT_ID   # billingEnabled must be true
```

## 2. Enable the APIs (once per project)

```bash
gcloud services enable \
  run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com artifactregistry.googleapis.com
gcloud services list --enabled | grep -E 'run|cloudbuild|secretmanager|artifactregistry'
```

## 3. Store the keys in Secret Manager

Keys are read from `.env` and piped in, so they never appear on screen or in the image.

```bash
for k in GEMINI_API_KEY PINECONE_API_KEY; do
  name=$(echo $k | tr 'A-Z_' 'a-z-')            # gemini-api-key, pinecone-api-key
  grep -E "^$k=" .env | head -1 | cut -d= -f2- | tr -d '\n' | \
    gcloud secrets create $name --data-file=- --replication-policy=automatic
done
gcloud secrets list
```

To change a key later: `... | gcloud secrets versions add gemini-api-key --data-file=-`.

Let Cloud Run's service account read them:

```bash
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor --condition=None
```

## 4. Prepare the build folder

The cloud image bakes in the models and ShopLite's demo database, so it needs a small staging folder:

```bash
mkdir -p gcp/data
cp -r askmyrepo index app.py corpus.toml pyproject.toml uv.lock README.md gcp/
cp Dockerfile.cloud gcp/Dockerfile
cp ../shoplite/instance/shoplite.db gcp/data/shoplite.db     # seed ShopLite first
```

`index/` (the saved keyword index) comes from running `python -m askmyrepo index`.

## 5. Build and deploy

```bash
cd gcp
gcloud run deploy $SERVICE --source . \
  --project $PROJECT_ID --region $REGION \
  --port 8501 --memory 16Gi --cpu 4 --cpu-boost --no-cpu-throttling \
  --timeout 600 --concurrency 1 \
  --min-instances 1 --max-instances 2 \
  --session-affinity --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest,PINECONE_API_KEY=pinecone-api-key:latest \
  --quiet
```

| Flag | Why |
|---|---|
| `--source .` | Builds the image on Cloud Build (amd64); takes about 15–20 minutes because of PyTorch and the 2.4 GB of models. |
| `--memory 16Gi --cpu 4` | Both models are loaded in memory; with 8 GiB the instance was killed ("Memory limit exceeded"). |
| `--concurrency 1` | Each question is heavy (reranking and several model calls); one at a time per instance. |
| `--min-instances 1 --no-cpu-throttling` | One instance stays warm with its models loaded (billed continuously). Set `--min-instances 0` to save money. |
| `--cpu-boost` | Faster cold start. `serve.py` also loads the models in the background at startup. |
| `--timeout 600` | An agent answer can take minutes. |
| `--session-affinity` | Streamlit uses a websocket; keep a browser session on one instance. |
| `--max-instances 2` | Caps the cost. |
| `--allow-unauthenticated` | Public URL. Remove it (see below) to require sign-in. |

## 6. Check it

```bash
URL=$(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')
echo $URL
curl -s $URL/_stcore/health                    # prints: ok
gcloud run services describe $SERVICE --region $REGION
gcloud builds list --region $REGION --limit 3  # build status
gcloud run services logs read $SERVICE --region $REGION --limit 50
```

## 7. Update, lock down, delete

```bash
# Redeploy after a code change: repeat steps 4 and 5.

# Make it private (needs a Google sign-in; open it with the proxy)
gcloud run services update $SERVICE --region $REGION --no-allow-unauthenticated
gcloud run services proxy $SERVICE --region $REGION        # then open http://localhost:8080

# Keep one instance warm (no cold start, but billed continuously)
gcloud run services update $SERVICE --region $REGION --min-instances 1

# Delete everything this created
gcloud run services delete $SERVICE --region $REGION
gcloud secrets delete gemini-api-key
gcloud secrets delete pinecone-api-key
```

## 8. Auto-deploy from GitHub (redeploy on every push to `main`)

`cloudbuild.yaml` builds `Dockerfile.cloud`, pushes the image to Artifact Registry and deploys it to Cloud Run. The repo is self-contained for this (`index/` and `data/shoplite.db` are committed), so the build needs nothing from your laptop.

```bash
# a) Let the build service account push images and deploy (once)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
SA=$PROJECT_NUMBER-compute@developer.gserviceaccount.com
for role in roles/run.admin roles/iam.serviceAccountUser roles/artifactregistry.writer \
            roles/logging.logWriter roles/cloudbuild.builds.builder; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$SA --role=$role --condition=None --quiet
done

# b) Connect GitHub (opens a link you must approve once in the browser)
gcloud builds connections create github github-conn --region=$REGION
gcloud builds connections describe github-conn --region=$REGION   # shows the authorization link if pending

# c) Link the repository
gcloud builds repositories create askmyrepo \
  --remote-uri=https://github.com/Sakshamxx/AskMyRepo.git \
  --connection=github-conn --region=$REGION

# d) Trigger: every push to main runs cloudbuild.yaml
gcloud builds triggers create github --name=askmyrepo-main \
  --region=$REGION \
  --repository=projects/$PROJECT_ID/locations/$REGION/connections/github-conn/repositories/askmyrepo \
  --branch-pattern='^main$' --build-config=cloudbuild.yaml \
  --service-account=projects/$PROJECT_ID/serviceAccounts/$SA
```

After that, `git push origin main` rebuilds and redeploys on its own (about 15 minutes). Watch it with `gcloud builds list --region $REGION --limit 3`.

**Changing the Gemini key:** add a new secret version, then redeploy (or push any commit). Running instances keep the old value until they restart.

```bash
printf '%s' "NEW_KEY" | gcloud secrets versions add gemini-api-key --data-file=-
gcloud run services update $SERVICE --region $REGION --update-env-vars=KEY_ROTATED=$(date +%s)   # forces a new revision
```

## Notes

- **Cost:** Cloud Run bills while a request is being handled (scale to zero when idle). Cloud Build and Secret Manager have small free tiers.
- **Public means anyone can spend your Gemini quota.** Prefer a private service or add a password.
- The Gemini free tier allows about 500 requests a day; the app stops answering (429) when it is used up.
