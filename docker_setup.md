# Docker setup

Run the AskMyRepo web app (Streamlit, port 8501) in a container. Two Dockerfiles:

| File | Use | What is inside |
|---|---|---|
| `Dockerfile` | Local Docker | Code and libraries only. Models, database and keys are **mounted at run time**. |
| `Dockerfile.cloud` | Cloud Run (see `GCP_setup.md`) | Same, plus both models and ShopLite's database baked in. |

## 1. Before you start

- Docker Desktop installed and running (`docker info` must print a *Server* section).
- Docker Desktop memory: **6 GB or more** (Settings → Resources). The two Qwen models need about 3 GB when loaded.
- About **8 GB of free disk**: the image includes PyTorch.
- `.env` filled in (copy `.env.example`): `GEMINI_API_KEY` and `PINECONE_API_KEY`.
- The index already built once (`python -m askmyrepo index`, see `README.md`). The container reads the saved keyword index from `index/` and the vectors from Pinecone.

## 2. Build

```bash
cd askmyrepo/repo
docker build -t askmyrepo:latest .
```

## 3. Run

```bash
docker run --rm -p 8501:8501 \
  --env-file .env \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)/../shoplite/instance/shoplite.db:/data/shoplite.db:ro" \
  askmyrepo:latest
```

Open http://localhost:8501.

| Flag | Why |
|---|---|
| `--env-file .env` | Passes the keys at run time. They are never written into the image. |
| `-v ~/.cache/huggingface:...` | Reuses the models you already downloaded (about 2.4 GB), so the container doesn't download them again. |
| `-v .../shoplite.db:/data/shoplite.db:ro` | ShopLite's demo database, mounted read-only. Create it first: `cd ../shoplite && uv run python scripts/seed.py`. |
| `-p 8501:8501` | The Streamlit port. |

The first question loads the models (30–60 s); later ones are faster.

## 4. Everyday commands

```bash
docker ps                                  # is it running?
docker logs -f <container id>              # follow the output
docker stop <container id>                 # stop it
docker images askmyrepo                    # image size
```

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Start Docker Desktop (`open -a Docker`) and wait for it to finish starting. |
| Container exits or is killed while loading models | Out of memory: raise Docker Desktop's memory limit to 6 GB+. |
| `no database at /data/shoplite.db` | The database mount is missing or the path is wrong. Seed ShopLite first. |
| `PINECONE_API_KEY` / `GEMINI_API_KEY` missing | Add them to `.env` and pass `--env-file .env`. |
| `no space left on device` | `docker rmi askmyrepo:latest && docker builder prune -f`, then `df -h /`. |
| Questions fail with 429 | Gemini free-tier daily quota is used up; wait for it to reset or use a paid key. |

## 6. Free disk space again

```bash
docker rmi askmyrepo:latest     # remove the image
docker builder prune -f         # remove the build cache
docker system df                # see what Docker still holds
```
