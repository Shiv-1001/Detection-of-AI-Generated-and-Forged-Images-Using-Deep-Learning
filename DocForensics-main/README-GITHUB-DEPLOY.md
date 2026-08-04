# GitHub deployment quickstart

This repository already contains a Docker-based app. To publish it on GitHub:

1. Push this repo to a GitHub repository.
2. In the repo settings, confirm GitHub Actions are enabled.
3. On the `main` branch, the workflow in `.github/workflows/docker-publish.yml` will build and push the image to GitHub Container Registry.
4. The resulting image reference will look like:

   `ghcr.io/<your-github-user>/docforensics:latest`

5. Use that image in GitHub-hosted runners, another host, or a container service.

Notes:

- The app listens on port `7860` by default, or any `PORT` environment variable you supply.
- The first request downloads the AI model into the runtime cache.
- `DOCFORENSICS_DISABLE_AI_MODEL=1` can be used on smaller hosts.
