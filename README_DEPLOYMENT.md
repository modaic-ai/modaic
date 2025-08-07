# Documentation Trigger System

This repository includes an automated documentation trigger system that notifies external repositories when documentation changes occur, allowing them to pull and build the latest documentation.

## Workflows

### 1. CI Workflow (`ci.yml`)
- **Trigger**: Push to `main` or `master` branch
- **Purpose**: Standard CI/CD pipeline for documentation deployment
- **Output**: Deploys to GitHub Pages using `mkdocs gh-deploy`

### 2. Documentation Trigger Workflow (`doc-sync.yml`)
- **Triggers**: 
  - Push to main/master with documentation changes
  - Manual trigger (workflow_dispatch)
- **Purpose**: Notify external repositories when documentation is updated
- **Output**: Sends `repository_dispatch` events to configured external repositories

## How It Works

### Push-Based Triggers
When documentation files are modified in this repository:

1. **Change Detection**: The workflow detects changes in:
   - `docs/**` - Documentation content
   - `src/**` - Source code (for API docs)
   - `mkdocs.yml` - Documentation configuration
   - `pyproject.toml` - Project dependencies

2. **External Notification**: Sends `repository_dispatch` events to configured external repositories with payload:
   ```json
   {
     "event_type": "modaic-docs-updated",
     "client_payload": {
       "source_repo": "modaic-ai/modaic",
       "source_commit": "abc123...",
       "source_branch": "main",
       "triggered_by": "documentation_update",
       "timestamp": "2024-01-01T12:00:00Z"
     }
   }
   ```

### Manual Triggers
You can manually trigger external repositories from the GitHub Actions tab:

1. Go to Actions → Documentation Trigger
2. Click "Run workflow"
3. Optionally specify target repositories (comma-separated list of `owner/repo`)

## Configuration

### Setting Target Repositories
Edit the workflow file `.github/workflows/doc-sync.yml` and update the default repositories:

```yaml
DEFAULT_REPOS="modaic-ai/website,your-org/your-website"
```

### External Repository Setup
External repositories should implement a workflow that:

1. **Listens** for `repository_dispatch` events with type `modaic-docs-updated`
2. **Pulls** the modaic repository source
3. **Builds** the documentation using MkDocs
4. **Deploys** the built documentation

See `scripts/example-website-workflow.yml` for a complete example.

## External Repository Workflow Example

The external repository workflow should:

```yaml
name: Build Documentation
on:
  repository_dispatch:
    types: [modaic-docs-updated]
  workflow_dispatch:

jobs:
  build-and-deploy-docs:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout website repository
        uses: actions/checkout@v4
      
      - name: Checkout modaic documentation
        uses: actions/checkout@v4
        with:
          repository: modaic-ai/modaic
          ref: ${{ github.event.client_payload.source_branch || 'main' }}
          path: modaic-source
      
      - name: Set up Python and build docs
        run: |
          pip install mkdocs-material mkdocstrings[python] pymdown-extensions
          cd modaic-source
          pip install -e .
          mkdocs build
      
      - name: Deploy to GitHub Pages
        # ... deployment steps
```

## Permissions Required

### This Repository
- Uses default `GITHUB_TOKEN` with `contents: read` permissions
- No additional secrets required for triggering external repositories

### External Repositories
- Need `contents: read` and `pages: write` permissions
- May need a Personal Access Token if pulling from private repositories

## Troubleshooting

### Common Issues
1. **Trigger not received**: Verify the target repository name in the workflow configuration
2. **External build fails**: Check that the external repository has the required workflow file
3. **Permission denied**: Ensure the external repository workflow has proper permissions

### Debugging
- Check the Actions tab in both repositories for detailed logs
- Verify the `repository_dispatch` event type matches (`modaic-docs-updated`)
- Test manual triggers to isolate issues

### Configuration Check
```bash
# Test the trigger manually
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/YOUR-ORG/YOUR-WEBSITE/dispatches \
  -d '{"event_type":"modaic-docs-updated"}'
```