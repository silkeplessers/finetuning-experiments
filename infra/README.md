# Infrastructure (Bicep)

Provisions the full Azure footprint needed to run the project in a fresh subscription.

## What it creates

- Resource group (`<projectName>-rg`)
- Storage account + the user-managed `finetuning-training-data` container
- Key Vault, Application Insights, and an Azure ML workspace
  (the workspace auto-creates the `azureml-blobstore-<workspaceId>` container)
- User-assigned managed identity (referenced by `azureml.managed_identity_client_id`)
- GPU compute cluster `gpu-cluster` (scales 0 → 1)
- Azure OpenAI account with two model deployments (chat + judge2)
- All RBAC role assignments the project needs:
  - You: `Storage Blob Data Contributor` + `AzureML Data Scientist` + `Cognitive Services OpenAI User`
  - Workspace MSI + UAMI: `Storage Blob Data Contributor`
  - UAMI: `Cognitive Services OpenAI User`

## Prerequisites

- Azure CLI logged in to the **target** subscription (`az login && az account set --subscription <id>`)
- Sufficient GPU quota in the chosen region. If you can't get A100, switch to
  `Standard_NC6s_v3` via the `GPU_VM_SIZE` env var.
- Azure OpenAI access enabled on the subscription and the chosen models available
  in `OPENAI_LOCATION` ([region availability matrix](https://learn.microsoft.com/azure/ai-services/openai/concepts/models)).

## One-shot deploy

```bash
./deploy.sh
```

Override defaults via env vars:

```bash
PROJECT_NAME=dutchmistral \
LOCATION=westeurope \
OPENAI_LOCATION=swedencentral \
GPU_VM_SIZE=Standard_NC6s_v3 \
./deploy.sh
```

## Manual deploy

```bash
az deployment sub create \
  --location westeurope \
  --template-file infra/main.bicep \
  --parameters \
      projectName=dutchmistral \
      location=westeurope \
      openAiLocation=swedencentral \
      developerObjectId=$(az ad signed-in-user show --query id -o tsv)
```

## After deployment

The deployment outputs include a `dotenvHint` that you can paste straight into `.env`:

```bash
az deployment sub show --name <deployment-name> --query properties.outputs.dotenvHint.value -o tsv > .env
```

You still need to fill in `HF_TOKEN` and `WANDB_API_KEY` (those aren't Azure resources).

Then update [configs/qlora_config.json](../configs/qlora_config.json):

- `azureml.managed_identity_client_id` → `userManagedIdentityClientId` from the deployment output
- `azureml.compute` → `gpu-cluster` (default)

And update the hard-coded constants listed in the root [README.md](../README.md#hard-coded-references-to-be-aware-of-when-migrating)
so they point at the new storage account / blobstore container.

## Tear down

```bash
az group delete --name dutchmistral-rg --yes --no-wait
```

Note: Key Vault has soft-delete + purge protection enabled (required for AzureML).
After deleting the RG, the Key Vault will linger for 7 days; purge with
`az keyvault purge --name <kv-name>` if you need to redeploy with the same name.

## Cost notes

With everything idle (cluster at 0 nodes, no AOAI calls), expect roughly
€2–5/month for App Insights + Key Vault + storage. Costs spike only when the
GPU cluster scales up or when you run judge calls.
