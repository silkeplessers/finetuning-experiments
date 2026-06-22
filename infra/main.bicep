// =============================================================================
// Bicep template that recreates the Azure footprint for the Dutch Mistral
// QLoRA finetuning project. Designed to be run once per new subscription:
//
//   az deployment sub create \
//     --location westeurope \
//     --template-file infra/main.bicep \
//     --parameters \
//         projectName=dutchmistral \
//         location=westeurope \
//         openAiLocation=swedencentral
//
// The template provisions:
//   - Resource group
//   - Storage account + the user-managed `finetuning-training-data` container
//   - Azure ML workspace (auto-creates Key Vault, App Insights, ACR,
//     and the `azureml-blobstore-<workspaceId>` container)
//   - User-assigned managed identity for training/inference jobs
//   - GPU compute cluster (scales 0 -> 1)
//   - Azure OpenAI account with three deployments (gen, judge1, judge2)
//   - RBAC role assignments so jobs + the developer can read/write blobs and
//     call Azure OpenAI
//
// Quota note: A100 / NC-class GPUs require quota in the target region. If you
// don't have any, switch `gpuVmSize` to `Standard_NC6s_v3` and request quota.
// Azure OpenAI model availability varies by region; check before running.
// =============================================================================

targetScope = 'subscription'

@description('Short project slug used as prefix. Lowercase, 3-12 chars.')
@minLength(3)
@maxLength(12)
param projectName string = 'dutchmistral'

@description('Primary Azure region for the resource group, workspace, storage and compute.')
param location string = 'westeurope'

@description('Region for the Azure OpenAI account (may differ — model availability varies).')
param openAiLocation string = 'swedencentral'

@description('VM size for the GPU training cluster.')
param gpuVmSize string = 'Standard_NC24ads_A100_v4'

@description('Maximum nodes in the training cluster (cluster scales from 0 to this).')
param gpuMaxNodes int = 1

@description('Object id of the developer (you). Get with `az ad signed-in-user show --query id -o tsv`.')
param developerObjectId string

@description('Name of the Azure OpenAI chat deployment used for synthetic data + judge 1.')
param chatDeploymentName string = 'gpt-4o-chat'

@description('Model name backing the chat deployment.')
param chatModelName string = 'gpt-4o'

@description('Model version backing the chat deployment.')
param chatModelVersion string = '2024-08-06'

@description('Name of the second judge deployment.')
param judge2DeploymentName string = 'gpt-4o-mini-judge'

@description('Model name for judge 2.')
param judge2ModelName string = 'gpt-4o-mini'

@description('Model version for judge 2.')
param judge2ModelVersion string = '2024-07-18'

// -----------------------------------------------------------------------------
// Resource group
// -----------------------------------------------------------------------------

var rgName = '${projectName}-rg'

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: rgName
  location: location
}

// -----------------------------------------------------------------------------
// Core resources (storage, workspace, compute, identity, AOAI) — in the RG
// -----------------------------------------------------------------------------

module core 'modules/core.bicep' = {
  name: 'core'
  scope: rg
  params: {
    projectName: projectName
    location: location
    openAiLocation: openAiLocation
    gpuVmSize: gpuVmSize
    gpuMaxNodes: gpuMaxNodes
    developerObjectId: developerObjectId
    chatDeploymentName: chatDeploymentName
    chatModelName: chatModelName
    chatModelVersion: chatModelVersion
    judge2DeploymentName: judge2DeploymentName
    judge2ModelName: judge2ModelName
    judge2ModelVersion: judge2ModelVersion
  }
}

output resourceGroupName string = rg.name
output workspaceName string = core.outputs.workspaceName
output storageAccountName string = core.outputs.storageAccountName
output userManagedIdentityClientId string = core.outputs.userManagedIdentityClientId
output openAiEndpoint string = core.outputs.openAiEndpoint
output computeName string = core.outputs.computeName
output dotenvHint string = core.outputs.dotenvHint
