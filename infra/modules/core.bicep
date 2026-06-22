// Resource-group-scoped module: storage, AzureML workspace, GPU cluster,
// user-assigned managed identity, Azure OpenAI account + deployments, and
// the role assignments needed for the project to work end-to-end.

targetScope = 'resourceGroup'

param projectName string
param location string
param openAiLocation string
param gpuVmSize string
param gpuMaxNodes int
param developerObjectId string
param chatDeploymentName string
param chatModelName string
param chatModelVersion string
param judge2DeploymentName string
param judge2ModelName string
param judge2ModelVersion string

var suffix = uniqueString(resourceGroup().id)
var storageName = take(toLower('${projectName}st${suffix}'), 24)
var kvName = take('${projectName}-kv-${suffix}', 24)
var appInsightsName = '${projectName}-appi-${suffix}'
var workspaceName = '${projectName}-aml'
var uamiName = '${projectName}-uami'
var openAiName = '${projectName}-aoai-${suffix}'
var userContainerName = 'finetuning-training-data'

// Built-in role definition ids
var roleStorageBlobDataContributor = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
var roleAzureMLDataScientist = 'f6c7c914-8db3-469d-8ca1-694a8f32e121'
var roleCognitiveServicesOpenAIUser = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'

// -----------------------------------------------------------------------------
// Storage account + user-managed container
// -----------------------------------------------------------------------------

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    allowSharedKeyAccess: true // AzureML workspace creation still needs this
    encryption: {
      services: {
        blob: { enabled: true }
        file: { enabled: true }
      }
      keySource: 'Microsoft.Storage'
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource userContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: userContainerName
  properties: {
    publicAccess: 'None'
  }
}

// -----------------------------------------------------------------------------
// Key Vault + Application Insights (AzureML workspace dependencies)
// -----------------------------------------------------------------------------

resource keyVault 'Microsoft.KeyVault/vaults@2024-04-01-preview' = {
  name: kvName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: true
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    Request_Source: 'rest'
  }
}

// -----------------------------------------------------------------------------
// User-assigned managed identity (referenced by training/inference jobs)
// -----------------------------------------------------------------------------

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: uamiName
  location: location
}

// -----------------------------------------------------------------------------
// Azure ML workspace
// -----------------------------------------------------------------------------

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: workspaceName
  location: location
  identity: {
    type: 'SystemAssigned,UserAssigned'
    userAssignedIdentities: {
      '${uami.id}': {}
    }
  }
  properties: {
    friendlyName: '${projectName} workspace'
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    publicNetworkAccess: 'Enabled'
  }
}

// -----------------------------------------------------------------------------
// GPU compute cluster (scales 0 -> gpuMaxNodes)
// -----------------------------------------------------------------------------

resource computeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2024-04-01' = {
  parent: workspace
  name: 'gpu-cluster'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${uami.id}': {}
    }
  }
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: gpuVmSize
      vmPriority: 'Dedicated'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: gpuMaxNodes
        nodeIdleTimeBeforeScaleDown: 'PT15M'
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Azure OpenAI account + deployments
// -----------------------------------------------------------------------------

resource openAi 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: openAiName
  location: openAiLocation
  kind: 'OpenAI'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: openAiName
    publicNetworkAccess: 'Enabled'
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAi
  name: chatDeploymentName
  sku: { name: 'Standard', capacity: 50 }
  properties: {
    model: {
      format: 'OpenAI'
      name: chatModelName
      version: chatModelVersion
    }
  }
}

resource judge2Deployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAi
  name: judge2DeploymentName
  sku: { name: 'Standard', capacity: 50 }
  properties: {
    model: {
      format: 'OpenAI'
      name: judge2ModelName
      version: judge2ModelVersion
    }
  }
  dependsOn: [
    chatDeployment // CognitiveServices serialises deployments per-account
  ]
}

// -----------------------------------------------------------------------------
// RBAC
// -----------------------------------------------------------------------------

// Developer -> Storage Blob Data Contributor
resource devBlobRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, developerObjectId, roleStorageBlobDataContributor)
  scope: storage
  properties: {
    principalId: developerObjectId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleStorageBlobDataContributor)
  }
}

// UAMI -> Storage Blob Data Contributor (used by AML jobs)
resource uamiBlobRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, uami.id, roleStorageBlobDataContributor)
  scope: storage
  properties: {
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleStorageBlobDataContributor)
  }
}

// Workspace system identity -> Storage Blob Data Contributor (so the
// workspaceblobstore datastore works for the orchestrator).
resource workspaceBlobRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, workspace.id, roleStorageBlobDataContributor)
  scope: storage
  properties: {
    principalId: workspace.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleStorageBlobDataContributor)
  }
}

// Developer -> Azure ML Data Scientist on the workspace
resource devAmlRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(workspace.id, developerObjectId, roleAzureMLDataScientist)
  scope: workspace
  properties: {
    principalId: developerObjectId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleAzureMLDataScientist)
  }
}

// Developer -> Cognitive Services OpenAI User
resource devOpenAiRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAi.id, developerObjectId, roleCognitiveServicesOpenAIUser)
  scope: openAi
  properties: {
    principalId: developerObjectId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleCognitiveServicesOpenAIUser)
  }
}

// UAMI -> Cognitive Services OpenAI User (so AML jobs can also call AOAI)
resource uamiOpenAiRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAi.id, uami.id, roleCognitiveServicesOpenAIUser)
  scope: openAi
  properties: {
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleCognitiveServicesOpenAIUser)
  }
}

// -----------------------------------------------------------------------------
// Outputs
// -----------------------------------------------------------------------------

output workspaceName string = workspace.name
output storageAccountName string = storage.name
output userContainerName string = userContainerName
output userManagedIdentityClientId string = uami.properties.clientId
output openAiEndpoint string = 'https://${openAi.properties.customSubDomainName}.openai.azure.com/openai/v1/'
output computeName string = computeCluster.name
output dotenvHint string = format(
  'ENDPOINT={0}\nDEPLOYMENT={1}\nJUDGE_LLM_1={1}\nJUDGE_LLM_2={2}\nSTORAGE_ACCOUNT={3}\nCONTAINER_NAME={4}',
  'https://${openAi.properties.customSubDomainName}.openai.azure.com/openai/v1/',
  chatDeploymentName,
  judge2DeploymentName,
  storage.name,
  userContainerName
)
