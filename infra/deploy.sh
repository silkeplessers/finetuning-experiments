#!/usr/bin/env bash
# One-shot deployer for a fresh subscription. Idempotent — re-running updates in place.
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-dutchmistral}"
LOCATION="${LOCATION:-westeurope}"
OPENAI_LOCATION="${OPENAI_LOCATION:-swedencentral}"
GPU_VM_SIZE="${GPU_VM_SIZE:-Standard_NC24ads_A100_v4}"
DEPLOYMENT_NAME="${PROJECT_NAME}-deploy-$(date +%s)"

if ! command -v az >/dev/null 2>&1; then
  echo "az CLI not found. Install: https://learn.microsoft.com/cli/azure/install-azure-cli" >&2
  exit 1
fi

DEVELOPER_OBJECT_ID="$(az ad signed-in-user show --query id -o tsv)"
SUBSCRIPTION_ID="$(az account show --query id -o tsv)"

echo "Subscription:        ${SUBSCRIPTION_ID}"
echo "Resource group:      ${PROJECT_NAME}-rg in ${LOCATION}"
echo "Developer object id: ${DEVELOPER_OBJECT_ID}"
echo "GPU VM size:         ${GPU_VM_SIZE}"
echo "OpenAI region:       ${OPENAI_LOCATION}"
echo

az deployment sub create \
  --name "${DEPLOYMENT_NAME}" \
  --location "${LOCATION}" \
  --template-file "$(dirname "$0")/main.bicep" \
  --parameters \
      projectName="${PROJECT_NAME}" \
      location="${LOCATION}" \
      openAiLocation="${OPENAI_LOCATION}" \
      gpuVmSize="${GPU_VM_SIZE}" \
      developerObjectId="${DEVELOPER_OBJECT_ID}"

echo
echo "Done. Outputs:"
az deployment sub show --name "${DEPLOYMENT_NAME}" --query properties.outputs -o jsonc
