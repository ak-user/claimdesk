# Run ClaimDesk on Azure

This guide connects ClaimDesk to Azure OpenAI, signs in without API keys, and deploys the app to
Azure Container Apps. It takes about an hour the first time. Azure changes its CLI often, so if a
flag is rejected, run the command with `--help` and adjust.

What you need: an Azure subscription, the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli),
and this repository.

## 1. Create the resources

Canadian regions keep the data in Canada. Model availability differs by region, so check before
you pick one.

```bash
az login
az group create --name rg-claimdesk --location canadaeast

# The custom domain is required for keyless (Microsoft Entra ID) sign-in.
az cognitiveservices account create \
  --name <unique-name> \
  --resource-group rg-claimdesk \
  --kind OpenAI \
  --sku S0 \
  --location canadaeast \
  --custom-domain <unique-name>

# See which models this resource can deploy.
az cognitiveservices account list-models \
  --name <unique-name> --resource-group rg-claimdesk --output table
```

Deploy one chat model and one embedding model. Pick a current small chat model from the list,
and an embedding model such as `text-embedding-3-small` if it's offered.

```bash
az cognitiveservices account deployment create \
  --name <unique-name> --resource-group rg-claimdesk \
  --deployment-name chat \
  --model-name <chat-model> --model-version <version> --model-format OpenAI \
  --sku-name Standard --sku-capacity 10

az cognitiveservices account deployment create \
  --name <unique-name> --resource-group rg-claimdesk \
  --deployment-name embedding \
  --model-name <embedding-model> --model-version <version> --model-format OpenAI \
  --sku-name Standard --sku-capacity 10
```

If a model isn't available as `Standard` in your region, try `--sku-name GlobalStandard`.

## 2. Run locally against Azure OpenAI, without a key

Give your own account the data-plane role on the resource. Role assignments can take about five
minutes to apply; a `403` right after assigning means wait and retry.

```bash
OPENAI_ID=$(az cognitiveservices account show \
  --name <unique-name> --resource-group rg-claimdesk --query id --output tsv)

az role assignment create \
  --assignee "$(az ad signed-in-user show --query id --output tsv)" \
  --role "Cognitive Services OpenAI User" \
  --scope "$OPENAI_ID"
```

Then create `backend/.env`:

```bash
AZURE_OPENAI_ENDPOINT=https://<unique-name>.openai.azure.com
AZURE_OPENAI_CHAT_DEPLOYMENT=chat
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=embedding
```

With no `AZURE_OPENAI_API_KEY`, ClaimDesk signs in with `DefaultAzureCredential`, which uses your
`az login` session.

```bash
cd backend
pip install -e ".[dev,azure]"
uvicorn claimdesk.api:app --reload
curl http://localhost:8000/api/health    # "llm_provider": "azure", "retrieval": "hybrid"
python -m claimdesk.eval.run_eval        # the same eval, now on the model path
```

## 3. Deploy to Azure Container Apps

`az containerapp up` builds the Dockerfile in Azure Container Registry, so you don't need Docker
locally. The Dockerfile's `EXPOSE 8000` sets the ingress port.

```bash
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights

az containerapp up \
  --name claimdesk \
  --resource-group rg-claimdesk \
  --location canadacentral \
  --source . \
  --ingress external
```

Give the app a managed identity, let it call Azure OpenAI, and set its configuration:

```bash
az containerapp identity assign \
  --name claimdesk --resource-group rg-claimdesk --system-assigned

PRINCIPAL_ID=$(az containerapp show \
  --name claimdesk --resource-group rg-claimdesk --query identity.principalId --output tsv)

az role assignment create \
  --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services OpenAI User" \
  --scope "$OPENAI_ID"

az containerapp update \
  --name claimdesk --resource-group rg-claimdesk \
  --set-env-vars \
    AZURE_OPENAI_ENDPOINT=https://<unique-name>.openai.azure.com \
    AZURE_OPENAI_CHAT_DEPLOYMENT=chat \
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=embedding
```

Open the app's URL:

```bash
az containerapp show --name claimdesk --resource-group rg-claimdesk \
  --query properties.configuration.ingress.fqdn --output tsv
```

No key is stored anywhere: the container gets tokens from its managed identity.

## 4. Clean up

Azure OpenAI bills per token and Container Apps per use, so delete everything when you're done:

```bash
az group delete --name rg-claimdesk --yes --no-wait
```

## Next steps

- **Azure AI Search.** Replace the in-memory index in `retrieval.py` with an Azure AI Search
  index (keyword plus vector, with semantic ranking). Keep the `Retriever.search` signature so
  nothing else changes.
- **Document Intelligence.** Read scanned claim forms with the prebuilt layout model, then pass
  the text to `ClaimAssessor.assess`.
- **Databricks.** Set `MLFLOW_TRACKING_URI=databricks` with `DATABRICKS_HOST` and
  `DATABRICKS_TOKEN`, then run `python -m claimdesk.eval.run_eval --mlflow` to track eval runs in
  a Databricks workspace.

References: [Container Apps from local source](https://learn.microsoft.com/azure/container-apps/quickstart-code-to-cloud),
[Azure OpenAI with managed identity](https://learn.microsoft.com/azure/ai-foundry/openai/how-to/managed-identity).
