---
title: Bicep functions - parameters file
description: This article describes the Bicep functions to be used in Bicep parameter files.
ms.topic: reference
ms.custom: devx-track-bicep
ms.date: 08/09/2024
---

# Parameters file function for Bicep

Bicep provides a function called `readEnvironmentVariable()` that allows you to retrieve values from environment variables. It also offers the flexibility to set a default value if the environment variable doesn't exist. This function can only be used in the `.bicepparam` files. For more information, see [Bicep parameters file](./parameter-files.md).

## getSecret

`getSecret(subscriptionId, resourceGroupName, keyVaultName, secretName, secretVersion)`

Returns a secret from an [Azure Key Vault](/azure/key-vault/secrets/about-secrets). Use this function to pass a secret to a secure string parameter of a Bicep file.

> [!NOTE]
> You can also use the [keyVaultName.getSecret(secretName)](./bicep-functions-resource.md#getsecret) function from within a `.bicep` file.

```bicep
using './main.bicep'

param secureUserName = getSecret('exampleSubscription', 'exampleResourceGroup', 'exampleKeyVault', 'exampleSecretUserName')
param securePassword = getSecret('exampleSubscription', 'exampleResourceGroup', 'exampleKeyVault', 'exampleSecretPassword')
```

You get an error if you use this function with string interpolation.

A [namespace qualifier](bicep-functions.md#namespaces-for-functions) (`az`) can be used, but it's optional, because the function is available from the _default_ Azure Namespace.

### Parameters

| Parameter | Required | Type | Description |
|:--- |:--- |:--- |:--- |
| subscriptionId | Yes | string | The ID of the subscription that has the key vault resource. |
| resourceGroupName | Yes | string | The name of the resource group that has the key vault resource. |
| keyVaultName | Yes | string | The name of the key vault. |
| secretName | Yes | string | The name of the secret stored in the key vault. |
| secretVersion | No | string | The version of the secret stored in the key vault. |

### Return value

The value for the secret.

### Example

The following `.bicepparam` file has a `securePassword` parameter that has the latest value of the _\<secretName\>_ secret.

```bicep
using './main.bicep'

param securePassword = getSecret('exampleSubscription', 'exampleResourceGroup', 'exampleKeyVault', 'exampleSecretPassword')
```

The following `.bicepparam` file has a `securePassword` parameter that has the value of the _\<secretName\>_ secret, but it's pinned to a specific _\<secretValue\>_.

```bicep
using './main.bicep'

param securePassword = getSecret('exampleSubscription', 'exampleResourceGroup', 'exampleKeyVault', 'exampleSecretPassword', 'exampleSecretVersion')
```

## readEnvironmentVariable

`readEnvironmentVariable(variableName, [defaultValue])`

Returns the value of the environment variable, or set a default value if the environment variable doesn't exist. Variable loading occurs during compilation, not at runtime.

Namespace: [sys](bicep-functions.md#namespaces-for-functions).

### Parameters

| Parameter | Required | Type | Description |
|:--- |:--- |:--- |:--- |
| variableName | Yes | string | The name of the variable. |
| defaultValue | No | string | A default string value to be used if the environment variable doesn't exist. |

### Return value

The string value of the environment variable or a default value.

### Remarks

The following command sets the environment variable only for the PowerShell process in which it's executed. You get [BCP338](./diagnostics/bcp338.md) from Visual Studio Code.

```PowerShell
$env:testEnvironmentVariable = "Hello World!"
```

To set the environment variable at the user level, use the following command:

```powershell
[System.Environment]::SetEnvironmentVariable('testEnvironmentVariable','Hello World!', 'User')
```

To set the environment variable at the machine level, use the following command:

```powershell
[System.Environment]::SetEnvironmentVariable('testEnvironmentVariable','Hello World!', 'Machine')
```

For more information, see [Environment.SetEnvironmentVariable Method](/dotnet/api/system.environment.setenvironmentvariable).

### Examples

The following examples show how to retrieve the values of environment variables.

```bicep
use './main.bicep'

param adminPassword = readEnvironmentVariable('admin_password')
param boolfromEnvironmentVariables = bool(readEnvironmentVariable('boolVariableName','false'))
```

## Next steps

For more information about Bicep parameters file, see [Parameters file](./parameter-files.md).
