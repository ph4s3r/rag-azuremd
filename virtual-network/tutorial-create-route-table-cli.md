---
title: Route network traffic - Azure CLI
description: In this article, learn how to route network traffic with a route table using the Azure CLI.
author: asudbring
ms.service: azure-virtual-network
ms.devlang: azurecli
ms.topic: how-to
ms.date: 08/08/2024
ms.author: allensu
ms.custom: devx-track-azurecli
# Customer intent: I want to route traffic from one subnet, to a different subnet, through a network virtual appliance.
---

# Route network traffic with a route table using the Azure CLI

Azure automatically routes traffic between all subnets within a virtual network, by default. You can create your own routes to override Azure's default routing. The ability to create custom routes is helpful if, for example, you want to route traffic between subnets through a network virtual appliance (NVA). In this article, you learn how to:

* Create a route table
* Create a route
* Create a virtual network with multiple subnets
* Associate a route table to a subnet
* Create a basic NVA that routes traffic from an Ubuntu VM
* Deploy virtual machines (VM) into different subnets
* Route traffic from one subnet to another through an NVA

[!INCLUDE [quickstarts-free-trial-note](~/reusable-content/ce-skilling/azure/includes/quickstarts-free-trial-note.md)]

[!INCLUDE [azure-cli-prepare-your-environment.md](~/reusable-content/azure-cli/azure-cli-prepare-your-environment.md)]

- This article requires version 2.0.28 or later of the Azure CLI. If using Azure Cloud Shell, the latest version is already installed.

## Create a route table

Before you can create a route table, create a resource group with [az group create](/cli/azure/group) for all resources created in this article.

```azurecli-interactive
# Create a resource group.
az group create \
  --name test-rg \
  --location westus2
```

Create a route table with [az network route-table create](/cli/azure/network/route-table#az-network-route-table-create). The following example creates a route table named *route-table-public*.

```azurecli-interactive
# Create a route table
az network route-table create \
  --resource-group test-rg \
  --name route-table-public
```

## Create a route

Create a route in the route table with [az network route-table route create](/cli/azure/network/route-table/route#az-network-route-table-route-create).

```azurecli-interactive
az network route-table route create \
  --name to-private-subnet \
  --resource-group test-rg \
  --route-table-name route-table-public \
  --address-prefix 10.0.1.0/24 \
  --next-hop-type VirtualAppliance \
  --next-hop-ip-address 10.0.2.4
```

## Associate a route table to a subnet

Before you can associate a route table to a subnet, you have to create a virtual network and subnet. Create a virtual network with one subnet with [az network vnet create](/cli/azure/network/vnet).

```azurecli-interactive
az network vnet create \
  --name vnet-1 \
  --resource-group test-rg \
  --address-prefix 10.0.0.0/16 \
  --subnet-name subnet-public \
  --subnet-prefix 10.0.0.0/24
```

Create two more subnets with [az network vnet subnet create](/cli/azure/network/vnet/subnet).

```azurecli-interactive
# Create a private subnet.
az network vnet subnet create \
  --vnet-name vnet-1 \
  --resource-group test-rg \
  --name subnet-private \
  --address-prefix 10.0.1.0/24

# Create a DMZ subnet.
az network vnet subnet create \
  --vnet-name vnet-1 \
  --resource-group test-rg \
  --name subnet-dmz \
  --address-prefix 10.0.2.0/24
```

Associate the *route-table-subnet-public* route table to the *subnet-public* subnet with [az network vnet subnet update](/cli/azure/network/vnet/subnet).

```azurecli-interactive
az network vnet subnet update \
  --vnet-name vnet-1 \
  --name subnet-public \
  --resource-group test-rg \
  --route-table route-table-public
```

## Create an NVA

An NVA is a VM that performs a network function, such as routing, firewalling, or WAN optimization. We create a basic NVA from a general purpose Ubuntu VM, for demonstration purposes.

Create a VM to be used as the NVA in the *subnet-dmz* subnet with [az vm create](/cli/azure/vm). When you create a VM, Azure creates and assigns a network interface *vm-nvaVMNic* and a subnet-public IP address to the VM, by default. The `--public-ip-address ""` parameter instructs Azure not to create and assign a subnet-public IP address to the VM, since the VM doesn't need to be connected to from the internet. 

The following example creates a VM and adds a user account. The `--generate-ssh-keys` parameter causes the CLI to look for an available ssh key in `~/.ssh`. If one is found, that key is used. If not, one is generated and stored in `~/.ssh`. Finally, we deploy the latest `Ubuntu 22.04` image.

```azurecli-interactive
az vm create \
  --resource-group test-rg \
  --name vm-nva \
  --image Ubuntu2204 \
  --public-ip-address "" \
  --subnet subnet-dmz \
  --vnet-name vnet-1 \
  --generate-ssh-keys
```

The VM takes a few minutes to create. Don't continue to the next step until Azure finishes creating the VM and returns output about the VM.

For a network interface **vm-nvaVMNic** to be able to forward network traffic sent to it, that isn't destined for its own IP address, IP forwarding must be enabled for the network interface. Enable IP forwarding for the network interface with [az network nic update](/cli/azure/network/nic).

```azurecli-interactive
az network nic update \
  --name vm-nvaVMNic \
  --resource-group test-rg \
  --ip-forwarding true
```

Within the VM, the operating system, or an application running within the VM, must also be able to forward network traffic. We use the `sysctl` command to enable the Linux kernel to forward packets. To run this command without logging onto the VM, we use the [Custom Script extension](/azure/virtual-machines/extensions/custom-script-linux) [az vm extension set](/cli/azure/vm/extension):

```azurecli-interactive
az vm extension set \
  --resource-group test-rg \
  --vm-name vm-nva \
  --name customScript \
  --publisher Microsoft.Azure.Extensions \
  --settings '{"commandToExecute":"sudo sysctl -w net.ipv4.ip_forward=1"}'
```

The command might take up to a minute to execute. This change won't persist after a VM reboot, so if the NVA VM is rebooted for any reason, the script will need to be repeated.

## Create virtual machines

Create two VMs in the virtual network so you can validate that traffic from the *subnet-public* subnet is routed to the *subnet-private* subnet through the NVA in a later step.

Create a VM in the *subnet-public* subnet with [az vm create](/cli/azure/vm). The `--no-wait` parameter enables Azure to execute the command in the background so you can continue to the next command.

The following example creates a VM and adds a user account. The `--generate-ssh-keys` parameter causes the CLI to look for an available ssh key in `~/.ssh`. If one is found, that key is used. If not, one is generated and stored in `~/.ssh`. Finally, we deploy the latest `Ubuntu 22.04` image.

```azurecli-interactive
az vm create \
  --resource-group test-rg \
  --name vm-public \
  --image Ubuntu2204 \
  --vnet-name vnet-1 \
  --subnet subnet-public \
  --admin-username azureuser \
  --generate-ssh-keys \
  --no-wait
```

Create a VM in the *subnet-private* subnet.

```azurecli-interactive
az vm create \
  --resource-group test-rg \
  --name vm-private \
  --image Ubuntu2204 \
  --vnet-name vnet-1 \
  --subnet subnet-private \
  --admin-username azureuser \
  --generate-ssh-keys
```

The VM takes a few minutes to create. After the VM is created, the Azure CLI shows information similar to the following example:

```output
{
  "fqdns": "",
  "id": "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/test-rg/providers/Microsoft.Compute/virtualMachines/vm-private",
  "location": "westus2",
  "macAddress": "00-0D-3A-23-9A-49",
  "powerState": "VM running",
  "privateIpAddress": "10.0.1.4",
  "publicIpAddress": "203.0.113.24",
  "resourceGroup": "test-rg"
}
```

## Enable Microsoft Entra ID sign in for the virtual machines

The following code example installs the extension to enable a Microsoft Entra ID sign-in for a Linux VM. VM extensions are small applications that provide post-deployment configuration and automation tasks on Azure virtual machines.

```bash
az vm extension set \
    --publisher Microsoft.Azure.ActiveDirectory \
    --name AADSSHsign-inForLinux \
    --resource-group test-rg \
    --vm-name vm-private
```

```bash
az vm extension set \
    --publisher Microsoft.Azure.ActiveDirectory \
    --name AADSSHsign-inForLinux \
    --resource-group test-rg \
    --vm-name vm-public
```

## Route traffic through an NVA

Using an SSH client of your choice, connect to the VMs created previously. For example, the following command can be used from a command line interface such as [Windows Subsystem for Linux](/windows/wsl/install) to create an SSH session with the *vm-private* VM. In the previous steps, we enabled Microsoft Entra ID sign-in for the VMs. You can sign-in to the virtual machines using your Microsoft Entra ID credentials or you can use the SSH key that you used to create the VMs. In the following example, we use the SSH key to sign-in to the VMs.

For more information about how to SSH to a Linux VM and sign in with Microsoft Entra ID, see [Sign in to a Linux virtual machine in Azure by using Microsoft Entra ID and OpenSSH](/entra/identity/devices/howto-vm-sign-in-azure-ad-linux).

```bash

### Store IP address of VM in order to SSH

Run the following command to store the IP address of the VM as an environment variable:

```bash
export IP_ADDRESS=$(az vm show --show-details --resource-group test-rg --name vm-private --query publicIps --output tsv)
```

```bash
ssh -o StrictHostKeyChecking=no azureuser@$IP_ADDRESS
```

Use the following command to install trace route on the *vm-private* VM:

```bash
sudo apt update
sudo apt install traceroute
```

Use the following command to test routing for network traffic to the *vm-public* VM from the *vm-private* VM.

```bash
traceroute vm-public
```

The response is similar to the following example:

```output
azureuser@vm-private:~$ traceroute vm-public
traceroute to vm-public (10.0.0.4), 30 hops max, 60 byte packets
 1  vm-public.internal.cloudapp.net (10.0.0.4)  2.613 ms  2.592 ms  2.553 ms
```

You can see that traffic is routed directly from the *vm-private* VM to the *vm-public* VM. Azure's default routes, route traffic directly between subnets. Close the SSH session to the *vm-private* VM.

### Store IP address of VM in order to SSH

Run the following command to store the IP address of the VM as an environment variable:

```bash
export IP_ADDRESS=$(az vm show --show-details --resource-group test-rg --name vm-public --query publicIps --output tsv)
```

```bash
ssh -o StrictHostKeyChecking=no azureuser@$IP_ADDRESS
```

Use the following command to install trace route on the *vm-public* VM:

```bash
sudo apt update
sudo apt install traceroute
```

Use the following command to test routing for network traffic to the *vm-private* VM from the *vm-public* VM.

```bash
traceroute vm-private
```

The response is similar to the following example:

```output
azureuser@vm-public:~$ traceroute vm-private
traceroute to vm-private (10.0.1.4), 30 hops max, 60 byte packets
 1  vm-nva.internal.cloudapp.net (10.0.2.4)  1.010 ms  1.686 ms  1.144 ms
 2  vm-private.internal.cloudapp.net (10.0.1.4)  1.925 ms  1.911 ms  1.898 ms
```

You can see that the first hop is 10.0.2.4, which is the NVA's private IP address. The second hop is 10.0.1.4, the private IP address of the *vm-private* VM. The route added to the *route-table--public* route table and associated to the *subnet-public* subnet caused Azure to route the traffic through the NVA, rather than directly to the *subnet-private* subnet.

Close the SSH session to the *vm-public* VM.

## Clean up resources

When no longer needed, use [az group delete](/cli/azure/group) to remove the resource group and all of the resources it contains.

```azurecli-interactive
az group delete \
    --name test-rg \
    --yes \
    --no-wait
```

## Next steps

In this article, you created a route table and associated it to a subnet. You created a simple NVA that routed traffic from a subnet-public subnet to a private subnet. Deploy various preconfigured NVAs that perform network functions such as firewall and WAN optimization from the [Azure Marketplace](https://azuremarketplace.microsoft.com/marketplace/apps/category/networking). To learn more about routing, see [Routing overview](virtual-networks-udr-overview.md) and [Manage a route table](manage-route-table.yml).

While you can deploy many Azure resources within a virtual network, resources for some Azure PaaS services can't be deployed into a virtual network. You can still restrict access to the resources of some Azure PaaS services to traffic only from a virtual network subnet though. To learn how, see [Restrict network access to PaaS resources](tutorial-restrict-network-access-to-resources-cli.md).
