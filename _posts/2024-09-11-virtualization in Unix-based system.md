---
title: Virtualization in Unix-based system
description: Explaining virtualisation & containerisation using a unix-based os
date: 2024-09-11
categories:
  - Cloud Computing
  - DevOps
tags:
  - Computing
  - Infrastracture
  - Virtualisation
  - DevOps
pin: true
math: true
mermaid: true
image:
  path: /assets/img/panels/panel5@4x.png
---

### Virtualization in Unix-based system

### Introduction

Virtualization in Unix-based systems, including Linux, involves creating and managing virtual instances of computing resources, such as hardware, storage, and network, allowing multiple isolated environments to run on a single physical machine. This process is achieved through a combination of kernel features, hypervisors, and various user-space tools. Here’s a detailed explanation of how virtualization works under the hood:

### 0. What’s Virtualisation

Virtualization is a technology that allows for the creation of multiple simulated environments or dedicated resources from a single physical hardware system. It involves the abstraction of physical resources — such as servers, storage devices, and network infrastructure — into virtual counterparts, which can be used and managed independently.

<div style="text-align: center;">
  <figure>
    <img src="https://cdn-images-1.medium.com/max/1000/1*_J3YeyU3Rd2P9Tg84YUMWw.png" alt="" width="105%">
  </figure>
</div>


In Unix-based systems, virtualization enables the concurrent running of multiple isolated instances, often referred to as virtual machines (VMs) or containers, on a single physical machine, optimizing resource utilization and providing robust isolation and security. This process is facilitated by a combination of hardware-assisted features, hypervisors, kernel modules, and user-space tools.

### 1. Hardware-Assisted Virtualization

Modern processors support hardware-assisted virtualization features (Intel VT-x and AMD-V) that allow the efficient execution of multiple virtual machines (VMs).

- **VMX (Virtual Machine Extensions) and SVM (Secure Virtual Machine)**: These extensions add new CPU modes to facilitate virtualization. The processor can switch between the host (root mode) and guest (non-root mode) execution contexts efficiently.
- **VMCS (Virtual Machine Control Structure)**: A data structure used by the CPU to manage the state of each virtual machine.

### 2. Hypervisors

A hypervisor (virtual machine monitor) is software that creates and runs VMs. There are two types of hypervisors:

- **Type 1 (Bare-Metal)**: Runs directly on the host’s hardware (e.g., VMware ESXi, Xen).
- **Type 2 (Hosted)**: Runs on a host operating system (e.g., VMware Workstation, VirtualBox).

**KVM (Kernel-based Virtual Machine)** is a Type 2 hypervisor integrated into the Linux kernel.

#### KVM Workflow

1. **Loading the KVM Module**:

sudo modprobe kvm sudo modprobe kvm_intel  # For Intel processors

1. **Creating a Virtual Machine**:

- **QEMU**: A user-space emulator that works with KVM to provide hardware emulation.

```bash
qemu-system-x86_64 -enable-kvm -m 1024 -smp 2 -hda /mnt/disk/ubunut.iso
```

**2. VMX Operations**:

- **VMXON**: Enables VMX operation.
- **VMLAUNCH**: Starts the guest execution.
- **VMEXIT**: Transfers control back to the hypervisor.

#### Creating virtual machines using CLI

To create a VM using `virt-install` utility , you should first enable virtualization in the host system

Check if you have

- Good RAM memory amount and quality
- Good CPU cores and frequency
- Enough Disk Space
- Other Driver needed

All those resources values may vary significantly depending on the intended tasks and workload on the VMs

You should also let some resources to your host OS if you’re using an invited mode

<div style="text-align: center;">
  <figure>
    <img src="https://cdn-images-1.medium.com/max/1000/1*RG_Jc0rnukWRhy6a9k6qYA.png" alt="" width="50%">
  </figure>
</div>



### 3. Containers

Containers are a lightweight form of virtualization that provides process isolation through namespaces and resource control via cgroups (control groups).

#### Namespaces

Namespaces provide isolation for various system resources.

- **PID Namespace**: Isolates process IDs.
- **Network Namespace**: Isolates network interfaces.
- **Mount Namespace**: Isolates file system mounts.
- **User Namespace**: Isolates user and group IDs.
- **IPC Namespace**: Isolates inter-process communication resources.

Example using `unshare`:

```bash
sudo unshare --mount --uts --ipc --net --pid --fork --user --map-root-user /bin/bash
```

#### Control Groups (cgroups)

cgroups limit and isolate resource usage (CPU, memory, disk I/O, etc.).

- **Creating a cgroup**:

```bash
sudo mkdir /sys/fs/cgroup/cpu/my_cgroup echo 50000 | sudo tee /sys/fs/cgroup/cpu/my_cgroup/cpu.cfs_quota_us echo $$ | sudo tee /sys/fs/cgroup/cpu/my_cgroup/tasks
```

#### Docker

Docker simplifies container management and provides tools to build, ship, and run containers.

- **Running a Docker Container**:

```bash
docker run -it ubuntu /bin/bash
```

### 4. Virtual Filesystems and Networking

- **OverlayFS**: A union mount filesystem that allows the creation of layers, which is used in Docker to provide image layering.
- **Bridged Networking**: Allows VMs and containers to be on the same network as the host, providing them with virtual network interfaces.

### 5. Memory Management

Virtualization involves sophisticated memory management techniques:

- **Extended Page Tables (EPT) / Nested Page Tables (NPT)**: Hardware features that map virtual memory addresses used by VMs to physical memory addresses, reducing the overhead of memory translation.

### 6. I/O Virtualization

- **Virtio**: A virtualization standard for network and disk device drivers, providing efficient and high-performance I/O operations in virtualized environments.

### Example: Setting Up a KVM Virtual Machine

Here’s a step-by-step example of setting up a VM using KVM and QEMU:

1. **Install KVM and QEMU**:

```bash
sudo apt-get install qemu-kvm libvirt-bin
```

2. **Verify Installation**:

```bash
kvm-ok  # Check if your system supports KVM
```
3. **Create a Disk Image**:

```bash
qemu-img create -f qcow2 /var/lib/libvirt/images/myvm.qcow2 10G
```

4. **Install an Operating System**:

```bash
virt-install --name myvm --ram 1024 --disk path=/var/lib/libvirt/images/myvm.qcow2 --vcpus 1 --os-type linux --network bridge=virbr0 --graphics none --console pty,target_type=serial --location 'http://archive.ubuntu.com/ubuntu/dists/bionic/main/installer-amd64/' --extra-args 'console=ttyS0,115200n8 serial'
```

### Conclusion

<div style="text-align: center;">
  <figure>
    <img src="https://cdn-images-1.medium.com/max/1000/1*bAX-RH8xwTdIP6FCLD30sg.png" alt="" width="105%">
  </figure>
</div>

Virtualization in Unix-based systems involves a complex interplay of hardware features, kernel modules, and user-space tools. It leverages hardware support for virtualization, kernel features like namespaces and cgroups, and powerful user-space tools like QEMU, KVM, and Docker to create isolated, efficient, and scalable computing environments. This architecture enables multiple virtual environments to coexist on a single physical machine, optimizing resource utilization and providing robust isolation and security.
