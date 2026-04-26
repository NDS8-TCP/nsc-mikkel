# P8: Numerical Scientific Computing

## Compiling a mini project
In order to compile a specific mini project, such as mp1, simply execute the 
`compile_mp1.py`. It will combine multiple lecture notebooks and generate a 
single notebook and pdf.

## Instalment
You can install and run using the following options:
1. Using devcontainer [*OCI Container*] (**Recommended**)
2. Using podman/docker interactively via the CLI [*OCI Container*] (Untested)
3. Using host OS [*OCI Container less*] (Untested)

### 1. Using devcontainers via VS Code
#### 1. Install either `podman` or `docker` [Official installation guides](#installation-links-for-docker-and-podman)

##### Podman
If you are using Podman, you must inform VS Code to use `podman` rather than 
its default of `docker`. Update your `settings.json` replacing `1000` with 
your current user ID by using `echo $UID`

```json
"dev.containers.dockerPath": "podman",
"dev.containers.dockerComposePath": "podman-compose",
"dev.containers.dockerSocketPath": "/run/user/1000/podman/podman.sock"
```
###### Enable the `podman.socket` service
```sh
systemctl --user enable --now podman.socket
```
You can verify that the socket is online and listening by running:
```sh
systemctl --user status podman.socket
```

#### 2. Install the "Container Tools" extension
Go to extensions -> Search "Container Tools" -> Install

#### 3. Start the dev container
Ctrl + Shift + P -> "Dev Containers: Reopen in Container"

Note that the first container build can take about 4 minutes. You can follow 
the process by clicking on the "(Show logs)" in the bottom right corner.


### 3. Using the host OS
Please see 
[the official mamba docs](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
for more details.

##### Installation links for Docker and Podman
Note that both offer a desktop GUI.
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Podman](https://podman.io/docs/installation)


### Configuring GPU access (OpenCL)
Copy in host specific ICDs: `cp --archive /etc/OpenCL/vendors opencl-vendors`
> EDIT AND REMOVE ICD FOR INTEL(CPU) AND AMD. These are provided via the container runtime.

#### NVIDIA
1. Install their [container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. Generate CDI: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`

#### AMD
1. Install their [container toolkit](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/cdi-guide.html)
2. Generate CDI: `sudo amd-ctk cdi generate --output=/etc/cdi/amd.json`
