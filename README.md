# P8: Numerical Scientific Computing
1. Course Intro and Development Tools
2. Computer Architecture and Memory
3. Optimization and Numba

## Compiling a mini project
In order to compile a specific mini project, such as mp1, simply execute the 
`compile_mp1.py`. It will combine multiple lecture notebooks and generate a 
single notebook and pdf.

## Instalment
You can install and run using the following options:
1. Using devcontainer [*OCI Container*] (**Recommended**)
2. Using podman/docker interactively via the CLI [*OCI Container*]
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

Note that the first container build can take about 2 minutes. You can follow 
the process by clicking on the "(Show logs)" in the bottom right corner.


### 2. Using `podman` or `docker` CLI
#### 1. Install either `podman` or `docker` [Official installation guides](#installation-links-for-docker-and-podman)
Note that the subsequent commands, `podman` can be exchanged with `docker`.

#### 2. Create a (disposable) image.
Ensure that your current pwd is the root of this repo.

```sh
podman run --rm -it \
    --name nsc \
    --workdir /nsc \
    --volume .:/nsc \
    quay.io/condaforge/miniforge3:25.11.0-1 \
    /usr/bin/bash
```

#### 3. Make `mamba` and `conda` available in subsequent shells
```bash
mamba shell init --shell bash
```
Also make then available immediately in this current shell.
```bash
eval "$(mamba shell hook --shell bash)"
```

#### 4. Create the virtual mamba environment
```bash
mamba env create --file environment.yaml --yes
```

#### 5. Activate the virtual mamba environment
```bash
mamba activate nsc
```

### 3. Using the host OS
Please see 
[the official mamba docs](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
for more details.

##### Installation links for Docker and Podman
Note that both offer a desktop GUI.
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Podman](https://podman.io/docs/installation)

