# MLflow Guide with SSH Tunneling

This guide explains how to set up an MLflow tracking server on a remote machine and access it locally using an SSH tunnel.

## 1. Prerequisites

Ensure `mlflow` is installed on both the remote server (training machine) and your local environment (recommended, but not strictly necessary if only viewing the UI).

```bash
pip install mlflow
```

## 2. Starting the MLflow Server (Remote Machine)

On your remote server (where the training happens), start the MLflow server. We will make it listen on `localhost` (127.0.0.1) or `0.0.0.0` depending on your security needs.

**Recommended:** Run on `0.0.0.0` if you are inside a private network, or `localhost` if you strictly want to use SSH tunneling for access.

```bash
# Example: Listen on port 5000, store data in a local sqlite database
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

*   `--backend-store-uri`: Where experiment metadata (params, metrics) is stored.
*   `--default-artifact-root`: Where large files (models, images) are stored.

> **Note:** To run this in the background, you can use `nohup`: 
> `nohup mlflow server ... > mlflow.log 2>&1 &`

## 3. Setting up SSH Tunnel (Local Machine)

To securely access the MLflow UI running on the remote server from your local computer, setup an SSH tunnel. This forwards a port on your local machine to the remote server's port.

**Command:**
```bash
ssh -N -f -L 5000:localhost:5000 user@remote_host_ip
```

### Breakdown:
*   `ssh`: The secure shell command.
*   `-N`: **Do not execute a remote command.** This is useful for just forwarding ports.
*   `-f`: **Background mode.** Requests ssh to go to background just before command execution.
*   `-L 5000:localhost:5000`: **Local port forwarding.** 
    *   The first `5000` is the port on your **local machine**.
    *   `localhost:5000` is the target on the **remote machine** (since the MLflow server is running on the remote machine's localhost/0.0.0.0 port 5000).
*   `user@remote_host_ip`: Your username and the IP address of the remote server.

## 4. Accessing the UI

1.  Ensure the SSH tunnel command ran successfully (it might ask for a password if you don't have key-based auth).
2.  Open your web browser on your local machine.
3.  Navigate to: [http://localhost:5000](http://localhost:5000)

You should now see the MLflow Tracking UI.

## 5. Integrating with Python Code

In your training script (`trainer.py`, `main.py`, etc.), set the tracking URI to point to the server.

```python
import mlflow

# If the script runs on the SAME server as the running mlflow server:
mlflow.set_tracking_uri("http://localhost:5000")

# Set the experiment name
mlflow.set_experiment("DeepSeek_Research")

# Example usage
with mlflow.start_run():
    mlflow.log_param("epochs", 10)
    mlflow.log_param("lr", 0.001)
    
    # Train your model...
    accuracy = 0.95
    
    mlflow.log_metric("accuracy", accuracy)
```
