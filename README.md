# Vertex AI Labs
---

**Hands-on labs introducing GCP Vertex AI features**

- Vertex Notebooks
- Vertex AI Training
    - Using pre-built and custom containers
    - Hyperparameter tuning
    - Distributed Training
- Vertex AI Predictions
    - using pre-built and custom containers
- Vertex Tensorboard
- Vertex ML Metadata


## Environment Setup

The following section describes requirements for setting up a GCP environment required for the workshop. Note that we have provided example [Terraform](https://www.terraform.io/) scripts to automate the process. You can find the scripts and the instructions in the `env-setup` folder.

### GCP Project

Ideally each participant should have their own sandbox GCP project. If this is not feasible, multiple participants can share a single project but other resources used during the labs like GCS buckets should be created for each participant. See below for details. You need to be a project owner to complete some of the setup steps.

### Cloud APIs

The following APIs need to be enabled in the project:

- compute.googleapis.com
- iam.googleapis.com
- container.googleapis.com
- artifactregistry.googleapis.com
- cloudresourcemanager.googleapis.com
- cloudtrace.googleapis.com
- iamcredentials.googleapis.com
- monitoring.googleapis.com
- logging.googleapis.com
- notebooks.googleapis.com
- aiplatform.googleapis.com
- dataflow.googleapis.com
- bigquery.googleapis.com
- cloudbuild.googleapis.com
- bigquerydatatransfer.googleapis.com

### GCP Region

Note that some services used during the notebook are only available in a limited number of regions. We recommend using `us-central1`.

### Service accounts

Two service accounts must be created in the project.

#### Vertex AI training service account

This account will be used by Vertex Training service. The account needs the following permissions:

- storage.admin
- aiplatform.user
- bigquery.admin

The account email should be 

`training-sa@{PROJECT_ID}.iam.gserviceaccount.com`

#### Vertex AI pipelines service account

This account will be used by Vertex Pipelines service. The account needs the following permissions:

- storage.admin
- aiplatform.user
- bigquery.admin

The account email should be 

`pipelines-sa@{PROJECT_ID}.iam.gserviceaccount.com`

### GCS buckets

Each participant should have their own regional GCS bucket. The bucket should be created in the GCP region that will be used during the workshop. The bucket name should use the following naming convention

`gs://{PREFIX}-bucket`

The goal of the prefix is too avoid conflicts between participants as such it should be unique for each participant. **The prefix should start with a letter and include letters and digits only**

The workshop notebooks assume this naming convention.


### Vertex AI Notebook

Each participant should have any instance of Vertex AI Notebook. The instances can be pre-created or can be created during the workshop.

The instance should be configured as follows:

- Machine type: **n1-standard-4**
- Optionally a T4 GPU can be added to the machine configuration if participants want to experiment with GPUs
- Image family: **tf-2-4-cpu** or **tf-2-4-cu110** (if using GPUs)
- Configured with the default compute engine service account

#### Vertex AI Notebook setup

The following setup steps will be performed during the workshop, individually by each of the participants.

In JupyterLab, open a terminal and:

#####  Install the required Python packages

```
pip install --user google-cloud-aiplatform
pip install --user kfp
pip install --user google-cloud-pipeline-components
pip install --user google-cloud-bigquery-datatransfer
```

##### Create a Tensorboard instance

Each participant will use their own Vertex Tensorboard instance.

```
PROJECT=jk-vertex-workshop
REGION=us-central1
PREFIX=jkvw
DISPLAY_NAME=${PREFIX}-tensorboard

gcloud beta ai tensorboards create --display-name $DISPLAY_NAME \
  --project $PROJECT --region $REGION

```

Save the tensorboard name returned by the command as it will be needed when configuring the workshop notebooks.

You can get it at any time by listing Tensorboards in the project

```
gcloud beta ai tensorboards list \
  --project $PROJECT --region $REGION
```

##### Clone this repo
```
git clone https://github.com/jarokaz/vertex-ai-workshop
```


#### References:

- https://github.com/jarokaz/vertex-ai-workshop/
