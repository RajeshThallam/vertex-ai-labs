```
# set variables
PROJECT_ID=$(gcloud config list --format 'value(core.project)')
REGION=us-central1
TRAIN_IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/vertex-rapidsai/distributed-xgboost-dask

STAGING_BUCKET_NAME=cloud-ai-platform-2f444b6a-a742-444b-b91a-c7519f51bd77
TRAIN_FILES=gs://rthallam-demo-project/rapids-on-gcp/data/latest/a/higgs_00.csv

# create artifact registry repository
gcloud artifacts repositories create vertex-rapidsai \
 --repository-format=docker \
 --location=${REGION} \
 --description="Vertex AI RAPIDS"

# build training image and push to Artifact Registry
gcloud builds submit --tag $TRAIN_IMAGE_URI --timeout=3600 .

# create training job config for multi-node multi-gpu dask job
date_now=$(date "+%Y%m%d-%H%M%S")

cat << EOF > ./dask-xgb-multi-node.yml

baseOutputDirectory:
    outputUriPrefix: gs://${STAGING_BUCKET_NAME}/rapidsai/distributed-xgboost-dask/${date_now}/
workerPoolSpecs:
  -
    machineSpec:
      machineType: n1-highmem-4
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: ${TRAIN_IMAGE_URI}
      args:
      - --train-files=${TRAIN_FILES}
      - --rmm-pool-size=4G
      - --num-workers=4
      - --nthreads=4
  -
    machineSpec:
      machineType: n1-highmem-4
      acceleratorType: NVIDIA_TESLA_T4
      acceleratorCount: 2
    replicaCount: 4
    containerSpec:
      imageUri: ${TRAIN_IMAGE_URI}
      args:
      - --train-files=${TRAIN_FILES}
      - --rmm-pool-size=4G
      - --num-workers=4
      - --nthreads=4
EOF

# submit vertex ai custom training job
gcloud beta ai custom-jobs create \
  --display-name=rapids-dstrbtd-xgb-dask-multi-node \
  --region=$REGION \
  --project=$PROJECT_ID \
  --config=dask-xgb-multi-node.yml
```