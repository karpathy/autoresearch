#!/usr/bin/env bash
# Spike (pi-autoresearch-serverless, plan 17a): the image and one smoke-test execution on Cloud Run's
# RTX PRO 6000, outside autoresearch, in the project that `/autoresearch init` provisioned; since
# Trial 3, the build, the repository and the job all in the GPU's region.
#
#   spike/cloud.sh setup          ar-build may push to the region's repository (once; idempotent)
#   spike/cloud.sh build          the image, by Cloud Build as ar-build, into that repository
#   spike/cloud.sh run            the job (re)defined on that image, one execution waited on, its log
#                                 and its conditions (the image's import, the start) saved
#   spike/cloud.sh logs EXECUTION an execution's log and conditions, again, into spike/results/
#
# The image's tag is a hash of what it is built from, so `run` finds what `build` made.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT=${PROJECT:-autoresearch-test-1}
BUCKET=${BUCKET:-autoresearch-$PROJECT}
REGION=${REGION:-europe-west4}                          # one of the regions with the GPU
REPOSITORY=${REPOSITORY:-autoresearch-europe-west4}     # the user's, in REGION
JOB=ar-spike-17a
TAG=$(cat pyproject.toml uv.lock .python-version prepare.py train.py spike/Containerfile spike/*.py spike/*.sh | shasum -a 256 | cut -c1-16)
IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPOSITORY/spike-17a:$TAG

logs() {
    mkdir -p spike/results
    gcloud logging read --project "$PROJECT" --order asc --limit 100000 --format 'value(textPayload)' \
        "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$JOB\" AND labels.\"run.googleapis.com/execution_name\"=\"$1\"" \
        > "spike/results/$1.log"
    gcloud run jobs executions describe "$1" --project "$PROJECT" --region "$REGION" \
        --format 'yaml(metadata.creationTimestamp,status.startTime,status.completionTime,status.conditions)' \
        > "spike/results/$1.conditions.yaml"
    echo "spike/results/$1.log: $(wc -l < "spike/results/$1.log") lines"
}

case "${1:-}" in
setup)
    gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" --project "$PROJECT" --location "$REGION" \
        --member "serviceAccount:ar-build@$PROJECT.iam.gserviceaccount.com" --role roles/artifactregistry.writer
    ;;
build)
    gcloud builds submit --project "$PROJECT" --region "$REGION" \
        --service-account "projects/$PROJECT/serviceAccounts/ar-build@$PROJECT.iam.gserviceaccount.com" \
        --gcs-source-staging-dir "gs://$BUCKET/sources/spike-17a" \
        --config spike/cloudbuild.yaml --substitutions "_IMAGE=$IMAGE" .
    ;;
run)
    gcloud run jobs deploy "$JOB" --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
        --gpu 1 --gpu-type nvidia-rtx-pro-6000 --no-gpu-zonal-redundancy --cpu 20 --memory 80Gi \
        --tasks 1 --max-retries 0 --task-timeout 30m \
        --service-account "ar-job@$PROJECT.iam.gserviceaccount.com" \
        --set-env-vars "AR_STEPS=${AR_STEPS:-953}"
    gcloud run jobs execute "$JOB" --project "$PROJECT" --region "$REGION" --wait || echo "the execution failed: $?"
    execution=$(gcloud run jobs executions list --project "$PROJECT" --region "$REGION" --job "$JOB" --limit 1 --format 'value(metadata.name)')
    for _ in 1 2 3 4 5 6; do # the last lines reach Cloud Logging a little after the task ends
        logs "$execution"
        grep -q '^=== done' "spike/results/$execution.log" && break
        sleep 20
    done
    grep -E '^(SPIKE_|train .*: exit|full_eval|=== )' "spike/results/$execution.log" || true
    grep -E 'message' "spike/results/$execution.conditions.yaml" || true
    ;;
logs)
    logs "$2"
    ;;
*)
    sed -n '2,12p' "$0"
    exit 2
    ;;
esac
