#!/usr/bin/env bash
# Spike (pi-autoresearch-serverless, plan 17a): the image and one smoke-test execution on Cloud Run's
# RTX PRO 6000, outside autoresearch, in the project that `/autoresearch init` provisioned.
#
#   spike/cloud.sh build          the image, by Cloud Build as ar-build, into the project's repository
#   spike/cloud.sh run            the job (re)defined on that image, one execution waited on, its log saved
#   spike/cloud.sh logs EXECUTION an execution's log, again, into spike/results/EXECUTION.log
#
# The image's tag is a hash of what it is built from, so `run` finds what `build` made.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT=${PROJECT:-autoresearch-test-1}
BUCKET=${BUCKET:-autoresearch-$PROJECT}
BUILD_REGION=${BUILD_REGION:-europe-north1}  # the repository's and the builds' region (cloud.json)
RUN_REGION=${RUN_REGION:-europe-west4}       # the nearest region with the GPU
JOB=ar-spike-17a
TAG=$(cat pyproject.toml uv.lock .python-version prepare.py train.py spike/Containerfile spike/*.py spike/*.sh | shasum -a 256 | cut -c1-16)
IMAGE=$BUILD_REGION-docker.pkg.dev/$PROJECT/autoresearch/spike-17a:$TAG

logs() {
    mkdir -p spike/results
    gcloud logging read --project "$PROJECT" --order asc --limit 100000 --format 'value(textPayload)' \
        "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$JOB\" AND labels.\"run.googleapis.com/execution_name\"=\"$1\"" \
        > "spike/results/$1.log"
    echo "spike/results/$1.log: $(wc -l < "spike/results/$1.log") lines"
}

case "${1:-}" in
build)
    gcloud builds submit --project "$PROJECT" --region "$BUILD_REGION" \
        --service-account "projects/$PROJECT/serviceAccounts/ar-build@$PROJECT.iam.gserviceaccount.com" \
        --gcs-source-staging-dir "gs://$BUCKET/sources/spike-17a" \
        --config spike/cloudbuild.yaml --substitutions "_IMAGE=$IMAGE" .
    ;;
run)
    gcloud run jobs deploy "$JOB" --project "$PROJECT" --region "$RUN_REGION" --image "$IMAGE" \
        --gpu 1 --gpu-type nvidia-rtx-pro-6000 --no-gpu-zonal-redundancy --cpu 20 --memory 80Gi \
        --tasks 1 --max-retries 0 --task-timeout 30m \
        --service-account "ar-job@$PROJECT.iam.gserviceaccount.com" \
        --set-env-vars "AR_TIME_BUDGET=${AR_TIME_BUDGET:-60}"
    gcloud run jobs execute "$JOB" --project "$PROJECT" --region "$RUN_REGION" --wait || echo "the execution failed: $?"
    execution=$(gcloud run jobs executions list --project "$PROJECT" --region "$RUN_REGION" --job "$JOB" --limit 1 --format 'value(metadata.name)')
    for _ in 1 2 3 4 5 6; do # the last lines reach Cloud Logging a little after the task ends
        logs "$execution"
        grep -q '^=== done' "spike/results/$execution.log" && break
        sleep 20
    done
    grep -E '^(SPIKE_|train .*: exit|=== )' "spike/results/$execution.log" || true
    ;;
logs)
    logs "$2"
    ;;
*)
    sed -n '2,9p' "$0"
    exit 2
    ;;
esac
