REGISTRY="gcr.io/salesforce-research-internal"
REPO_BASE="hguo-scratchpad"
DOCKER_TAG="IMAGE_TAG"

docker run -ti --rm \
    ${REGISTRY}/${REPO_BASE}:${DOCKER_TAG}
