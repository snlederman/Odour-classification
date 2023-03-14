# build image from Dockerfile
docker build . --tag odor_classification

# see that you have the image installed
docker image list

# run image to generate container
docker run -it odor_classification

# see that you have the container activated
docker container list

# see all containers
docker container list --all

# remove unused containers and images
docker image prune --all

# remove specific container or image
docker image remove <name>
docker container remove <name>