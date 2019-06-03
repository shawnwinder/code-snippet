# create a new container from docker image with GPU support
docker run --hostname Detectron --name Detectron -w /detectron --runtime=nvidia -it caffe2:objectdect bash
