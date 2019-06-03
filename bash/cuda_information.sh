# check cuda version
cat /usr/local/cuda/version.txt 

# check cudnn version (just for reference)
cat /path/to/cudnn_installation/include/cudnn.h | grep CUDNN_MAJOR -A 2
# e.g: cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
# e.g: cat /home/zhibin/qzhong/thirdparty/cudnn/include/cudnn.h | grep CUDNN_MAJOR -A 2


