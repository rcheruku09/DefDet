#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include "MWCNNLayerImpl.hpp"
 void MWTargetNetworkImpl::preSetup(int BufSize, int numBufsToAlloc) { numBufs 
= numBufsToAlloc; } void MWTargetNetworkImpl::postSetup(MWCNNLayer* 
layers[],int numLayers) { int maxBufSize = -1; for(int i = 0; i < numLayers; 
i++) { if(layers[i]->getImpl() != NULL) { maxBufSize = std::max(maxBufSize, 
(int)((layers[i]->getImpl()->armTensor.info()->total_size()) / 4)); } } for(int 
i = 0; i < numBufs; i++) { float *memPtr ; memPtr = 
(float*)calloc(maxBufSize,sizeof(float)); memBuffer.push_back(memPtr); } } void 
MWTargetNetworkImpl::createWorkSpace(float** xHViLEwTujGGrPZZgmbF) { } void 
MWTargetNetworkImpl::setWorkSpaceSize(size_t wss) { } size_t* 
MWTargetNetworkImpl::getWorkSpaceSize() { return NULL; } float* 
MWTargetNetworkImpl::getWorkSpace() { return NULL; } void 
MWTargetNetworkImpl::cleanup() { for(int i = 0; i < numBufs; i++) { float 
*memPtr = memBuffer[i]; free(memPtr); } }