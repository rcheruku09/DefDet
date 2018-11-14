#ifndef CNN_EXEC_HPP_
#define CNN_EXEC_HPP_
#include "rtwtypes.h"
#include "stdio.h"
#include "string.h"
#include "cnn_api.hpp"

class CnnMain
{
  public:
    int32_T batchSize;
    int32_T numLayers;
    real32_T *inputData;
    real32_T *outputData;
    MWCNNLayer *layers[12];
  private:
    MWTargetNetworkImpl *targetImpl;
  public:
    CnnMain();
    void presetup();
    void postsetup();
    void setup();
    void predict();
    void cleanup();
    real32_T *getLayerOutput(int32_T layerIndex, int32_T portIndex);
    ~CnnMain();
};
#endif
