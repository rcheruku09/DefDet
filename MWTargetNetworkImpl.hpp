/* Copyright 2017 The MathWorks, Inc. */
#ifndef CNN_TARGET_NTWK_IMPL
#define CNN_TARGET_NTWK_IMPL

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "cnn_api.hpp"

class MWTargetNetworkImpl {
  public:
    MWTargetNetworkImpl() {
    }
    ~MWTargetNetworkImpl() {
    }
    void preSetup(int, int);
    void postSetup(MWCNNLayer* layers[],int numLayers);

    void setWorkSpaceSize(size_t); // Set the workspace size of this layer and previous layers
    size_t* getWorkSpaceSize();    // Get the workspace size of this layer and previous layers
    float* getWorkSpace();         // Get the workspace buffer in GPU memory
    void cleanup();
    void createWorkSpace(float**); // Create the workspace needed for this layer

    int numBufs;
    std::vector<float *> memBuffer;
};
#endif
