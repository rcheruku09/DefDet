/* Copyright 2017 The MathWorks, Inc. */
#ifndef CNN_API_IMPL
#define CNN_API_IMPL

#include <map>
#include <vector>
class MWTensor;
class MWCNNLayer;
class MWTargetNetworkImpl;

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/SubTensor.h"
#include "arm_compute/core/SubTensorInfo.h"
using namespace arm_compute;

#define MW_LAYERS_TAP 0

#if MW_LAYERS_TAP
#define MW_INPUT_TAP 1
#define MW_CONV_TAP 1
#define MW_RELU_TAP 1
#define MW_NORM_TAP 1
#define MW_BNORM_TAP 1
#define MW_ADDITION_TAP 1
#define MW_POOL_TAP 1
#define MW_AVG_POOL_TAP 1
#define MW_FC_TAP 1
#define MW_SFMX_TAP 1
#define MW_DCONCATENATE_TAP 1
#else
#define MW_INPUT_TAP 0
#define MW_CONV_TAP 0
#define MW_RELU_TAP 0
#define MW_AVG_POOL_TAP 0
#define MW_NORM_TAP 0
#define MW_BNORM_TAP 0
#define MW_ADDITION_TAP 0
#define MW_POOL_TAP 0
#define MW_FC_TAP 0
#define MW_SFMX_TAP 0
#define MW_DCONCATENATE_TAP 0
#endif

class MWCNNLayerImpl {
  public:
    MWCNNLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* ntwk_impl);
    virtual ~MWCNNLayerImpl() {
    }
    virtual void predict() {
    }
    virtual void cleanup();
    virtual void allocate() {
    }
    virtual void postSetup() {
    }
    float* getData() {
        return eybNKlJCSDUvsznWynwK;
    }
    void setData(float* data);
    Tensor armTensor; // Ouput of the current layer

    MWCNNLayer* getLayer() {
        return jaqKGCwoANNDMHgAsehk;
    }
    Tensor* getprevLayerarmTensor(MWTensor*);

  protected:
    MWCNNLayer* jaqKGCwoANNDMHgAsehk;
    MWTargetNetworkImpl* kNsviQGMPdXzNMRixGWR;
    std::string getLinuxPath(const char* );
    float* eybNKlJCSDUvsznWynwK;
};

class MWInputLayerImpl : public MWCNNLayerImpl {

  private:
    int hnewnpwgzKmOdualajhn;
    std::vector<float> *TxNFOfYScyqGlEFFxbAv;
    float* m_inputImage;

    void createInputLayer(int, int, int, int, int, const char* avg_file_name, int );
    void loadAvg(const char*, int);
    void allocate();
    void predict();
    void cleanup();

  public:
    MWInputLayerImpl(MWCNNLayer* layer,
                     MWTargetNetworkImpl* ntwk_impl,
                     int,
                     int,
                     int,
                     int,
                     int,
                     const char* avg_file_name,
                     int);
    ~MWInputLayerImpl();
};

// ReLULayer
class MWReLULayerImpl : public MWCNNLayerImpl {
  private:
    NEActivationLayer m_actLayer;

    void createReLULayer(int);
    void allocate();
    void predict();

  public:
    MWReLULayerImpl(MWCNNLayer* , MWTargetNetworkImpl*, int, int );
    ~MWReLULayerImpl();
};


// CrossChannelNormalizationLayer
class MWNormLayerImpl : public MWCNNLayerImpl {
  private:
    NENormalizationLayer m_normLayer;

    void createNormLayer(unsigned, double, double, double, int);
    void allocate();
    void predict();

  public:
    MWNormLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, unsigned, double, double, double, int outbufIdx);
    ~MWNormLayerImpl();
};

// maxpoolingLayer
class MWMaxPoolingLayerImpl : public MWCNNLayerImpl {
  private:
    NEPoolingLayer m_maxPoolLayer;

    void createMaxPoolingLayer(int, int, int, int, int, int, int, int, int, const std::vector<int>& bufIndices );
    void allocate();
    void predict();

  public:
    MWMaxPoolingLayerImpl(MWCNNLayer *, MWTargetNetworkImpl*, int, int, int, int, int, int, int, int, bool, int, const std::vector<int>& bufIndices);
    ~MWMaxPoolingLayerImpl();
    float* getIndexData();
};

// FullyConnectedLayer
class MWFCLayerImpl : public MWCNNLayerImpl {
  private:
    NEFullyConnectedLayer m_fcLayer;
    Tensor m_fcLayerWgtTensor;
    Tensor m_fcLayerBiasTensor;

    void createFCLayer(int, const char*, const char*, int);
    void loadWeights(const char*, int);
    void loadBias(const char*);
    void allocate();
    void predict();
    void cleanup();

  public:
    MWFCLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int ,const char*, const char*, int);
    ~MWFCLayerImpl();
};

// SoftmaxLayer
class MWSoftmaxLayerImpl : public MWCNNLayerImpl {
  private:
    NESoftmaxLayer m_softmaxLayer;
    NEFlattenLayer m_flattenLayer;
    Tensor m_flattenArmTensor;
    void createSoftmaxLayer(int);
    void allocate();
    void predict();
    void cleanup();

  public:
    MWSoftmaxLayerImpl(MWCNNLayer* , MWTargetNetworkImpl*, int);
    ~MWSoftmaxLayerImpl();
};

// AvgPoolingLayer
class MWAvgPoolingLayerImpl : public MWCNNLayerImpl {
  private:
    NEPoolingLayer m_avgPoolLayer;
    void createAvgPoolingLayer(int, int, int, int, int, int);
    void allocate();
    void predict();

  public:
    MWAvgPoolingLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int, int, int, int, int, int, int);
    ~MWAvgPoolingLayerImpl();
};

class MWOutputLayerImpl : public MWCNNLayerImpl {
  private:
    float* m_outputData;
    Tensor* m_outputArmTensor;

    void createOutputLayer(int);
    void allocate();
    void predict();
    void cleanup();

  public:
    MWOutputLayerImpl(MWCNNLayer*, MWTargetNetworkImpl*, int);
    ~MWOutputLayerImpl();
};

#endif
