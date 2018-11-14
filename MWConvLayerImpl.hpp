
/* Copyright 2018 The MathWorks, Inc. */

#ifndef _MW_CONV_LAYER_IMPL
#define _MW_CONV_LAYER_IMPL

#include "MWCNNLayerImpl.hpp"

// Convolution2DWCNNLayer
class MWConvLayerImpl : public MWCNNLayerImpl {
    
  private:
    int GFienSVKLlDQuZeqAdLC;
    int GnxRkpzrPZimKtYYHSuG;
    int IwKnaBoXVubIRYcxEJLH;

    NEConvolutionLayer m_convLayer;            // used for Convolution/1st half of grouped conv
    NEConvolutionLayer m_convLayerSecondGroup; // used for 2nd half of grouped conv
    Tensor m_convLayerWgtTensor;
    Tensor m_convLayerBiasTensor;
    SubTensor* m_prevLayer1; // subtensor for current layer input (1st half in grp conv)
    SubTensor* m_prevLayer2; // subtensor for current layer input (2nd half in grp conv)
    SubTensor* m_curLayer1;  // subtensor for current layer output (1st half in grp conv)
    SubTensor* m_curLayer2;  // subtensor for current layer output (2nd half in grp conv)

    SubTensor* m_convLayerWgtMWTensor;  // subtensor for conv weights (1st half in grp conv)
    SubTensor* m_convLayerWgtTensor2;   // subtensor for conv weights (2nd half in grp conv)
    SubTensor* m_convLayerBiasMWTensor; // subtensor for conv bias (1st half in grp conv)
    SubTensor* m_convLayerBiasTensor2;  // subtensor for conv bias (2nd half in grp conv)
    void createConvLayer(int, int, int, int, int, int, int, int, const char*, const char*, int);

    void allocate();
    void predict();
    void cleanup();
    void loadWeights(const char*);
    void loadBias(const char*);

  public:
    MWConvLayerImpl(MWCNNLayer*,
                    MWTargetNetworkImpl*,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    const char*,
                    const char*,
                    int );
    ~MWConvLayerImpl();
};

#endif
