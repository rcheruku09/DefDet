#include "MWConvLayer.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include "cnn_exec.hpp"
CnnMain::CnnMain()
{
    this->numLayers = 12;
    this->targetImpl = 0;
    this->layers[0] = new MWInputLayer;
    this->layers[1] = new MWConvLayer;
    this->layers[2] = new MWReLULayer;
    this->layers[3] = new MWMaxPoolingLayer;
    this->layers[4] = new MWNormLayer;
    this->layers[5] = new MWConvLayer;
    this->layers[6] = new MWReLULayer;
    this->layers[7] = new MWMaxPoolingLayer;
    this->layers[8] = new MWFCLayer;
    this->layers[9] = new MWFCLayer;
    this->layers[10] = new MWSoftmaxLayer;
    this->layers[11] = new MWOutputLayer;
}
void CnnMain::presetup()
{
    this->targetImpl->preSetup(0, 0);
}
void CnnMain::postsetup()
{
    int32_T idx;
    this->targetImpl->postSetup(this->layers, this->numLayers);
    for (idx = 0; idx < 12; idx++) {
        this->layers[idx]->allocate();
    }
}
void CnnMain::setup()
{
    this->targetImpl = new MWTargetNetworkImpl;
    this->presetup();
    (dynamic_cast<MWInputLayer *>(this->layers[0]))->createInputLayer(this->targetImpl, 1, 128, 128, 1, 1, "./codegen/cnn_CnnMain_avg", -1);
    (dynamic_cast<MWConvLayer *>(this->layers[1]))->createConvLayer(this->targetImpl, this->layers[0]->getOutputTensor(0), 5, 5, 1, 20, 1, 1, 0, 0, 0, 0, 1, 1, 1, "./codegen/cnn_CnnMain_conv_1_w", "./codegen/cnn_CnnMain_conv_1_b", -1);
    (dynamic_cast<MWReLULayer *>(this->layers[2]))->createReLULayer(this->targetImpl, this->layers[1]->getOutputTensor(0), 1, -1);
    (dynamic_cast<MWMaxPoolingLayer *>(this->layers[3]))->createMaxPoolingLayer(this->targetImpl, this->layers[2]->getOutputTensor(0), 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, -1);
    (dynamic_cast<MWNormLayer *>(this->layers[4]))->createNormLayer(this->targetImpl, this->layers[3]->getOutputTensor(0), 5, 0.0001, 0.75, 1.0, -1);
    (dynamic_cast<MWConvLayer *>(this->layers[5]))->createConvLayer(this->targetImpl, this->layers[4]->getOutputTensor(0), 5, 5, 20, 20, 1, 1, 0, 0, 0, 0, 1, 1, 1, "./codegen/cnn_CnnMain_conv_2_w", "./codegen/cnn_CnnMain_conv_2_b", -1);
    (dynamic_cast<MWReLULayer *>(this->layers[6]))->createReLULayer(this->targetImpl, this->layers[5]->getOutputTensor(0), 1, -1);
    (dynamic_cast<MWMaxPoolingLayer *>(this->layers[7]))->createMaxPoolingLayer(this->targetImpl, this->layers[6]->getOutputTensor(0), 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, -1);
    (dynamic_cast<MWFCLayer *>(this->layers[8]))->createFCLayer(this->targetImpl, this->layers[7]->getOutputTensor(0), 16820, 512, "./codegen/cnn_CnnMain_fc_1_w", "./codegen/cnn_CnnMain_fc_1_b", -1);
    (dynamic_cast<MWFCLayer *>(this->layers[9]))->createFCLayer(this->targetImpl, this->layers[8]->getOutputTensor(0), 512, 2, "./codegen/cnn_CnnMain_fc_2_w", "./codegen/cnn_CnnMain_fc_2_b", -1);
    (dynamic_cast<MWSoftmaxLayer *>(this->layers[10]))->createSoftmaxLayer(this->targetImpl, this->layers[9]->getOutputTensor(0), -1);
    (dynamic_cast<MWOutputLayer *>(this->layers[11]))->createOutputLayer(this->targetImpl, this->layers[10]->getOutputTensor(0), -1);
    this->postsetup();
    this->inputData = this->layers[0]->getData(0);
    this->outputData = this->layers[11]->getData(0);
}
void CnnMain::predict()
{
    int32_T idx;
    for (idx = 0; idx < 12; idx++) {
        this->layers[idx]->predict();
    }
}
void CnnMain::cleanup()
{
    int32_T idx;
    for (idx = 0; idx < 12; idx++) {
        this->layers[idx]->cleanup();
    }
    if (this->targetImpl) {
        this->targetImpl->cleanup();
    }
}
real32_T *CnnMain::getLayerOutput(int32_T layerIndex, int32_T portIndex)
{
    return this->layers[layerIndex]->getData(portIndex);
}
CnnMain::~CnnMain()
{
    int32_T idx;
    this->cleanup();
    for (idx = 0; idx < 12; idx++) {
        delete this->layers[idx];    }
    if (this->targetImpl) {
        delete this->targetImpl;    }
}
