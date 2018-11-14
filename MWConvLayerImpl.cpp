#include "MWConvLayerImpl.hpp"
#include "MWConvLayer.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include <cassert>
#include <cstring>
#include <stdio.h>
#if MW_CONV_TAP
 extern void mw_interm_tap(float*, int, int); extern int tap_count;
#endif
 MWConvLayerImpl::MWConvLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int filt_H, int filt_W, int numGrps, int numChnls, int numFilts, int 
PmFfARVzoHVAYkfpuvqK, int QjgQHaUACFNSteMrRtRj, int MNuwXDSoGEYeABeVTwOh, int 
MIBnYCbKBdUrlfqlHdoo, int NDjzAZSYJuWymuKDNZYB, int NZjOkZPwLzQsdEVkwMcX, 
int BRSPqxNffoBYKqpSVHne, int CZNYmBcNFSZWvaCklqeM, const char* 
sxuOMwKXOKfuExclRaSe, const char* cQBKlCKXxecGPJrXBXdk, int outbufIdx) : 
MWCNNLayerImpl(layer, ntwk_impl) , GFienSVKLlDQuZeqAdLC(filt_H) , 
GnxRkpzrPZimKtYYHSuG(filt_W) , IwKnaBoXVubIRYcxEJLH(numGrps) { 
createConvLayer(PmFfARVzoHVAYkfpuvqK, QjgQHaUACFNSteMrRtRj, MNuwXDSoGEYeABeVTwOh, 
MIBnYCbKBdUrlfqlHdoo, NDjzAZSYJuWymuKDNZYB, NZjOkZPwLzQsdEVkwMcX, 
BRSPqxNffoBYKqpSVHne,CZNYmBcNFSZWvaCklqeM, sxuOMwKXOKfuExclRaSe, 
cQBKlCKXxecGPJrXBXdk,outbufIdx); } MWConvLayerImpl::~MWConvLayerImpl() { } void 
MWConvLayerImpl::createConvLayer(int PmFfARVzoHVAYkfpuvqK, int 
QjgQHaUACFNSteMrRtRj, int MNuwXDSoGEYeABeVTwOh, int MIBnYCbKBdUrlfqlHdoo, int 
NDjzAZSYJuWymuKDNZYB, int NZjOkZPwLzQsdEVkwMcX, int 
BRSPqxNffoBYKqpSVHne, int CZNYmBcNFSZWvaCklqeM, const char* 
sxuOMwKXOKfuExclRaSe, const char* cQBKlCKXxecGPJrXBXdk,  int outbufIdx) { if 
((BRSPqxNffoBYKqpSVHne != 1) || (CZNYmBcNFSZWvaCklqeM != 1)){ 
printf("Dilated Convolution is not supported in arm-compute platform"); throw 
std::runtime_error("Unsupported Dilation Factor"); } MWConvLayer* convLayer = 
static_cast<MWConvLayer*>(getLayer()); MWTensor* ipTensor = 
convLayer->getInputTensor(0); MWTensor* opTensor = 
convLayer->getOutputTensor(0); Tensor* prevLayerarmTensor = 
getprevLayerarmTensor(ipTensor); m_convLayerWgtTensor.allocator()->init( 
TensorInfo(TensorShape((long unsigned int)GnxRkpzrPZimKtYYHSuG, (long 
unsigned int)GFienSVKLlDQuZeqAdLC, (long unsigned 
int)ipTensor->getChannels() / IwKnaBoXVubIRYcxEJLH, (long unsigned 
int)opTensor->getChannels()), 1, DataType::F32, 4)); 
m_convLayerBiasTensor.allocator()->init( TensorInfo(TensorShape((long unsigned 
int)opTensor->getChannels()), 1, DataType::F32, 4)); 
armTensor.allocator()->init(TensorInfo(TensorShape((long unsigned 
int)opTensor->getWidth(), (long unsigned int)opTensor->getHeight(), (long 
unsigned int)opTensor->getChannels()), 1, DataType::F32, 4)); 
getLayer()->getOutputTensor(0)->setopBufIndex(outbufIdx); if 
(IwKnaBoXVubIRYcxEJLH != 1) { m_prevLayer1 = new SubTensor( prevLayerarmTensor, 
TensorShape((long unsigned int)ipTensor->getHeight(), (long unsigned 
int)ipTensor->getWidth(), (long unsigned int)(ipTensor->getChannels() / 
IwKnaBoXVubIRYcxEJLH), (long unsigned int)ipTensor->getBatchSize()), 
Coordinates()); m_prevLayer2 = new SubTensor( prevLayerarmTensor, 
TensorShape((long unsigned int)ipTensor->getHeight(), (long unsigned 
int)ipTensor->getWidth(), (long unsigned int)(ipTensor->getChannels() / 
IwKnaBoXVubIRYcxEJLH), (long unsigned int)ipTensor->getBatchSize()), 
Coordinates(0, 0, ipTensor->getChannels() / IwKnaBoXVubIRYcxEJLH)); m_curLayer1 
= new SubTensor( &armTensor, TensorShape((long unsigned 
int)opTensor->getWidth(), (long unsigned int)opTensor->getHeight(), (long 
unsigned int)(opTensor->getChannels() / IwKnaBoXVubIRYcxEJLH), (long unsigned 
int)opTensor->getBatchSize()), Coordinates()); m_curLayer2 = new SubTensor( 
&armTensor, TensorShape((long unsigned int)opTensor->getWidth(), (long unsigned 
int)opTensor->getHeight(), (long unsigned int)(opTensor->getChannels() / 
IwKnaBoXVubIRYcxEJLH), (long unsigned int)opTensor->getBatchSize()), 
Coordinates(0, 0, opTensor->getChannels() / IwKnaBoXVubIRYcxEJLH)); 
m_convLayerWgtMWTensor = new SubTensor( &m_convLayerWgtTensor, 
TensorShape((long unsigned int)GFienSVKLlDQuZeqAdLC, (long unsigned 
int)GnxRkpzrPZimKtYYHSuG, (long unsigned int)(ipTensor->getChannels() / 
IwKnaBoXVubIRYcxEJLH), (long unsigned int)(opTensor->getChannels() / 
IwKnaBoXVubIRYcxEJLH)), Coordinates()); m_convLayerWgtTensor2 = new SubTensor( 
&m_convLayerWgtTensor, TensorShape((long unsigned int)GFienSVKLlDQuZeqAdLC, 
(long unsigned int)GnxRkpzrPZimKtYYHSuG, (long unsigned 
int)(ipTensor->getChannels() / IwKnaBoXVubIRYcxEJLH), (long unsigned 
int)(opTensor->getChannels() / IwKnaBoXVubIRYcxEJLH)), Coordinates(0, 0, 0, 
opTensor->getChannels() / IwKnaBoXVubIRYcxEJLH)); m_convLayerBiasMWTensor = new 
SubTensor( &m_convLayerBiasTensor, TensorShape((long unsigned 
int)(opTensor->getChannels() / IwKnaBoXVubIRYcxEJLH)), Coordinates()); 
m_convLayerBiasTensor2 = new SubTensor( &m_convLayerBiasTensor, 
TensorShape((long unsigned int)(opTensor->getChannels() / 
IwKnaBoXVubIRYcxEJLH)), Coordinates(opTensor->getChannels() / 
IwKnaBoXVubIRYcxEJLH)); m_convLayer.configure( m_prevLayer1, 
m_convLayerWgtMWTensor, m_convLayerBiasMWTensor, m_curLayer1, 
PadStrideInfo(QjgQHaUACFNSteMrRtRj, PmFfARVzoHVAYkfpuvqK, NDjzAZSYJuWymuKDNZYB, 
NZjOkZPwLzQsdEVkwMcX, MNuwXDSoGEYeABeVTwOh, MIBnYCbKBdUrlfqlHdoo, 
DimensionRoundingType::FLOOR), WeightsInfo(false, (long unsigned 
int)GnxRkpzrPZimKtYYHSuG, (long unsigned int)GFienSVKLlDQuZeqAdLC, (long 
unsigned int)opTensor->getChannels())); m_convLayerSecondGroup.configure( 
m_prevLayer2, m_convLayerWgtTensor2, m_convLayerBiasTensor2, m_curLayer2, 
PadStrideInfo(QjgQHaUACFNSteMrRtRj, PmFfARVzoHVAYkfpuvqK, NDjzAZSYJuWymuKDNZYB, 
NZjOkZPwLzQsdEVkwMcX, MNuwXDSoGEYeABeVTwOh, MIBnYCbKBdUrlfqlHdoo, 
DimensionRoundingType::FLOOR), WeightsInfo(false, (long unsigned 
int)GnxRkpzrPZimKtYYHSuG, (long unsigned int)GFienSVKLlDQuZeqAdLC, (long 
unsigned int)opTensor->getChannels())); } else { m_convLayer.configure( 
prevLayerarmTensor, &m_convLayerWgtTensor, &m_convLayerBiasTensor, &armTensor, 
PadStrideInfo(QjgQHaUACFNSteMrRtRj, PmFfARVzoHVAYkfpuvqK, NDjzAZSYJuWymuKDNZYB, 
NZjOkZPwLzQsdEVkwMcX, MNuwXDSoGEYeABeVTwOh, MIBnYCbKBdUrlfqlHdoo, 
DimensionRoundingType::FLOOR), WeightsInfo(false, (long unsigned 
int)GnxRkpzrPZimKtYYHSuG, (long unsigned int)GFienSVKLlDQuZeqAdLC, (long 
unsigned int)opTensor->getChannels())); } loadWeights(sxuOMwKXOKfuExclRaSe); 
loadBias(cQBKlCKXxecGPJrXBXdk); return; } void MWConvLayerImpl::allocate() { 
MWTensor* opTensor = getLayer()->getOutputTensor(0); 
if(opTensor->getopBufIndex() < 0) { armTensor.allocator()->allocate(); } else { 
armTensor.allocator()->import_memory(Memory((uint8_t 
*)kNsviQGMPdXzNMRixGWR->memBuffer[opTensor->getopBufIndex()])); } 
setData((float*)armTensor.buffer()); MWConvLayer* convLayer = 
static_cast<MWConvLayer*>(getLayer()); 
convLayer->getOutputTensor()->setData((float*)armTensor.buffer()); } void 
MWConvLayerImpl::predict() { MWConvLayer* convLayer = 
static_cast<MWConvLayer*>(getLayer()); MWTensor* opTensor = 
convLayer->getOutputTensor(0); if (IwKnaBoXVubIRYcxEJLH == 1) { 
m_convLayer.run(); } else { m_convLayer.run(); m_convLayerSecondGroup.run(); }
#if MW_CONV_TAP
 mw_interm_tap((float*)armTensor.buffer(), opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(), tap_count++);
#endif
 return; } void MWConvLayerImpl::cleanup() { if (IwKnaBoXVubIRYcxEJLH != 1) { 
delete m_prevLayer1; delete m_prevLayer2; delete m_curLayer1; delete 
m_curLayer2; delete m_convLayerWgtMWTensor; delete m_convLayerWgtTensor2; 
delete m_convLayerBiasMWTensor; delete m_convLayerBiasTensor2; } 
MWCNNLayerImpl::cleanup(); return; } void MWConvLayerImpl::loadWeights(const 
char* fSKMHAqIghbYYgyIpNDw) { MWConvLayer* convLayer = 
static_cast<MWConvLayer*>(getLayer()); MWTensor* ipTensor = 
convLayer->getInputTensor(); MWTensor* opTensor = convLayer->getOutputTensor(); 
float* puSFZkRJmyuFPfQRswDK = (float*)calloc(ipTensor->getChannels() / 
IwKnaBoXVubIRYcxEJLH * opTensor->getChannels() * GFienSVKLlDQuZeqAdLC * 
GnxRkpzrPZimKtYYHSuG, sizeof(float)); size_t retVal; std::string fileString = 
getLinuxPath(fSKMHAqIghbYYgyIpNDw); FILE* fxxCPKTclxXPxrdMAkwi = 
MWCNNLayer::openBinaryFile(fileString.c_str()); int kkqTyvjYvRFtTOyQUwrF = 
ipTensor->getChannels() / IwKnaBoXVubIRYcxEJLH * opTensor->getChannels() * 
GFienSVKLlDQuZeqAdLC * GnxRkpzrPZimKtYYHSuG;  retVal = 
fread(puSFZkRJmyuFPfQRswDK, sizeof(float), kkqTyvjYvRFtTOyQUwrF, fxxCPKTclxXPxrdMAkwi); if (retVal 
!= (size_t)kkqTyvjYvRFtTOyQUwrF) { 
printf("MWConvLayer::loadWeights - File read Failed\n"); } if 
(GFienSVKLlDQuZeqAdLC != 1 && GnxRkpzrPZimKtYYHSuG != 1) { float* 
oYbqYsqgVhrUzFEKbBbR = (float*)malloc(sizeof(float) * GFienSVKLlDQuZeqAdLC * 
GnxRkpzrPZimKtYYHSuG); for (int k = 0; k < kkqTyvjYvRFtTOyQUwrF / 
GFienSVKLlDQuZeqAdLC / GnxRkpzrPZimKtYYHSuG; k++) { for (int i = 0; i < 
GFienSVKLlDQuZeqAdLC * GnxRkpzrPZimKtYYHSuG; i++) { oYbqYsqgVhrUzFEKbBbR[i] = 
puSFZkRJmyuFPfQRswDK[k * GFienSVKLlDQuZeqAdLC * GnxRkpzrPZimKtYYHSuG + i]; } for 
(int j = 0; j < GFienSVKLlDQuZeqAdLC; j++) for (int i = 0; i < 
GnxRkpzrPZimKtYYHSuG; i++) { puSFZkRJmyuFPfQRswDK[k * GFienSVKLlDQuZeqAdLC * 
GnxRkpzrPZimKtYYHSuG + j * GnxRkpzrPZimKtYYHSuG + i] = oYbqYsqgVhrUzFEKbBbR[j + i 
* GFienSVKLlDQuZeqAdLC]; } } free(oYbqYsqgVhrUzFEKbBbR); } 
m_convLayerWgtTensor.allocator()->allocate(); std::copy_n((unsigned 
char*)puSFZkRJmyuFPfQRswDK, kkqTyvjYvRFtTOyQUwrF * sizeof(float), (unsigned 
char*)m_convLayerWgtTensor.buffer()); fclose(fxxCPKTclxXPxrdMAkwi); 
free(puSFZkRJmyuFPfQRswDK); return; } void MWConvLayerImpl::loadBias(const char* 
fSKMHAqIghbYYgyIpNDw) { size_t retVal; MWConvLayer* convLayer = 
static_cast<MWConvLayer*>(getLayer()); MWTensor* opTensor = 
convLayer->getOutputTensor(); float* aLsOwwcceEmRSYzllBNs = 
(float*)calloc(opTensor->getChannels(), sizeof(float)); std::string fileString 
= getLinuxPath(fSKMHAqIghbYYgyIpNDw); FILE* fxxCPKTclxXPxrdMAkwi = 
MWCNNLayer::openBinaryFile(fileString.c_str()); int kkqTyvjYvRFtTOyQUwrF = 
opTensor->getChannels();  retVal = fread(aLsOwwcceEmRSYzllBNs, sizeof(float), 
kkqTyvjYvRFtTOyQUwrF, fxxCPKTclxXPxrdMAkwi); if (retVal != (size_t)kkqTyvjYvRFtTOyQUwrF) { 
printf("MWConvLayer::loadBias - File read Failed\n"); } 
m_convLayerBiasTensor.allocator()->allocate(); std::copy_n((unsigned 
char*)aLsOwwcceEmRSYzllBNs, kkqTyvjYvRFtTOyQUwrF * sizeof(float), (unsigned 
char*)m_convLayerBiasTensor.buffer()); free(aLsOwwcceEmRSYzllBNs); 
fclose(fxxCPKTclxXPxrdMAkwi); return; }