#include "MWCNNLayerImpl.hpp"
#include "MWTargetNetworkImpl.hpp"
#include "cnn_api.hpp"
#include <cassert>
#include <cstring>
#include <stdio.h>
#include <omp.h>
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Memory.h"
 using namespace arm_compute; MWCNNLayerImpl::MWCNNLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl) : jaqKGCwoANNDMHgAsehk(layer) , 
kNsviQGMPdXzNMRixGWR(ntwk_impl) { } void MWCNNLayerImpl::setData(float* data) { 
eybNKlJCSDUvsznWynwK = data; } void MWCNNLayerImpl::cleanup() { 
if(jaqKGCwoANNDMHgAsehk->getOutputTensor()->getopBufIndex() < 0) 
armTensor.allocator()->free();  } std::string 
MWCNNLayerImpl::getLinuxPath(const char* fileName) { std::string 
fileS(fileName); std::string key ("\\"); std::size_t found = 0; while(found != 
std::string::npos){ found = fileS.rfind(key); if (found!=std::string::npos) 
fileS.replace (found,key.length(),"/"); } return fileS; } Tensor* 
MWCNNLayerImpl::getprevLayerarmTensor(MWTensor* ipTensor) {  if 
(ipTensor->getOwner()->getImpl() == NULL) { return 
&ipTensor->getOwner()->getInputTensor()->getOwner()->getImpl()->armTensor; } 
else { return &ipTensor->getOwner()->getImpl()->armTensor; } } 
MWInputLayerImpl::MWInputLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int jhFUWlztBndwjbXwYNaJ, int gTcJMwtYuwiqqUmqvKhT, int oJUVMnJggjhEdQLWzIUC, int 
euppfEoiaoCTcVgRPVhA, int vjDFlBZzKvbpPseAtMBP, const char* avg_file_name, int outbufIdx) 
: MWCNNLayerImpl(layer, ntwk_impl) { createInputLayer(jhFUWlztBndwjbXwYNaJ, 
gTcJMwtYuwiqqUmqvKhT, oJUVMnJggjhEdQLWzIUC, euppfEoiaoCTcVgRPVhA, vjDFlBZzKvbpPseAtMBP, avg_file_name, 
outbufIdx); } MWInputLayerImpl::~MWInputLayerImpl() { } int tap_count = 0; void 
mw_interm_tap(float* inp, int size, int count) { FILE* fp; int i; char str[500];
#define TXT_FILE 1
#if TXT_FILE
 sprintf(str, "taps/mw_interm_tap_%d.txt", count); fp = fopen(str, "w"); for (i 
= 0; i < size; i++) { fprintf(fp, "%f\n", inp[i]); }
#else
 sprintf(str, "taps/mw_interm_tap_%d.bin", count); fp = fopen(str, "wb"); 
fwrite(inp, 4, size, fp);
#endif
 fclose(fp); } void MWInputLayerImpl::createInputLayer(int jhFUWlztBndwjbXwYNaJ, int 
gTcJMwtYuwiqqUmqvKhT, int oJUVMnJggjhEdQLWzIUC, int euppfEoiaoCTcVgRPVhA, int vjDFlBZzKvbpPseAtMBP, const 
char* avg_file_name,  int outbufIdx) { MWInputLayer* inpLayer = 
static_cast<MWInputLayer*>(getLayer()); hnewnpwgzKmOdualajhn = 
vjDFlBZzKvbpPseAtMBP; m_inputImage = (float*)calloc(jhFUWlztBndwjbXwYNaJ * euppfEoiaoCTcVgRPVhA * 
gTcJMwtYuwiqqUmqvKhT * oJUVMnJggjhEdQLWzIUC, sizeof(float)); setData(m_inputImage); 
armTensor.allocator()->init( TensorInfo(TensorShape((long unsigned 
int)oJUVMnJggjhEdQLWzIUC, (long unsigned int)gTcJMwtYuwiqqUmqvKhT, (long unsigned 
int)euppfEoiaoCTcVgRPVhA), 1, DataType::F32, 4)); 
getLayer()->getOutputTensor(0)->setopBufIndex(outbufIdx); int kkqTyvjYvRFtTOyQUwrF = 
euppfEoiaoCTcVgRPVhA * gTcJMwtYuwiqqUmqvKhT * oJUVMnJggjhEdQLWzIUC; if( hnewnpwgzKmOdualajhn ) { 
loadAvg(avg_file_name, kkqTyvjYvRFtTOyQUwrF); } return; } void 
MWInputLayerImpl::loadAvg(const char* fSKMHAqIghbYYgyIpNDw, int kkqTyvjYvRFtTOyQUwrF) 
{ size_t retVal; std::string fileString = getLinuxPath(fSKMHAqIghbYYgyIpNDw); 
FILE* fxxCPKTclxXPxrdMAkwi = MWCNNLayer::openBinaryFile(fileString.c_str()); if 
(fxxCPKTclxXPxrdMAkwi == NULL) { printf("Unable to open Input Average file\n"); } 
TxNFOfYScyqGlEFFxbAv = new std::vector<float>; TxNFOfYScyqGlEFFxbAv->reserve(kkqTyvjYvRFtTOyQUwrF); 
if(hnewnpwgzKmOdualajhn==1){ retVal = fread(TxNFOfYScyqGlEFFxbAv->data(), 
sizeof(float), kkqTyvjYvRFtTOyQUwrF, fxxCPKTclxXPxrdMAkwi); if (retVal != 
(size_t)kkqTyvjYvRFtTOyQUwrF) { printf("MWInputLayer::loadAvg - File read Failed\n"); 
} } else{ MWInputLayer* inpLayer = static_cast<MWInputLayer*>(getLayer()); 
MWTensor* opTensor = inpLayer->getOutputTensor(0); int numChannels = 
opTensor->getChannels(); int channelSize = opTensor->getHeight() * 
opTensor->getWidth(); int channelOffset=0; std::vector<float> 
ZCArwzdUdwQuFQUWjnUE(numChannels); retVal = fread(ZCArwzdUdwQuFQUWjnUE.data(), 
sizeof(float), numChannels, fxxCPKTclxXPxrdMAkwi); if (retVal != (size_t)numChannels) 
{ printf("MWInputLayer::loadAvg - File read Failed\n"); } for(int 
i=0;i<numChannels;i++){ std::fill_n(TxNFOfYScyqGlEFFxbAv->begin()+channelOffset, 
channelSize, ZCArwzdUdwQuFQUWjnUE[i]); channelOffset = channelOffset+channelSize; 
} } fclose(fxxCPKTclxXPxrdMAkwi); return; } void MWInputLayerImpl::allocate() { 
MWInputLayer* inpLayer = static_cast<MWInputLayer*>(getLayer()); MWTensor* 
opTensor = inpLayer->getOutputTensor(0); if(opTensor->getopBufIndex() < 0) { 
armTensor.allocator()->allocate(); } else { 
armTensor.allocator()->import_memory(Memory((uint8_t 
*)kNsviQGMPdXzNMRixGWR->memBuffer[opTensor->getopBufIndex()]));  } if 
((armTensor.info()->total_size() / 4) == (opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth())) { 
setData((float*)armTensor.buffer()); } else { setData(m_inputImage); } 
inpLayer->getOutputTensor()->setData(getData()); } void fillIpToTensor(unsigned 
char* in_buffer, arm_compute::ITensor& tensor) { uint width = 
tensor.info()->dimension(0); uint height = tensor.info()->dimension(1); int 
data_size_in_bytes = 4;  int collapsed_upper = 
tensor.info()->tensor_shape().total_size_upper(2); uint8_t* ptr_out = 
tensor.buffer() + tensor.info()->offset_first_element_in_bytes(); const 
arm_compute::Strides& strides_in_bytes = tensor.info()->strides_in_bytes(); for 
(int i = 0; i < collapsed_upper; ++i) { size_t slice_offset = i * 
strides_in_bytes.z(); for (unsigned int y = 0; y < height; ++y) { size_t 
row_offset = y * strides_in_bytes.y(); memcpy(ptr_out + slice_offset + 
row_offset, in_buffer + i * width * height * data_size_in_bytes + y * width * 
data_size_in_bytes, width * data_size_in_bytes); } } } void 
fillTensorToIp(unsigned char* out_buffer, arm_compute::ITensor& tensor) { uint 
width = tensor.info()->dimension(0); uint height = tensor.info()->dimension(1); 
int data_size_in_bytes = 4;  int collapsed_upper = 
tensor.info()->tensor_shape().total_size_upper(2); uint8_t* ptr_out = 
tensor.buffer() + tensor.info()->offset_first_element_in_bytes(); const 
arm_compute::Strides& strides_in_bytes = tensor.info()->strides_in_bytes(); for 
(int i = 0; i < collapsed_upper; ++i) { size_t slice_offset = i * 
strides_in_bytes.z(); for (unsigned int y = 0; y < height; ++y) { size_t 
row_offset = y * strides_in_bytes.y(); memcpy(out_buffer + i * width * height * 
data_size_in_bytes + y * width * data_size_in_bytes, ptr_out + slice_offset + 
row_offset, width * data_size_in_bytes); } } } void MWInputLayerImpl::predict() 
{ float* inp = m_inputImage; int i, btch; MWInputLayer* inpLayer = 
static_cast<MWInputLayer*>(getLayer()); MWTensor* opTensor = 
inpLayer->getOutputTensor(0); float* out = m_inputImage; if 
((armTensor.info()->total_size() / 4) == (opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth())) { inp 
= (float*)armTensor.buffer(); out = (float*)armTensor.buffer(); } else { inp = 
m_inputImage; out = m_inputImage; } if (hnewnpwgzKmOdualajhn) { for (btch = 0; 
btch < opTensor->getBatchSize(); btch++) {
#pragma omp parallel for
 for (i = 0; i < opTensor->getChannels() * opTensor->getHeight() * 
opTensor->getWidth(); i++) { out[i] = inp[i] - TxNFOfYScyqGlEFFxbAv->data()[i]; } inp 
+= opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(); out 
+= opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(); } if 
((armTensor.info()->total_size() / 4) != (opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth())) { 
fillIpToTensor((unsigned char*)m_inputImage, armTensor); } }
#if MW_INPUT_TAP
 mw_interm_tap((float*)armTensor.buffer(), opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(), tap_count++);
#endif
 return; } void MWInputLayerImpl::cleanup() { MWCNNLayerImpl::cleanup(); 
free(m_inputImage); if (hnewnpwgzKmOdualajhn) { if (TxNFOfYScyqGlEFFxbAv) { 
free(TxNFOfYScyqGlEFFxbAv); } } return; } MWReLULayerImpl::MWReLULayerImpl(MWCNNLayer* 
layer, MWTargetNetworkImpl* ntwk_impl, int inPlace, int outbufIdx) : 
MWCNNLayerImpl(layer, ntwk_impl) { createReLULayer(outbufIdx); } 
MWReLULayerImpl::~MWReLULayerImpl() { } void 
MWReLULayerImpl::createReLULayer(int outbufIdx) { MWReLULayer* reluLayer = 
static_cast<MWReLULayer*>(getLayer()); MWTensor* ipTensor = 
reluLayer->getInputTensor(); MWTensor* opTensor = reluLayer->getOutputTensor(); 
Tensor* prevLayerarmTensor = getprevLayerarmTensor(ipTensor); if 
(ipTensor->getWidth() == 1 && ipTensor->getHeight() == 1) { 
armTensor.allocator()->init(TensorInfo( TensorShape((long unsigned 
int)opTensor->getChannels()), 1, DataType::F32, 4)); } else { 
armTensor.allocator()->init( TensorInfo(TensorShape((long unsigned 
int)ipTensor->getWidth(), (long unsigned int)ipTensor->getHeight(), (long 
unsigned int)opTensor->getChannels()), 1, DataType::F32, 4)); } 
getLayer()->getOutputTensor(0)->setopBufIndex(outbufIdx); 
m_actLayer.configure(prevLayerarmTensor, &armTensor, 
ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)); return; } 
void MWReLULayerImpl::allocate() { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); if(opTensor->getopBufIndex() < 0) { 
armTensor.allocator()->allocate(); } else { 
armTensor.allocator()->import_memory(Memory((uint8_t 
*)kNsviQGMPdXzNMRixGWR->memBuffer[opTensor->getopBufIndex()]));  } 
setData((float*)armTensor.buffer()); MWReLULayer* reluLayer = 
static_cast<MWReLULayer*>(getLayer()); 
reluLayer->getOutputTensor()->setData((float*)armTensor.buffer()); } void 
MWReLULayerImpl::predict() { MWReLULayer* reluLayer = 
static_cast<MWReLULayer*>(getLayer()); MWTensor* opTensor = 
reluLayer->getOutputTensor(); m_actLayer.run();
#if MW_RELU_TAP
 mw_interm_tap((float*)armTensor.buffer(), opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(), tap_count++);
#endif
 return; } MWNormLayerImpl::MWNormLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, unsigned REXdEoRjxuQJkqgIDihy,  double 
AFQBkxwYGKLsACiDKwRM,  double BLjrjqvCcCommiXWQLjs,  double IAlDgIFcchbwRGBSfVfA, int outbufIdx) 
: MWCNNLayerImpl(layer, ntwk_impl) { 
createNormLayer(REXdEoRjxuQJkqgIDihy, AFQBkxwYGKLsACiDKwRM, 
BLjrjqvCcCommiXWQLjs, IAlDgIFcchbwRGBSfVfA, outbufIdx); } 
MWNormLayerImpl::~MWNormLayerImpl() { } void 
MWNormLayerImpl::createNormLayer(unsigned REXdEoRjxuQJkqgIDihy, double 
AFQBkxwYGKLsACiDKwRM, double BLjrjqvCcCommiXWQLjs, double IAlDgIFcchbwRGBSfVfA,  int outbufIdx) 
{ MWNormLayer* normLayer = static_cast<MWNormLayer*>(getLayer()); MWTensor* 
ipTensor = normLayer->getInputTensor(); Tensor* prevLayerarmTensor = 
getprevLayerarmTensor(ipTensor); if (ipTensor->getWidth() == 1 && 
ipTensor->getHeight() == 1) { armTensor.allocator()->init(TensorInfo( 
TensorShape((long unsigned int)ipTensor->getChannels()), 1, DataType::F32, 4)); 
} else { armTensor.allocator()->init( TensorInfo(TensorShape((long unsigned 
int)ipTensor->getWidth(), (long unsigned int)ipTensor->getHeight(), (long 
unsigned int)ipTensor->getChannels()), 1, DataType::F32, 4)); } 
getLayer()->getOutputTensor(0)->setopBufIndex(outbufIdx); 
m_normLayer.configure(prevLayerarmTensor, &armTensor, 
NormalizationLayerInfo(NormType::CROSS_MAP, REXdEoRjxuQJkqgIDihy, 
AFQBkxwYGKLsACiDKwRM, BLjrjqvCcCommiXWQLjs, IAlDgIFcchbwRGBSfVfA)); return; } void 
MWNormLayerImpl::allocate() { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); if(opTensor->getopBufIndex() < 0) { 
armTensor.allocator()->allocate(); } else { 
armTensor.allocator()->import_memory(Memory((uint8_t 
*)kNsviQGMPdXzNMRixGWR->memBuffer[opTensor->getopBufIndex()]));  } 
setData((float*)armTensor.buffer()); MWNormLayer* normLayer = 
static_cast<MWNormLayer*>(getLayer()); 
normLayer->getOutputTensor()->setData((float*)armTensor.buffer()); } void 
MWNormLayerImpl::predict() { MWNormLayer* normLayer = 
static_cast<MWNormLayer*>(getLayer()); MWTensor* opTensor = 
normLayer->getOutputTensor(); m_normLayer.run();
#if MW_NORM_TAP
 mw_interm_tap((float*)armTensor.buffer(), opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(), tap_count++);
#endif
 return; } MWMaxPoolingLayerImpl::MWMaxPoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int NtWaRGCHLeTapjWdEHHS,  int OVOphSOolqRQDDoKPwxy,  
int PmFfARVzoHVAYkfpuvqK,  int QjgQHaUACFNSteMrRtRj, int MNuwXDSoGEYeABeVTwOh, int 
MIBnYCbKBdUrlfqlHdoo,  int NDjzAZSYJuWymuKDNZYB, int NZjOkZPwLzQsdEVkwMcX, 
bool UEESbUvbMihFnquvuFij, int nNULvWnBXnnWdpEkHPAH, const std::vector<int>& 
eFaDPmxDdzHlRYSAoMmX ) : MWCNNLayerImpl(layer, ntwk_impl) { 
assert(!UEESbUvbMihFnquvuFij); createMaxPoolingLayer(NtWaRGCHLeTapjWdEHHS, 
OVOphSOolqRQDDoKPwxy, PmFfARVzoHVAYkfpuvqK, QjgQHaUACFNSteMrRtRj, 
MNuwXDSoGEYeABeVTwOh, MIBnYCbKBdUrlfqlHdoo, NDjzAZSYJuWymuKDNZYB, 
NZjOkZPwLzQsdEVkwMcX,  nNULvWnBXnnWdpEkHPAH, eFaDPmxDdzHlRYSAoMmX); } 
MWMaxPoolingLayerImpl::~MWMaxPoolingLayerImpl() { } float* 
MWMaxPoolingLayerImpl::getIndexData() { assert(false); } void 
MWMaxPoolingLayerImpl::createMaxPoolingLayer(int NtWaRGCHLeTapjWdEHHS, int 
OVOphSOolqRQDDoKPwxy, int PmFfARVzoHVAYkfpuvqK, int QjgQHaUACFNSteMrRtRj, int 
MNuwXDSoGEYeABeVTwOh, int MIBnYCbKBdUrlfqlHdoo, int NDjzAZSYJuWymuKDNZYB, 
int NZjOkZPwLzQsdEVkwMcX, int nNULvWnBXnnWdpEkHPAH,  const std::vector<int>& 
eFaDPmxDdzHlRYSAoMmX) { MWMaxPoolingLayer* maxPoolLayer = 
static_cast<MWMaxPoolingLayer*>(getLayer()); MWTensor* ipTensor = 
maxPoolLayer->getInputTensor(); MWTensor* opTensor = 
maxPoolLayer->getOutputTensor(); Tensor* prevLayerarmTensor = 
getprevLayerarmTensor(ipTensor); 
armTensor.allocator()->init(TensorInfo(TensorShape((long unsigned 
int)opTensor->getWidth(), (long unsigned int)opTensor->getHeight(), (long 
unsigned int)opTensor->getChannels()), 1, DataType::F32, 4)); 
assert(nNULvWnBXnnWdpEkHPAH == 1); int outbufIdx = eFaDPmxDdzHlRYSAoMmX[0]; 
getLayer()->getOutputTensor(0)->setopBufIndex(outbufIdx); 
m_maxPoolLayer.configure( prevLayerarmTensor, &armTensor, PoolingLayerInfo( 
PoolingType::MAX, NtWaRGCHLeTapjWdEHHS, PadStrideInfo(QjgQHaUACFNSteMrRtRj, 
PmFfARVzoHVAYkfpuvqK, NDjzAZSYJuWymuKDNZYB, NZjOkZPwLzQsdEVkwMcX, 
MNuwXDSoGEYeABeVTwOh, MIBnYCbKBdUrlfqlHdoo, DimensionRoundingType::FLOOR))); 
return; } void MWMaxPoolingLayerImpl::allocate() { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); if(opTensor->getopBufIndex() < 0) { 
armTensor.allocator()->allocate(); } else { 
armTensor.allocator()->import_memory(Memory((uint8_t 
*)kNsviQGMPdXzNMRixGWR->memBuffer[opTensor->getopBufIndex()]));  } 
setData((float*)armTensor.buffer()); MWMaxPoolingLayer* maxPoolLayer = 
static_cast<MWMaxPoolingLayer*>(getLayer()); 
maxPoolLayer->getOutputTensor()->setData((float*)armTensor.buffer()); } void 
MWMaxPoolingLayerImpl::predict() { MWMaxPoolingLayer* maxPoolLayer = 
static_cast<MWMaxPoolingLayer*>(getLayer()); MWTensor* opTensor = 
maxPoolLayer->getOutputTensor(); m_maxPoolLayer.run();
#if MW_POOL_TAP
 mw_interm_tap((float*)armTensor.buffer(), opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(), tap_count++);
#endif
 return; } MWFCLayerImpl::MWFCLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int KHClOltUSuqFVVErSxVb, const char* 
sxuOMwKXOKfuExclRaSe,  const char* cQBKlCKXxecGPJrXBXdk, int outbufIdx) : 
MWCNNLayerImpl(layer, ntwk_impl) { createFCLayer(KHClOltUSuqFVVErSxVb, 
sxuOMwKXOKfuExclRaSe, cQBKlCKXxecGPJrXBXdk, outbufIdx); } 
MWFCLayerImpl::~MWFCLayerImpl() { } void MWFCLayerImpl::createFCLayer(int 
KHClOltUSuqFVVErSxVb, const char* sxuOMwKXOKfuExclRaSe, const char* 
cQBKlCKXxecGPJrXBXdk,  int outbufIdx) { MWFCLayer* fcLayer = 
static_cast<MWFCLayer*>(getLayer()); MWTensor* ipTensor = 
fcLayer->getInputTensor(); MWTensor* opTensor = fcLayer->getOutputTensor(); 
Tensor* prevLayerarmTensor = getprevLayerarmTensor(ipTensor); 
m_fcLayerWgtTensor.allocator()->init( TensorInfo(TensorShape((long unsigned 
int)(KHClOltUSuqFVVErSxVb), (long unsigned 
int)(opTensor->getChannels())), 1, DataType::F32, 4)); 
m_fcLayerBiasTensor.allocator()->init( TensorInfo(TensorShape((long unsigned 
int)(opTensor->getChannels())), 1, DataType::F32, 4)); 
armTensor.allocator()->init(TensorInfo( TensorShape((long unsigned 
int)(opTensor->getChannels() * opTensor->getBatchSize())), 1, DataType::F32, 
4)); getLayer()->getOutputTensor(0)->setopBufIndex(outbufIdx); 
m_fcLayer.configure(prevLayerarmTensor, &m_fcLayerWgtTensor, 
&m_fcLayerBiasTensor, &armTensor); m_fcLayerWgtTensor.allocator()->allocate(); 
m_fcLayerBiasTensor.allocator()->allocate(); 
loadWeights(sxuOMwKXOKfuExclRaSe,KHClOltUSuqFVVErSxVb); 
loadBias(cQBKlCKXxecGPJrXBXdk); return; } void MWFCLayerImpl::loadWeights(const 
char* fSKMHAqIghbYYgyIpNDw,int KHClOltUSuqFVVErSxVb) { size_t retVal; 
MWFCLayer* fcLayer = static_cast<MWFCLayer*>(getLayer()); MWTensor* ipTensor = 
fcLayer->getInputTensor(); MWTensor* opTensor = fcLayer->getOutputTensor(); int 
LtEgcYoEYjkrWuohutgw = opTensor->getChannels(); int kkqTyvjYvRFtTOyQUwrF = 
KHClOltUSuqFVVErSxVb * LtEgcYoEYjkrWuohutgw;  float* 
oJUVMnJggjhEdQLWzIUC = (float*)calloc(kkqTyvjYvRFtTOyQUwrF, sizeof(float)); std::string 
fileString = getLinuxPath(fSKMHAqIghbYYgyIpNDw); FILE* fxxCPKTclxXPxrdMAkwi = 
MWCNNLayer::openBinaryFile(fileString.c_str()); retVal = fread(oJUVMnJggjhEdQLWzIUC, 
sizeof(float), kkqTyvjYvRFtTOyQUwrF, fxxCPKTclxXPxrdMAkwi); if (retVal != 
(size_t)kkqTyvjYvRFtTOyQUwrF) { 
printf("MWFCLayer::loadWeights - File read Failed\n"); } if 
(ipTensor->getHeight() != 1 && ipTensor->getWidth() != 1) { float* 
oYbqYsqgVhrUzFEKbBbR = (float*)malloc(sizeof(float) * ipTensor->getHeight() * 
ipTensor->getWidth()); for (int k = 0; k < kkqTyvjYvRFtTOyQUwrF / 
ipTensor->getHeight() / ipTensor->getWidth(); k++) { for (int i = 0; i < 
ipTensor->getHeight() * ipTensor->getWidth(); i++) oYbqYsqgVhrUzFEKbBbR[i] = 
oJUVMnJggjhEdQLWzIUC[k * ipTensor->getHeight() * ipTensor->getWidth() + i]; for (int j 
= 0; j < ipTensor->getHeight(); j++) for (int i = 0; i < ipTensor->getWidth(); 
i++) oJUVMnJggjhEdQLWzIUC[k * ipTensor->getHeight() * ipTensor->getWidth() + j * 
ipTensor->getWidth() + i] = oYbqYsqgVhrUzFEKbBbR[j + i * ipTensor->getHeight()]; } 
free(oYbqYsqgVhrUzFEKbBbR); } std::copy_n((unsigned char*)oJUVMnJggjhEdQLWzIUC, kkqTyvjYvRFtTOyQUwrF 
* sizeof(float), (unsigned char*)m_fcLayerWgtTensor.buffer()); 
free(oJUVMnJggjhEdQLWzIUC); fclose(fxxCPKTclxXPxrdMAkwi); return; } void 
MWFCLayerImpl::loadBias(const char* fSKMHAqIghbYYgyIpNDw) { size_t retVal; 
MWFCLayer* fcLayer = static_cast<MWFCLayer*>(getLayer()); MWTensor* opTensor = 
fcLayer->getOutputTensor(); int getNumOutputFeatures = opTensor->getChannels(); 
float* ZDWLzHUkuZuIUZHfbGDY = (float*)calloc(getNumOutputFeatures, sizeof(float)); 
std::string fileString = getLinuxPath(fSKMHAqIghbYYgyIpNDw); FILE* fxxCPKTclxXPxrdMAkwi 
= MWCNNLayer::openBinaryFile(fileString.c_str()); int kkqTyvjYvRFtTOyQUwrF = 
getNumOutputFeatures;  retVal = fread(ZDWLzHUkuZuIUZHfbGDY, sizeof(float), 
kkqTyvjYvRFtTOyQUwrF, fxxCPKTclxXPxrdMAkwi); if (retVal != (size_t)kkqTyvjYvRFtTOyQUwrF) { 
printf("MWFCLayer::loadBias - File read Failed\n"); } std::copy_n((unsigned 
char*)ZDWLzHUkuZuIUZHfbGDY, kkqTyvjYvRFtTOyQUwrF * sizeof(float), (unsigned 
char*)m_fcLayerBiasTensor.buffer()); free(ZDWLzHUkuZuIUZHfbGDY); fclose(fxxCPKTclxXPxrdMAkwi); 
return; } void MWFCLayerImpl::allocate() { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); if(opTensor->getopBufIndex() < 0) { 
armTensor.allocator()->allocate(); } else { 
armTensor.allocator()->import_memory(Memory((uint8_t 
*)kNsviQGMPdXzNMRixGWR->memBuffer[opTensor->getopBufIndex()]));  } 
setData((float*)armTensor.buffer()); MWFCLayer* fcLayer = 
static_cast<MWFCLayer*>(getLayer()); 
fcLayer->getOutputTensor()->setData((float*)armTensor.buffer()); } void 
MWFCLayerImpl::predict() { MWFCLayer* fcLayer = 
static_cast<MWFCLayer*>(getLayer()); MWTensor* opTensor = 
fcLayer->getOutputTensor(); m_fcLayer.run();
#if MW_FC_TAP
 mw_interm_tap((float*)armTensor.buffer(), opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(), tap_count++);
#endif
 return; } void MWFCLayerImpl::cleanup() { MWCNNLayerImpl::cleanup(); 
m_fcLayerWgtTensor.allocator()->free(); 
m_fcLayerBiasTensor.allocator()->free(); return; } 
MWSoftmaxLayerImpl::MWSoftmaxLayerImpl(MWCNNLayer* layer, MWTargetNetworkImpl* 
ntwk_impl, int outbufIdx) : MWCNNLayerImpl(layer, ntwk_impl) { 
createSoftmaxLayer(outbufIdx); } MWSoftmaxLayerImpl::~MWSoftmaxLayerImpl() { } 
void MWSoftmaxLayerImpl::createSoftmaxLayer(int outbufIdx) { MWSoftmaxLayer* 
sfmxLayer = static_cast<MWSoftmaxLayer*>(getLayer()); MWTensor* ipTensor = 
sfmxLayer->getInputTensor(); MWTensor* opTensor = sfmxLayer->getOutputTensor(); 
Tensor* prevLayerarmTensor = getprevLayerarmTensor(ipTensor); 
armTensor.allocator()->init(TensorInfo(TensorShape((long unsigned 
int)opTensor->getWidth() * (long unsigned int)opTensor->getHeight() * (long 
unsigned int)opTensor->getChannels()), 1, DataType::F32, 4)); 
getLayer()->getOutputTensor(0)->setopBufIndex(outbufIdx); if 
(prevLayerarmTensor->info()->num_dimensions()>1){ 
m_flattenArmTensor.allocator()->init(TensorInfo(TensorShape((long unsigned 
int)ipTensor->getWidth() * (long unsigned int)ipTensor->getHeight() * (long 
unsigned int)ipTensor->getChannels()), 1, DataType::F32)); 
m_flattenLayer.configure(prevLayerarmTensor,&m_flattenArmTensor); 
m_flattenArmTensor.allocator()->allocate(); 
m_softmaxLayer.configure(&m_flattenArmTensor, &armTensor); } else  
m_softmaxLayer.configure(prevLayerarmTensor, &armTensor); return; } void 
MWSoftmaxLayerImpl::allocate() { MWTensor* opTensor = 
getLayer()->getOutputTensor(0); if(opTensor->getopBufIndex() < 0) { 
armTensor.allocator()->allocate(); } else { 
armTensor.allocator()->import_memory(Memory((uint8_t 
*)kNsviQGMPdXzNMRixGWR->memBuffer[opTensor->getopBufIndex()]));  } 
setData((float*)armTensor.buffer()); MWSoftmaxLayer* sfmxLayer = 
static_cast<MWSoftmaxLayer*>(getLayer()); 
sfmxLayer->getOutputTensor()->setData((float*)armTensor.buffer()); } void 
MWSoftmaxLayerImpl::predict() { MWSoftmaxLayer* sfmxLayer = 
static_cast<MWSoftmaxLayer*>(getLayer()); MWTensor* ipTensor = 
sfmxLayer->getInputTensor(); Tensor* prevLayerarmTensor = 
getprevLayerarmTensor(ipTensor); if 
(prevLayerarmTensor->info()->num_dimensions()>1){ m_flattenLayer.run(); } m_softmaxLayer.run();
#if MW_SFMX_TAP
 MWTensor* opTensor = sfmxLayer->getOutputTensor(); 
mw_interm_tap(opTensor->getData(), opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(), tap_count++);
#endif
 return; } void MWSoftmaxLayerImpl::cleanup() { MWCNNLayerImpl::cleanup(); 
if(getprevLayerarmTensor(getLayer()->getInputTensor())->info()->num_dimensions()>1) 
{ m_flattenArmTensor.allocator()->free(); } return; } 
MWAvgPoolingLayerImpl::MWAvgPoolingLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int NtWaRGCHLeTapjWdEHHS,  int OVOphSOolqRQDDoKPwxy,  
int PmFfARVzoHVAYkfpuvqK,  int QjgQHaUACFNSteMrRtRj,  int MCrRCXUsCsGPMgQbvMOt,  int 
MUmglsoWcEiRiAZsclur, int ) : MWCNNLayerImpl(layer, ntwk_impl) { 
createAvgPoolingLayer(NtWaRGCHLeTapjWdEHHS, OVOphSOolqRQDDoKPwxy, PmFfARVzoHVAYkfpuvqK, 
QjgQHaUACFNSteMrRtRj, MCrRCXUsCsGPMgQbvMOt, MUmglsoWcEiRiAZsclur); } 
MWAvgPoolingLayerImpl::~MWAvgPoolingLayerImpl() { } void 
MWAvgPoolingLayerImpl::createAvgPoolingLayer(int NtWaRGCHLeTapjWdEHHS, int 
OVOphSOolqRQDDoKPwxy, int PmFfARVzoHVAYkfpuvqK, int QjgQHaUACFNSteMrRtRj, int 
MCrRCXUsCsGPMgQbvMOt, int MUmglsoWcEiRiAZsclur) { MWAvgPoolingLayer* avgpoolLayer 
= static_cast<MWAvgPoolingLayer*>(getLayer()); MWTensor* opTensor = 
avgpoolLayer->getOutputTensor(); MWTensor* ipTensor = 
avgpoolLayer->getInputTensor(); Tensor* prevLayerarmTensor = 
getprevLayerarmTensor(ipTensor); 
armTensor.allocator()->init(TensorInfo(TensorShape((long unsigned 
int)opTensor->getWidth(), (long unsigned int)opTensor->getHeight(), (long 
unsigned int)opTensor->getChannels()), 1, DataType::F32)); 
m_avgPoolLayer.configure( prevLayerarmTensor, &armTensor, PoolingLayerInfo( 
PoolingType::AVG, NtWaRGCHLeTapjWdEHHS, PadStrideInfo(QjgQHaUACFNSteMrRtRj, 
PmFfARVzoHVAYkfpuvqK, MCrRCXUsCsGPMgQbvMOt, MUmglsoWcEiRiAZsclur, 
DimensionRoundingType::FLOOR))); return ; } void 
MWAvgPoolingLayerImpl::allocate() { armTensor.allocator()->allocate(); 
setData((float*)armTensor.buffer()); MWAvgPoolingLayer* avgpoolLayer = 
static_cast<MWAvgPoolingLayer*>(getLayer()); 
avgpoolLayer->getOutputTensor()->setData((float*)armTensor.buffer()); } void 
MWAvgPoolingLayerImpl::predict() { m_avgPoolLayer.run();
#if MW_AVG_POOL_TAP
 MWAvgPoolingLayer* avgpoolLayer = static_cast<MWAvgPoolingLayer*>(getLayer()); 
MWTensor* opTensor = avgpoolLayer->getOutputTensor(); 
mw_interm_tap(opTensor->getData(), opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(), tap_count++);
#endif
 return; } MWOutputLayerImpl::MWOutputLayerImpl(MWCNNLayer* layer, 
MWTargetNetworkImpl* ntwk_impl, int outbufIdx) : MWCNNLayerImpl(layer, 
ntwk_impl) { createOutputLayer(outbufIdx); } 
MWOutputLayerImpl::~MWOutputLayerImpl() { } void 
MWOutputLayerImpl::createOutputLayer( int outbufIdx) { MWOutputLayer* opLayer = 
static_cast<MWOutputLayer*>(getLayer()); MWTensor* ipTensor = 
opLayer->getInputTensor(0); MWTensor* opTensor = opLayer->getOutputTensor(0); 
m_outputData = (float*)calloc(opTensor->getBatchSize() * 
opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth(), 
sizeof(float)); setData(m_outputData); m_outputArmTensor = 
&ipTensor->getOwner()->getImpl()->armTensor; } void 
MWOutputLayerImpl::allocate() { MWOutputLayer* opLayer = 
static_cast<MWOutputLayer*>(getLayer()); MWTensor* ipTensor = 
opLayer->getInputTensor(0); MWTensor* opTensor = opLayer->getOutputTensor(0); 
if ((m_outputArmTensor->info()->total_size() / 4) == (opTensor->getBatchSize() 
* opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth())) { 
setData((float*)m_outputArmTensor->buffer()); } 
opLayer->getOutputTensor()->setData(getData()); } void 
MWOutputLayerImpl::predict() { MWOutputLayer* opLayer = 
static_cast<MWOutputLayer*>(getLayer()); MWTensor* ipTensor = 
opLayer->getInputTensor(0); MWTensor* opTensor = opLayer->getOutputTensor(0); 
if ((m_outputArmTensor->info()->total_size() / 4) != (opTensor->getBatchSize() 
* opTensor->getChannels() * opTensor->getHeight() * opTensor->getWidth())) { 
fillTensorToIp((unsigned char*)opTensor->getData(), *m_outputArmTensor); } 
return; } void MWOutputLayerImpl::cleanup() { free(m_outputData); }