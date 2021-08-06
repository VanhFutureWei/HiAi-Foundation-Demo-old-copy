/*
 * @file classify_jni.cpp
 *
 * Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <jni.h>
#include <string>

#include <memory.h>
#include "HiAiModelManagerService.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <sys/time.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

#define LOG_TAG "SYNC_DDK_MSG"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace std;
using namespace hiai;

static shared_ptr<AiModelMngerClient> g_clientSync = nullptr;
static vector<vector<TensorDimension>> inputDimension;
static vector<vector<TensorDimension>> outputDimension;
static vector<vector<shared_ptr<AiTensor>>> input_tensor;

static vector<vector<shared_ptr<AiTensor>>> output_tensor;

static map<string, int> sync_nameToIndex;
static long time_use_sync = 0;

static const int SUCCESS = 0;
static const int FAILED = -1;

void ResourceDestroy(shared_ptr<AiModelBuilder>& modelBuilder, vector<MemBuffer*>& memBuffers)
{
    if (modelBuilder == nullptr) {
        LOGE("[HIAI_DEMO_SYNC] modelBuilder is null.");
        return;
    }

    for (auto tmpBuffer : memBuffers) {
        modelBuilder->MemBufferDestroy(tmpBuffer);
    }
    return;
}

int LoadSync(vector<string>& names, vector<string>& modelPaths, shared_ptr<AiModelMngerClient>& client)
{
    int ret;
    vector<shared_ptr<AiModelDescription>> modelDescs;
    vector<MemBuffer*> memBuffers;
    shared_ptr<AiModelBuilder> modelBuilder = make_shared<AiModelBuilder>(client);
    if (modelBuilder == nullptr) {
        LOGI("[HIAI_DEMO_SYNC] creat modelBuilder failed.");
        return FAILED;
    }

    for (size_t i = 0; i < modelPaths.size(); ++i){
        string modelPath = modelPaths[i];
        string modelName = names[i];
        sync_nameToIndex[modelName] = i;

        // We can achieve the optimization by loading model from OM file.
        LOGI("[HIAI_DEMO_SYNC] modelpath is %s\n.", modelPath.c_str());
        MemBuffer* buffer = modelBuilder->InputMemBufferCreate(modelPath);
        if (buffer == nullptr) {
            LOGE("[HIAI_DEMO_SYNC] cannot find the model file.");
            return FAILED;
        }
        memBuffers.push_back(buffer);

        string modelNameFull = string(modelName) + string(".om");
        shared_ptr<AiModelDescription> desc = make_shared<AiModelDescription>(modelNameFull, AiModelDescription_Frequency_HIGH, HIAI_FRAMEWORK_NONE, HIAI_MODELTYPE_ONLINE, AiModelDescription_DeviceType_NPU);
        if (desc == nullptr) {
            LOGE("[HIAI_DEMO_SYNC] LoadModelSync: desc make_shared error.");
            ResourceDestroy(modelBuilder, memBuffers);
            return FAILED;
        }
        desc->SetModelBuffer(buffer->GetMemBufferData(), buffer->GetMemBufferSize());

        LOGE("[HIAI_DEMO_SYNC] loadModel %s IO Tensor.", desc->GetName().c_str());
        modelDescs.push_back(desc);
    }

    ret = client->Load(modelDescs);
    ResourceDestroy(modelBuilder, memBuffers);
    if (ret != 0) {
        LOGE("[HIAI_DEMO_SYNC] Model Load Failed.");
        return FAILED;
    }
    return SUCCESS;
}

shared_ptr<AiModelMngerClient> LoadModelSync(vector<string> names, vector<string> modelPaths, vector<bool> Aipps)
{
    shared_ptr<AiModelMngerClient> clientSync = make_shared<AiModelMngerClient>();
    if (clientSync == nullptr) {
        LOGE("[HIAI_DEMO_SYNC] Model Manager Client make_shared error.");
        return nullptr;
    }

    int ret = clientSync->Init(nullptr);
    if (ret != 0) {
        LOGE("[HIAI_DEMO_SYNC] Model Manager Init Failed.");
        return nullptr;
    }

    ret = LoadSync(names, modelPaths, clientSync);
    if (ret != SUCCESS) {
        LOGE("[HIAI_DEMO_ASYNC] LoadSync Failed.");
        return nullptr;
    }

    inputDimension.clear();
    outputDimension.clear();
    input_tensor.clear();
    output_tensor.clear();

    for (size_t i = 0; i < names.size(); ++i) {
        string modelName = names[i];
        bool isUseAipp = Aipps[i];
        LOGI("[HIAI_DEMO_SYNC] Get model %s IO Tensor. Use AIPP %d", modelName.c_str(), isUseAipp);
        vector<TensorDimension> inputDims, outputDims;
        ret = clientSync->GetModelIOTensorDim(string(modelName) + string(".om"), inputDims, outputDims);
        if (ret != 0) {
            LOGE("[HIAI_DEMO_SYNC] Get Model IO Tensor Dimension failed,ret is %d.", ret);
            return nullptr;
        }

        if (inputDims.size() == 0) {
            LOGE("[HIAI_DEMO_SYNC] inputDims.size() == 0");
            return nullptr;
        }

        inputDimension.push_back(inputDims);
        outputDimension.push_back(outputDims);

        vector<shared_ptr<AiTensor>> inputTensors, outputTensors;
        for (auto in_dim : inputDims) {
            shared_ptr<AiTensor> input = make_shared<AiTensor>();
            if (isUseAipp) {
                ret = input->Init(in_dim.GetNumber(), in_dim.GetHeight(), in_dim.GetWidth(), AiTensorImage_YUV420SP_U8);
                LOGI("[HIAI_DEMO_SYNC] model %s uses AIPP(input).", modelName.c_str());
            } else {
                ret = input->Init(&in_dim);
                LOGI("[HIAI_DEMO_SYNC] model %s does not use AIPP(input).", modelName.c_str());
            }
            if (ret != 0) {
                LOGE("[HIAI_DEMO_SYNC] model %s AiTensor Init failed(input).", modelName.c_str());
                return nullptr;
            }
            inputTensors.push_back(input);
        }
        input_tensor.push_back(inputTensors);

        if (input_tensor.size() == 0) {
            LOGE("[HIAI_DEMO_SYNC] input_tensor.size() == 0");
            return nullptr;
        }

        for (auto out_dim : outputDims) {
            shared_ptr<AiTensor> output = make_shared<AiTensor>();
            ret = output->Init(&out_dim);
            if (ret != 0) {
                LOGE("[HIAI_DEMO_SYNC] model %s AiTensor Init failed(output).", modelName.c_str());
                return nullptr;
            }
            outputTensors.push_back(output);
        }

        output_tensor.push_back(outputTensors);

        if (output_tensor.size() == 0) {
            LOGE("[HIAI_DEMO_SYNC] output_tensor.size() == 0");
            return nullptr;
        }
    }

    return clientSync;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_huawei_hiaidemo_utils_ModelManager_GetTimeUseSync(JNIEnv *env, jclass type)
{
    return time_use_sync;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_huawei_hiaidemo_utils_ModelManager_loadModelSync(JNIEnv *env, jclass type,jobject modelInfo){

    jclass classList = env->GetObjectClass(modelInfo);
    if(classList == nullptr){
        LOGE("[HIAI_DEMO_SYNC] can not find List class.");
    }

    jmethodID listGet = env->GetMethodID(classList, "get", "(I)Ljava/lang/Object;");
    jmethodID listSize = env->GetMethodID(classList, "size", "()I");
    int len = static_cast<int>(env->CallIntMethod(modelInfo, listSize));

    vector<string> names, modelPaths;
    vector<bool> aipps;
    for(int i = 0;i < len ;i++){
        jobject modelInfoObj = env->CallObjectMethod(modelInfo, listGet, i);
        jclass modelInfoClass = env->GetObjectClass(modelInfoObj);
        jmethodID getOfflineModelName = env->GetMethodID(modelInfoClass,"getOfflineModelName","()Ljava/lang/String;");
        jmethodID getModelPath = env->GetMethodID(modelInfoClass,"getModelPath","()Ljava/lang/String;");
        jmethodID getUseAIPP = env->GetMethodID(modelInfoClass,"getUseAIPP","()Z");

        if(getOfflineModelName == nullptr)
        {
            LOGE("[HIAI_DEMO_SYNC] can not find getOfflineModelName method.");
            return nullptr;
        }
        if(getModelPath == nullptr){
            LOGE("[HIAI_DEMO_SYNC] can not find getModelPath method.");
            return nullptr;
        }
        if(getUseAIPP == nullptr){
            LOGE("[HIAI_DEMO_SYNC] can not find getUseAIPP method.");
            return nullptr;
        }

        jboolean useaipp = (jboolean)env->CallBooleanMethod(modelInfoObj,getUseAIPP);
        jstring modelname = (jstring)env->CallObjectMethod(modelInfoObj,getOfflineModelName);
        jstring modelpath = (jstring)env->CallObjectMethod(modelInfoObj,getModelPath);
        const char* modelName = env->GetStringUTFChars(modelname, 0);
        LOGE("[HIAI_DEMO_SYNC] modelName is %s .",modelName);
        if(modelName == nullptr)
        {
            LOGE("[HIAI_DEMO_SYNC] modelName is invalid.");
            return nullptr;
        }
        const char *modelPath = env->GetStringUTFChars(modelpath, 0);
        if(modelPath == nullptr)
        {
            LOGE("[HIAI_DEMO_SYNC] modelPath is invalid.");
            return nullptr;
        }
        LOGE("[HIAI_DEMO_SYNC] useaipp is %d.", bool(useaipp==JNI_TRUE));
        aipps.push_back(bool(useaipp==JNI_TRUE));
        names.push_back(string(modelName));
        modelPaths.push_back(string(modelPath));
    }

    // load
    if (!g_clientSync)
    {
        g_clientSync = LoadModelSync(names, modelPaths, aipps);
        if (g_clientSync == nullptr)
        {
            LOGE("[HIAI_DEMO_SYNC] g_clientSync loadModel is nullptr.");
            return nullptr;
        }
    }

    // load model
    LOGI("[HIAI_DEMO_SYNC] sync load model INPUT NCHW : %d %d %d %d." , inputDimension[0][0].GetNumber(), inputDimension[0][0].GetChannel(), inputDimension[0][0].GetHeight(), inputDimension[0][0].GetWidth());
    LOGI("[HIAI_DEMO_SYNC] sync load model OUTPUT NCHW : %d %d %d %d." , outputDimension[0][0].GetNumber(), outputDimension[0][0].GetChannel(), outputDimension[0][0].GetHeight(), outputDimension[0][0].GetWidth());

    for(int i = 0;i < len ;i++){
        jobject modelInfoObj = env->CallObjectMethod(modelInfo, listGet, i);
        jclass modelInfoClass = env->GetObjectClass(modelInfoObj);
        jfieldID input_n_id = env->GetFieldID(modelInfoClass,"input_N","I");
        jfieldID input_c_id = env->GetFieldID(modelInfoClass,"input_C","I");
        jfieldID input_h_id = env->GetFieldID(modelInfoClass,"input_H","I");
        jfieldID input_w_id = env->GetFieldID(modelInfoClass,"input_W","I");
        jfieldID input_Number = env->GetFieldID(modelInfoClass,"input_Number","I");
        env->SetIntField(modelInfoObj,input_n_id,inputDimension[i][0].GetNumber());
        env->SetIntField(modelInfoObj,input_c_id,inputDimension[i][0].GetChannel());
        env->SetIntField(modelInfoObj,input_h_id,inputDimension[i][0].GetHeight());
        env->SetIntField(modelInfoObj,input_w_id,inputDimension[i][0].GetWidth());
        env->SetIntField(modelInfoObj,input_Number,inputDimension[i].size());

        jfieldID output_n_id = env->GetFieldID(modelInfoClass,"output_N","I");
        jfieldID output_c_id = env->GetFieldID(modelInfoClass,"output_C","I");
        jfieldID output_h_id = env->GetFieldID(modelInfoClass,"output_H","I");
        jfieldID output_w_id = env->GetFieldID(modelInfoClass,"output_W","I");
        jfieldID output_Number = env->GetFieldID(modelInfoClass,"output_Number","I");

        env->SetIntField(modelInfoObj,output_n_id,outputDimension[i][0].GetNumber());
        env->SetIntField(modelInfoObj,output_c_id,outputDimension[i][0].GetChannel());
        env->SetIntField(modelInfoObj,output_h_id,outputDimension[i][0].GetHeight());
        env->SetIntField(modelInfoObj,output_w_id,outputDimension[i][0].GetWidth());
        env->SetIntField(modelInfoObj,output_Number,outputDimension[i].size());
    }

    return modelInfo;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_huawei_hiaidemo_utils_ModelManager_runModelSync(JNIEnv *env, jclass type, jobject modelInfo, jobject bufList)
{
    // check params
    if(env == nullptr)
    {
        LOGE("[HIAI_DEMO_SYNC] runModelSync env is null");
        return nullptr;
    }

    jclass ModelInfo = env->GetObjectClass(modelInfo);
    if(ModelInfo == nullptr)
    {
        LOGE("[HIAI_DEMO_SYNC] can not find ModelInfo class.");
        return nullptr;
    }

    if (bufList == nullptr)
    {
        LOGE("[HIAI_DEMO_SYNC] buf_ is null.");
        return nullptr;
    }

    jmethodID getOfflineModelName = env->GetMethodID(ModelInfo,"getOfflineModelName","()Ljava/lang/String;");
    jmethodID getModelPath = env->GetMethodID(ModelInfo,"getModelPath","()Ljava/lang/String;");

    if(getOfflineModelName == nullptr){
        LOGE("[HIAI_DEMO_SYNC] can not find getOfflineModelName method.");
        return nullptr;
    }
    if(getModelPath == nullptr){
        LOGE("[HIAI_DEMO_SYNC] can not find getModelPath method.");
        return nullptr;
    }

    jstring modelname = (jstring)env->CallObjectMethod(modelInfo,getOfflineModelName);
    jstring modelpath = (jstring)env->CallObjectMethod(modelInfo,getModelPath);

    const char* modelName = env->GetStringUTFChars(modelname, 0);
    if(modelName == nullptr)
    {
        LOGE("[HIAI_DEMO_SYNC] modelName is invalid.");
        return nullptr;
    }

    int vecIndex = sync_nameToIndex[modelName];

    const char *modelPath = env->GetStringUTFChars(modelpath, 0);
    if(modelPath == nullptr)
    {
        LOGE("[HIAI_DEMO_SYNC] modelPath is invalid.");
        return nullptr;
    }
    // buf_list
    jclass classList = env->GetObjectClass(bufList);
    if(classList == nullptr){
        LOGE("[HIAI_DEMO_SYNC] can not find List class.");
    }
    jmethodID listGet = env->GetMethodID(classList, "get", "(I)Ljava/lang/Object;");
    jmethodID listSize = env->GetMethodID(classList, "size", "()I");
    if(listGet == nullptr){
        LOGE("[HIAI_DEMO_SYNC] can not find get method.");
    }
    if(listSize == nullptr){
        LOGE("[HIAI_DEMO_SYNC] can not find size method.");
    }
    int len = static_cast<int>(env->CallIntMethod(bufList, listSize));

    // load
    if (!g_clientSync)
    {
        LOGE("[HIAI_DEMO_SYNC] Model Manager Client is nullptr.");
    }
    env->ReleaseStringUTFChars(modelpath, modelPath);

    // run
    LOGI("[HIAI_DEMO_SYNC] INPUT NCHW : %d %d %d %d." , inputDimension[0][0].GetNumber(), inputDimension[0][0].GetChannel(), inputDimension[0][0].GetHeight(), inputDimension[0][0].GetWidth());
    LOGI("[HIAI_DEMO_SYNC] OUTPUT NCHW : %d %d %d %d." , outputDimension[0][0].GetNumber(), outputDimension[0][0].GetChannel(), outputDimension[0][0].GetHeight(), outputDimension[0][0].GetWidth());

    for(int i = 0;i < len ;i++){
        jbyteArray buf_ = (jbyteArray)(env->CallObjectMethod(bufList, listGet, i));
        jbyte *dataBuff = nullptr;
        int dataBuffSize = 0;
        dataBuff = env->GetByteArrayElements(buf_, nullptr);
        dataBuffSize = env->GetArrayLength(buf_);
        if(input_tensor[vecIndex][i]->GetSize() != dataBuffSize)
        {
            LOGE("[HIAI_DEMO_SYNC] input->GetSize(%d) != dataBuffSize(%d) ",input_tensor[vecIndex][i]->GetSize(),dataBuffSize);
            return nullptr;
        }
        memmove(input_tensor[vecIndex][i]->GetBuffer(), dataBuff, (size_t)dataBuffSize);
        env->ReleaseByteArrayElements(buf_, dataBuff, 0);
    }


    AiContext context;
    string key = "model_name";
    string value = modelName;
    value += ".om";
    context.AddPara(key, value);

    LOGI("[HIAI_DEMO_SYNC] runModel modelname:%s", modelName);

    // before process
    struct timeval tpstart, tpend;
    gettimeofday(&tpstart, nullptr);
    int istamp;
    int ret = g_clientSync->Process(context, input_tensor[vecIndex], output_tensor[vecIndex], 1000, istamp);
    if (ret) {
        LOGE("[HIAI_DEMO_SYNC] Runmodel Failed!, ret=%d\n", ret);
        return nullptr;
    }

    // after process
    gettimeofday(&tpend, nullptr);
    float time_use = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;

    time_use_sync =  time_use / 1000;

    LOGE("[HIAI_DEMO_SYNC] inference time %f ms.\n", time_use / 1000);

    // output_tensor
    jclass output_list_class = env->FindClass("java/util/ArrayList");
    jmethodID  output_list_init = env->GetMethodID(output_list_class,"<init>","()V");
    jmethodID list_add = env->GetMethodID(output_list_class,"add","(Ljava/lang/Object;)Z");
    jobject output_list = env->NewObject(output_list_class,output_list_init,"");

    long output_tensor_size = output_tensor[vecIndex].size();
    LOGI("[HIAI_DEMO_SYNC] output_tensor_size is %ld .",output_tensor_size);
    for(long j = 0; j < output_tensor_size; j++){
        float *outputBuffer = (float *)output_tensor[vecIndex][j]->GetBuffer();
        int outputsize = outputDimension[vecIndex][j].GetNumber() * outputDimension[vecIndex][j].GetChannel() * outputDimension[vecIndex][j].GetHeight() * outputDimension[vecIndex][j].GetWidth();
        jfloatArray  result = env->NewFloatArray(outputsize);
        jfloat temp[outputsize];
        for(int i =0;i < outputsize;i++)
        {
            temp[i] = outputBuffer[i];
        }
        env->SetFloatArrayRegion(result,0,outputsize,temp);
        jboolean output_add = env->CallBooleanMethod(output_list,list_add,result);
        LOGI("[HIAI_DEMO_SYNC] output_add result  is %d .",output_add);
    }
    env->ReleaseStringUTFChars(modelname, modelName);
    return output_list;
}
