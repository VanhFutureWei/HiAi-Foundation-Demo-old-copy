/*
*@file classify_async_jni.cpp
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
#include <unistd.h>

#define LOG_TAG "ASYNC_DDK_MSG"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

using namespace std;
using namespace hiai;
static jclass callbacksClass;
static jobject callbacksInstance;
JavaVM *jvm;

static float time_use;
struct timeval tpstart, tpend;

static map<int32_t ,vector<shared_ptr<AiTensor>>*> map_input_tensor;

static mutex mutex_map;
static condition_variable condition_;

//extern bool g_isAIPP;
static const int SUCCESS = 0;
static const int FAILED = -1;

static map<string, int> async_nameToIndex;

class JNIListener : public AiModelManagerClientListener
{
public:
    JNIListener(){}
    ~JNIListener(){}

    void OnProcessDone(const AiContext &context, int32_t result, const vector<shared_ptr<AiTensor>> &out_data, int32_t istamp);
    void OnServiceDied();
};

void JNIListener::OnProcessDone(const AiContext &context, int result, const vector<shared_ptr<AiTensor>> &output_tensor, int32_t istamp)
{
    std::unique_lock<std::mutex> lock(mutex_map);
    map_input_tensor.erase(istamp);
    condition_.notify_all();
    if (result != 0) {
        LOGI("[HIAI_DEMO_ASYNC] AYSNC infrence error is %d.", result);
        return;
    }
    gettimeofday(&tpend, nullptr);
    time_use = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
    LOGI("[HIAI_DEMO_ASYNC] AYSNC inference time %f ms, JNI layer onRunDone istamp: %d", time_use / 1000, istamp);
    JNIEnv *env = nullptr;
    jvm->AttachCurrentThread(&env, nullptr);
    vector<string> keys;
    ((AiContext)context).GetAllKeys(keys);
    for (auto key : keys)
    {
        string value = ((AiContext)context).GetPara(key);
        LOGI("[HIAI_DEMO_ASYNC] key: %s, value: %s.", key.c_str(), value.c_str());
    }
    jclass output_list_class = env->FindClass("java/util/ArrayList");
    jmethodID  output_list_init = env->GetMethodID(output_list_class,"<init>","()V");
    jobject output_list = env->NewObject(output_list_class,output_list_init,"");
    jmethodID list_add = env->GetMethodID(output_list_class,"add","(Ljava/lang/Object;)Z");
    long output_tensor_size = output_tensor.size();
    for(long j = 0; j < output_tensor_size; j++)
    {
        jfloat *outputBuffer = (jfloat *) output_tensor[j]->GetBuffer();
        auto output_count = output_tensor[j]->GetSize()/sizeof(jfloat);
        jfloatArray  result = env->NewFloatArray(output_count);
        jfloat temp[output_count];
        for(int i =0;i < output_count;i++)
        {
            temp[i] = outputBuffer[i];
        }
        env->SetFloatArrayRegion(result,0,output_count,temp);
        jboolean output_add = env->CallBooleanMethod(output_list,list_add,result);
    }
    jfloat infertime = time_use;
    if(callbacksInstance == nullptr)
    {
        return;
    }
    jmethodID onValueReceived = env->GetMethodID(callbacksClass, "OnProcessDone", "(ILjava/util/ArrayList;F)V");
    if(onValueReceived == nullptr){
        LOGI("[HIAI_DEMO_ASYNC] jni onValueReceived null");
    }
    env->CallVoidMethod(callbacksInstance, onValueReceived, istamp, output_list, infertime);
}

void JNIListener::OnServiceDied()
{
    LOGE("[HIAI_DEMO_ASYNC] JNI layer OnServiceDied:");

    JNIEnv *env = nullptr;

    jvm->AttachCurrentThread(&env, nullptr);

    if(callbacksInstance == nullptr)
    {
        return;
    }
    else
    {
        jmethodID onValueReceived = env->GetMethodID(callbacksClass, "OnServiceDied", "()V");
        env->CallVoidMethod(callbacksInstance, onValueReceived);
    }
}

static shared_ptr<AiModelMngerClient> mclientAsync = nullptr;
static vector<vector<TensorDimension>> inputDimension;
static vector<vector<TensorDimension>> outputDimension;
static vector<vector<shared_ptr<AiTensor>>> input_tensor1_vec;
static vector<vector<shared_ptr<AiTensor>>> input_tensor2_vec;
static vector<vector<shared_ptr<AiTensor>>> output_tensor_vec;

vector<shared_ptr<AiTensor>>* findInputTensor(int vecIdx)
{
    vector<shared_ptr<AiTensor>> & input_tensor1 = input_tensor1_vec[vecIdx];
    vector<shared_ptr<AiTensor>> & input_tensor2 = input_tensor2_vec[vecIdx];
    std::unique_lock<std::mutex> ulock(mutex_map);
    while(map_input_tensor.size()==2){
        condition_.wait_for(ulock,chrono::seconds(1));
    }
    if(map_input_tensor.size()==0){
        return &input_tensor1;
    }else if(map_input_tensor.size()==1) {
        map<int32_t, vector<shared_ptr<AiTensor>> *>::iterator ite;
        ite = map_input_tensor.begin();
        if (ite->second == &input_tensor1) {
            return &input_tensor2;
        } else {
            return &input_tensor1;
        }
    }
    return nullptr;
}

void AsyncResourceDestroy(shared_ptr<AiModelBuilder>& modelBuilder, vector<MemBuffer*>& memBuffers)
{
    if (modelBuilder == nullptr) {
        LOGE("[HIAI_DEMO_ASYNC] modelBuilder is null.");
        return;
    }

    for (auto tmpBuffer : memBuffers) {
        modelBuilder->MemBufferDestroy(tmpBuffer);
    }
    return;
}

shared_ptr<JNIListener> listener = make_shared<JNIListener>();

int LoadASync(vector<string>& names, vector<string>& modelPaths, shared_ptr<AiModelMngerClient>& client)
{
    int ret;
    vector<shared_ptr<AiModelDescription>> modelDescs;
    vector<MemBuffer*> memBuffers;
    shared_ptr<AiModelBuilder> modelBuilder = make_shared<AiModelBuilder>(client);
    if (modelBuilder == nullptr) {
        LOGE("[HIAI_DEMO_ASYNC] creat mcbuilder failed.");
        return FAILED;
    }

    for (size_t i = 0; i < modelPaths.size(); ++i){
        string modelPath = modelPaths[i];
        string modelName = names[i];
        async_nameToIndex[modelName] = i;

        // We can achieve the optimization by loading model from OM file.
        LOGI("[HIAI_DEMO_ASYNC] modelpath is %s\n.", modelPath.c_str());
        MemBuffer* buffer = modelBuilder->InputMemBufferCreate(modelPath);
        if (buffer == nullptr)
        {
            LOGE("[HIAI_DEMO_ASYNC] cannot find the model file.");
            return FAILED;
        }
        memBuffers.push_back(buffer);
        string modelNameFull = string(modelName) + string(".om");
        shared_ptr<AiModelDescription> desc = make_shared<AiModelDescription>(modelNameFull, AiModelDescription_Frequency_HIGH, HIAI_FRAMEWORK_NONE, HIAI_MODELTYPE_ONLINE, AiModelDescription_DeviceType_NPU);
        if (desc == nullptr) {
            LOGE("[HIAI_DEMO_ASYNC] LoadASync: desc make_shared error.");
            AsyncResourceDestroy(modelBuilder, memBuffers);
            return FAILED;
        }
        desc->SetModelBuffer(buffer->GetMemBufferData(), buffer->GetMemBufferSize());

        LOGE("[HIAI_DEMO_ASYNC] loadModel %s IO Tensor.", desc->GetName().c_str());
        modelDescs.push_back(desc);
    }

    ret = client->Load(modelDescs);
    AsyncResourceDestroy(modelBuilder, memBuffers);
    if (ret != 0)
    {
        LOGE("[HIAI_DEMO_ASYNC] Model Load Failed.");
        return FAILED;
    }
    return SUCCESS;
}

shared_ptr<AiModelMngerClient> LoadModelASync(vector<string> names, vector<string> modelPaths, vector<bool> Aipps)
{
    shared_ptr<AiModelMngerClient> client_ptr = make_shared<AiModelMngerClient>();
    if (client_ptr == nullptr) {
        LOGE("[HIAI_DEMO_ASYNC] Model Manager Client make_shared error.");
        return nullptr;
    }
    int ret = client_ptr->Init(listener);
    if (ret != 0) {
        LOGE("[HIAI_DEMO_ASYNC] Model Manager Init Failed.");
        return nullptr;
    }

    ret = LoadASync(names, modelPaths, client_ptr);
    if (ret != SUCCESS) {
        LOGE("[HIAI_DEMO_ASYNC] LoadASync Failed.");
        return nullptr;
    }

    inputDimension.clear();
    outputDimension.clear();
    input_tensor1_vec.clear();
    input_tensor2_vec.clear();
    output_tensor_vec.clear();

    for (size_t i = 0; i < names.size(); ++i) {
        string modelName = names[i];
        bool isUseAipp = Aipps[i];
        LOGI("[HIAI_DEMO_ASYNC] Get model %s IO Tensor. Use AIPP %d", modelName.c_str(), isUseAipp);
        vector<TensorDimension> inputDims, outputDims;
        ret = client_ptr->GetModelIOTensorDim(string(modelName) + string(".om"), inputDims, outputDims);
        if (ret != 0) {
            LOGE("[HIAI_DEMO_ASYNC] Get Model IO Tensor Dimension failed,ret is %d.", ret);
            return nullptr;
        }

        if (inputDims.size() == 0) {
            LOGE("[HIAI_DEMO_ASYNC] inputDims.size() == 0");
            return nullptr;
        }

        inputDimension.push_back(inputDims);
        outputDimension.push_back(outputDims);
        // two identical input tensors for async runmodel
        vector<shared_ptr<AiTensor>> inputTensors1, inputTensors2, outputTensors;
        // input 1
        for (auto in_dim : inputDims) {
            shared_ptr<AiTensor> input = make_shared<AiTensor>();
            if (isUseAipp) {
                ret = input->Init(in_dim.GetNumber(), in_dim.GetHeight(), in_dim.GetWidth(), AiTensorImage_YUV420SP_U8);
                LOGI("[HIAI_DEMO_ASYNC] model %s uses AIPP(input1).", modelName.c_str());
            } else {
                ret = input->Init(&in_dim);
                LOGI("[HIAI_DEMO_ASYNC] model %s does not use AIPP(input1).", modelName.c_str());
            }
            if (ret != 0) {
                LOGE("[HIAI_DEMO_ASYNC] model %s AiTensor Init failed(input1).", modelName.c_str());
                return nullptr;
            }
            inputTensors1.push_back(input);
        }
        input_tensor1_vec.push_back(inputTensors1);

        if (input_tensor1_vec.size() == 0) {
            LOGE("[HIAI_DEMO_ASYNC] input_tensor1_vec.size() == 0");
            return nullptr;
        }
        LOGE("[HIAI_DEMO_ASYNC] input_tensor1_vec.size %zu ", input_tensor1_vec.size());

        // input 2
        for (auto in_dim : inputDims) {
            shared_ptr<AiTensor> input = make_shared<AiTensor>();
            if (isUseAipp) {
                ret = input->Init(in_dim.GetNumber(), in_dim.GetHeight(), in_dim.GetWidth(), AiTensorImage_YUV420SP_U8);
                LOGI("[HIAI_DEMO_ASYNC] model %s uses AIPP(input2).", modelName.c_str());
            } else {
                ret = input->Init(&in_dim);
                LOGI("[HIAI_DEMO_ASYNC] model %s does not use AIPP(input2).", modelName.c_str());
            }
            if (ret != 0) {
                LOGE("[HIAI_DEMO_ASYNC] model %s AiTensor Init failed(input2).", modelName.c_str());
                return nullptr;
            }
            inputTensors2.push_back(input);
        }
        input_tensor2_vec.push_back(inputTensors2);

        if (input_tensor2_vec.size() == 0) {
            LOGE("[HIAI_DEMO_ASYNC] input_tensor2_vec.size() == 0");
            return nullptr;
        }
        LOGE("[HIAI_DEMO_ASYNC] input_tensor2_vec.size %zu ", input_tensor2_vec.size());

        // output
        for (auto out_dim : outputDims) {
            shared_ptr<AiTensor> output = make_shared<AiTensor>();
            ret = output->Init(&out_dim);
            if (ret != 0) {
                LOGE("[HIAI_DEMO_ASYNC] model %s AiTensor Init failed(output).", modelName.c_str());
                return nullptr;
            }
            outputTensors.push_back(output);
        }
        output_tensor_vec.push_back(outputTensors);
        // In this demo, model may has many output
        if (output_tensor_vec.size() == 0) {
            LOGE("[HIAI_DEMO_ASYNC] output_tensor_vec.size() == 0");
            return nullptr;
        }
    }
    return client_ptr;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_huawei_hiaidemo_utils_ModelManager_loadModelAsync(JNIEnv *env, jclass type,jobject modelInfo){

    jclass classList = env->GetObjectClass(modelInfo);
    if(classList == nullptr){
        LOGE("[HIAI_DEMO_ASYNC] can not find List class.");
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
            LOGE("[HIAI_DEMO_ASYNC] can not find getOfflineModelName method.");
            return nullptr;
        }
        if(getModelPath == nullptr){
            LOGE("[HIAI_DEMO_ASYNC] can not find getModelPath method.");
            return nullptr;
        }
        if(getUseAIPP == nullptr){
            LOGE("[HIAI_DEMO_ASYNC] can not find getUseAIPP method.");
            return nullptr;
        }

        jboolean useaipp = (jboolean)env->CallBooleanMethod(modelInfoObj,getUseAIPP);
        jstring modelname = (jstring)env->CallObjectMethod(modelInfoObj,getOfflineModelName);
        jstring modelpath = (jstring)env->CallObjectMethod(modelInfoObj,getModelPath);
        const char* modelName = env->GetStringUTFChars(modelname, 0);
        LOGE("[HIAI_DEMO_ASYNC] modelName is %s .",modelName);
        if(modelName == nullptr)
        {
            LOGE("[HIAI_DEMO_ASYNC] modelName is invalid.");
            return nullptr;
        }
        const char *modelPath = env->GetStringUTFChars(modelpath, 0);
        if(modelPath == nullptr)
        {
            LOGE("[HIAI_DEMO_ASYNC] modelPath is invalid.");
            return nullptr;
        }
        LOGE("[HIAI_DEMO_ASYNC] useaipp is %d.", bool(useaipp==JNI_TRUE));
        aipps.push_back(bool(useaipp==JNI_TRUE));
        names.push_back(string(modelName));
        modelPaths.push_back(string(modelPath));
    }

    // load
    if (!mclientAsync)
    {
        mclientAsync = LoadModelASync(names, modelPaths, aipps);
        if (mclientAsync == nullptr)
        {
            LOGE("[HIAI_DEMO_ASYNC] mclientAsync loadModel is nullptr.");
            return nullptr;
        }
    }

    // load model
    LOGI("[HIAI_DEMO_ASYNC] sync load model INPUT NCHW : %d %d %d %d." , inputDimension[0][0].GetNumber(), inputDimension[0][0].GetChannel(), inputDimension[0][0].GetHeight(), inputDimension[0][0].GetWidth());
    LOGI("[HIAI_DEMO_ASYNC] sync load model OUTPUT NCHW : %d %d %d %d." , outputDimension[0][0].GetNumber(), outputDimension[0][0].GetChannel(), outputDimension[0][0].GetHeight(), outputDimension[0][0].GetWidth());

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
JNIEXPORT void JNICALL
Java_com_huawei_hiaidemo_utils_ModelManager_runModelAsync(JNIEnv *env, jclass type, jobject modelInfo, jobject bufList, jobject callbacks)
{
    // check params
    if(env == nullptr)
    {
        LOGE("[HIAI_DEMO_ASYNC] runModelAsync env is null");
        return;
    }
    jclass ModelInfo = env->GetObjectClass(modelInfo);
    if(ModelInfo == nullptr)
    {
        LOGE("[HIAI_DEMO_ASYNC] can not find ModelInfo class.");
        return;
    }

    if (bufList == nullptr)
    {
        LOGE("[HIAI_DEMO_ASYNC] buf_ is null.");
        return;
    }

    jmethodID getOfflineModelName = env->GetMethodID(ModelInfo,"getOfflineModelName","()Ljava/lang/String;");
    jmethodID getModelPath = env->GetMethodID(ModelInfo,"getModelPath","()Ljava/lang/String;");

    if(getOfflineModelName == nullptr)
    {
        LOGE("[HIAI_DEMO_ASYNC] can not find getOfflineModelName method.");
        return;
    }
    if(getModelPath == nullptr){
        LOGE("[HIAI_DEMO_ASYNC] can not find getModelPath method.");
        return;
    }

    jstring modelname = (jstring)env->CallObjectMethod(modelInfo,getOfflineModelName);
    jstring modelpath = (jstring)env->CallObjectMethod(modelInfo,getModelPath);

    const char* modelName = env->GetStringUTFChars(modelname, 0);
    if(modelName == nullptr)
    {
        LOGE("[HIAI_DEMO_ASYNC] modelName is invalid.");
        return;
    }

    int vecIndex = async_nameToIndex[modelName];

    const char *modelPath = env->GetStringUTFChars(modelpath, 0);
    if(modelPath == nullptr)
    {
        LOGE("[HIAI_DEMO_ASYNC] modelPath is invalid.");
        return;
    }

    // buf_list
    jclass classList = env->GetObjectClass(bufList);
    if(classList == nullptr){
        LOGE("[HIAI_DEMO_ASYNC] can not find List class.");
    }
    // method in class
    jmethodID listGet = env->GetMethodID(classList, "get", "(I)Ljava/lang/Object;");
    jmethodID listSize = env->GetMethodID(classList, "size", "()I");

    if(listGet == nullptr){
        LOGE("[HIAI_DEMO_ASYNC] can not find get method.");
    }
    if(listSize == nullptr){
        LOGE("[HIAI_DEMO_ASYNC] can not find size method.");
    }

    int listLength = static_cast<int>(env->CallIntMethod(bufList, listSize));

    if(listLength > 0){
        LOGE("[HIAI_DEMO_ASYNC] input data length is %d .",listLength);
    }

    callbacksInstance = env->NewGlobalRef(callbacks);
    jclass objClass = env->GetObjectClass(callbacks);
    if (!objClass)
    {
        LOGE("[HIAI_DEMO_ASYNC] objClass is nullptr.");
        return;
    }
    env->GetJavaVM(&jvm);
    callbacksClass = reinterpret_cast<jclass>(env->NewGlobalRef(objClass));
    env->DeleteLocalRef(objClass);

    // load
    if (!mclientAsync)
    {
        LOGE("[HIAI_DEMO_ASYNC] mclientAsync is nullptr.");
        return;
    }

    env->ReleaseStringUTFChars(modelpath, modelPath);

    //run
    LOGI("[HIAI_DEMO_ASYNC] INPUT NCHW : %d %d %d %d." , inputDimension[0][0].GetNumber(), inputDimension[0][0].GetChannel(), inputDimension[0][0].GetHeight(), inputDimension[0][0].GetWidth());
    LOGI("[HIAI_DEMO_ASYNC] OUTPUT NCHW : %d %d %d %d." , outputDimension[0][0].GetNumber(), outputDimension[0][0].GetChannel(), outputDimension[0][0].GetHeight(), outputDimension[0][0].GetWidth());

    auto input_tensor0 = findInputTensor(vecIndex);

    for(int i = 0;i < listLength;i++){

        jbyteArray buf_ = (jbyteArray)(env->CallObjectMethod(bufList, listGet, i));
        if (buf_ == nullptr)
        {
            LOGE("[HIAI_DEMO_ASYNC] buf_ is nullptr.");
            return;
        }
        jbyte *dataBuff = nullptr;
        int databuffsize = 0;
        dataBuff = env->GetByteArrayElements(buf_, nullptr);
        databuffsize = env->GetArrayLength(buf_);

        if(((*input_tensor0)[i]->GetSize() != databuffsize))
        {
            LOGE("[HIAI_DEMO_ASYNC] input->GetSize(%d) != databuffsize(%d) ",(*input_tensor0)[i]->GetSize(),databuffsize);
            return ;
        }
        memmove((*input_tensor0)[i]->GetBuffer(), dataBuff, (size_t)databuffsize);
        env->ReleaseByteArrayElements(buf_, dataBuff, 0);
    }

    AiContext context;
    string key = "model_name";
    string value = modelName;
    value += ".om";
    context.AddPara(key, value);
    LOGI("[HIAI_DEMO_ASYNC] JNI runModel modelname:%s", value.c_str());

    int istamp = 0;
    gettimeofday(&tpstart, nullptr);
    int ret = mclientAsync->Process(context, *input_tensor0, output_tensor_vec[vecIndex], 300, istamp);
    if (ret != 0)
    {
        LOGE("[HIAI_DEMO_ASYNC] Runmodel Failed! ret=%d.",ret);
        return ;
    }


    LOGE("[HIAI_DEMO_ASYNC] Runmodel Succ! istamp=%d.",istamp);


    std::unique_lock<std::mutex> lock(mutex_map);
    map_input_tensor.insert(pair<int32_t,vector<shared_ptr<AiTensor>>*>(istamp, input_tensor0));
    condition_.notify_all();

    env->ReleaseStringUTFChars(modelname, modelName);
}
