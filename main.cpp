#include <iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParserRuntime.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include "common.h"
#include "logging.h"
#include <chrono>
#include <functional>

// Just for convenience
using Seconds = std::chrono::duration<double>;

// Measure how much time the given function takes to execute using chrono
// Pass the function name, then all relevant arguments, including the object as the first if it's a member function
template<typename Function, typename... Args>
Seconds measure(Function&& toTime, Args&&... a)
{
    auto start{std::chrono::steady_clock::now()};                                                   // Start timer
    std::invoke(std::forward<Function>(toTime), std::forward<Args>(a)...);  // Forward and call
    auto stop{std::chrono::steady_clock::now()};                                                   // Stop timer
    return (stop - start);
}

using namespace nvinfer1;

static Logger mLogger;
Logger gLoggerSample{Logger::Severity::kINFO};
LogStreamConsumer gLogError{LOG_ERROR(gLoggerSample)};

int maxBatchSize;
int OUTPUT_C;
int INPUT_H;
int INPUT_W;
int INPUT_C;
std::string modelFile;
std::string enginePath;
void* buffers[2];

IHostMemory* trtModelStream{nullptr};

void doInference(IExecutionContext& context, const ICudaEngine& engine, float**& io_buffers, size_t &size_in, size_t &size_out, int &inputIndex, int batchSize);
void saveEngine();
void readEngine(std::unique_ptr<char[]> &data, size_t &data_len);
void buildEngine();
void createRandomBuffers(const ICudaEngine& engine, float**& io_buffers, size_t &size_in, size_t &size_out, int &inputIndex);


int main(){
    modelFile = "/home/smocilac/dipl_seminar/swiftnet/swiftnet.onnx";
    enginePath = "/home/smocilac/dipl_seminar/dipl_seminar/swiftnet.trt";
    maxBatchSize = 1;
    OUTPUT_C = 19;
    INPUT_H = 1024;
    INPUT_W = 2048;
    INPUT_C = 3;
    
    bool createNewEngine = true;

    if (createNewEngine)
        buildEngine();
    
    std::unique_ptr<char[]> data;
    size_t data_len;
    readEngine(data, data_len);
    //std::cout << data_len << std::endl;
    //exit(-1);

    // deserialize the engine
    IRuntime* runtime = createInferRuntime(mLogger.getTRTLogger());
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(data.get(), data_len, nullptr);
    assert(engine != nullptr);
    data.release();
    
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // run inference
    float **buffers;
    size_t size_in, size_out;
    int inputIndex;
    const ICudaEngine& ctxEngine = context->getEngine();
    
    createRandomBuffers(ctxEngine, buffers, size_in, size_out, inputIndex);
    doInference(*context, ctxEngine, buffers, size_in, size_out, inputIndex, maxBatchSize);
    
    //auto t1 = std::chrono::high_resolution_clock::now();
    //for (int i = 0; i < 1000; i++)
        doInference(*context, ctxEngine, buffers, size_in, size_out, inputIndex, maxBatchSize);
    //auto t2 = std::chrono::high_resolution_clock::now();

    //int timeAvg = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000;

    //std::cout << "inference timing: " << timeAvg << " ms. " << std::endl;

    // for (int b = 0; b < 2; ++b) 
    // {
    //     CHECK(cudaFree(buffers[b]));
    // }

    // destroy the engine2
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}


void doInference(IExecutionContext& context, const ICudaEngine& engine, float**& io_buffers, size_t &size_in, size_t &size_out, int &inputIndex, int batchSize)
{
    assert(engine.getNbBindings() == 2);
    
    // create GPU buffers and a stream
    int outputIndex = (inputIndex + 1) % 2;
    
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], io_buffers[inputIndex], size_in * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++)
        context.enqueue(batchSize, buffers, stream, nullptr);    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    int timeAvg = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000;
    std::cout << "inference timing: " << timeAvg << " ms. " << std::endl;

    
    CHECK(cudaMemcpyAsync(io_buffers[outputIndex], buffers[outputIndex], size_out * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    
    // release the stream and the buffers
    cudaStreamDestroy(stream);
}

void buildEngine(){

    // Create builder and network definition object neccessary for building engine
    IBuilder* builder = createInferBuilder(mLogger.getTRTLogger());
    assert(builder != nullptr);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    // now we are able to create parser with network object
    auto parser = nvonnxparser::createParser(*network, mLogger.getTRTLogger());

    // print conversion
    // config->setPrintLayerInfo(true);
    // parser->reportParsingInfo();

    // parse 
    if ( !parser->parseFromFile( modelFile.c_str(), static_cast<int>(mLogger.getReportableSeverity()) ) )
    {
        gLogError << "Failure while parsing ONNX file" << std::endl;
        exit(-1);
    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    builder->setFp16Mode(false);
    builder->setInt8Mode(false);
    
    int nbOuts = network->getNbOutputs();
    std::cout << "Found " << nbOuts << " output tensors.\n";
    for (int i = 1; i < nbOuts; i++){
        nvinfer1::Dims outDim = network->getOutput(1)->getDimensions();
        std::cout << "Unmarking output tensor " << i << ": ";
        for (int j = 0; j < outDim.nbDims; j++){
            std::cout << outDim.d[j] << " ";
        }
        std::cout << std::endl;
        network->unmarkOutput(*network->getOutput(1));
    }
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    saveEngine();
    engine->destroy();
    network->destroy();
    builder->destroy();
}


void saveEngine()
{
    if(trtModelStream)
    {
        std::ofstream file;
        file.open(enginePath,std::ios::binary | std::ios::out);
        if(!file.is_open())
        {
            std::cout << "create engine file " << enginePath <<" failed" << std::endl;
            return;
        }

        file.write((const char*)trtModelStream->data(), trtModelStream->size());
        file.close();
    }
};

void readEngine(std::unique_ptr<char[]> &data, size_t &data_len)
{
    std::ifstream file;
    file.open(enginePath,std::ios::binary | std::ios::in);
    if(!file.is_open())
    {
        std::cout << "read engine file " << enginePath <<" failed" << std::endl;
        return;
    }
    file.seekg(0, ios::end); 
    data_len = file.tellg();         
    file.seekg(0, ios::beg); 
    data = std::unique_ptr<char[]>(new char[data_len]);
    //std::cout << data_len << std::endl;

    file.read(data.get(), data_len);
    file.close();
};

void createRandomBuffers(const ICudaEngine& engine, float**& io_buffers, size_t &size_in, size_t &size_out, int &inputIndex){
    const int const_nbBindings = engine.getNbBindings();
    
    assert(const_nbBindings == 2);
    int totalSizes[const_nbBindings];
    inputIndex = -1;
    
    std::cout << "NbBindings: " << const_nbBindings << std::endl;
    io_buffers =  (float **) malloc (const_nbBindings * sizeof(float*));

    for (int b = 0; b < const_nbBindings; ++b)
    {
        nvinfer1::Dims dims = engine.getBindingDimensions(b);
        int totalSize = 1;
        std::cout << "indeks " << b << ": ";
        for (int j = 0; j < dims.nbDims; j++){
            totalSize *= dims.d[j];
            std::cout << dims.d[j] << " ";
        }
        totalSizes[b] = totalSize;
        std::cout << " ; total size = " << totalSize ;
        if (engine.bindingIsInput(b)){
            assert(inputIndex == -1);
            inputIndex = b; 
            size_in = totalSize;
        } else {
            size_out = totalSize;
        }
        io_buffers[b] = new float[totalSize];
        
        std::cout << " ; is input " << (inputIndex == b) << std::endl;
    }
    CHECK(cudaMalloc(&buffers[inputIndex], size_in * sizeof(float)));
    CHECK(cudaMalloc(&buffers[(inputIndex + 1) % 2], size_out * sizeof(float)));


}
