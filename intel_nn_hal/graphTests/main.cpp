#include <fstream>
#include "helpers-test.hpp"

#include <android/log.h>
#include <log/log.h>

using namespace IRBuilder;

template <typename T>
void createAlexNet(IRDocument &doc) {
    auto input = doc.createInput("in1", {1, 3, 227, 227});

    auto &pp = input->getPreProcess();

    pp.init(3);
    pp.setVariant(MEAN_VALUE);  // reverse of 123.68, 116.779, 103.939
    pp[0]->meanValue = 103.939f;
    pp[1]->meanValue = 116.779f;
    pp[2]->meanValue = 123.680f;

    ConvolutionParams prms;
    //<convolution_data stride-x="4" stride-y="4" pad-x="0" pad-y="0" kernel-x="11" kernel-y="11"
    //output="96" group="1"/>
    prms.weights = readBlobFromFile<T>("AlexNet-bins/conv1_weights.bin");
    prms.kernel = {11, 11};
    prms.stride = {4, 4};
    prms.num_output_planes = 96;

    auto b1 = readBlobFromFile<T>("AlexNet-bins/conv1_bias.bin");
    auto c1 = Convolution(input->getInputData(), prms) + b1;
    auto r1 = ReLU(c1);
    //<norm_data alpha = "9.9999997e-05" beta = "0.75" local-size = "5" region = "across" / >
    auto n1 = LRN(r1, 0.0001f, 0.75f, 5, true);
    //<pooling_data kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" stride-x="2" stride-y="2"
    //rounding-type="ceil" pool-method="max"/>
    auto p1 = Pooling(n1, {3, 3}, {2, 2}, {0, 0}, PoolingLayer::MAX);

    /// 2nd part
    auto s1 = Split(p1, 2);
    //<convolution_data stride-x="1" stride-y="1" pad-x="2" pad-y="2" kernel-x="5" kernel-y="5"
    //output="128" group="1"/>
    prms.kernel = {5, 5};
    prms.stride = {1, 1};
    prms.pad_start = {2, 2};
    prms.pad_end = {2, 2};
    prms.num_output_planes = 128;

    prms.weights = readBlobFromFile<T>("AlexNet-bins/conv2_0_weights.bin");
    auto c2_0 = Convolution(s1[0], prms) + readBlobFromFile<T>("AlexNet-bins/conv2_0_bias.bin");
    prms.weights = readBlobFromFile<T>("AlexNet-bins/conv2_1_weights.bin");
    auto c2_1 = Convolution(s1[1], prms) + readBlobFromFile<T>("AlexNet-bins/conv2_1_bias.bin");
    auto c2 = Concat({c2_0, c2_1});
    auto relu2 = ReLU(c2);
    auto n2 = LRN(relu2, 0.0001f, 0.75f, 5, true);
    //<pooling_data kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" stride-x="2" stride-y="2"
    //rounding-type="ceil" pool-method="max"/>
    auto p2 = Pooling(n2, {3, 3}, {2, 2}, {0, 0}, PoolingLayer::MAX);

    // 3rd part
    //<convolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="3" kernel-y="3"
    //output="384" group="1"/>
    prms.kernel = {3, 3};
    prms.stride = {1, 1};
    prms.pad_start = {1, 1};
    prms.pad_end = {1, 1};
    prms.num_output_planes = 384;
    prms.weights = readBlobFromFile<T>("AlexNet-bins/conv3_weights.bin");
    auto c3 = ReLU(Convolution(p2, prms) + readBlobFromFile<T>("AlexNet-bins/conv3_bias.bin"));

    // 4th part
    auto s4 = Split(c3, 2);
    //<convolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="3" kernel-y="3"
    //output="192" group="1"/>
    prms.num_output_planes = 192;
    prms.weights = readBlobFromFile<T>("AlexNet-bins/conv4_0_weights.bin");
    auto c4_0 =
        ReLU(Convolution(s4[0], prms) + readBlobFromFile<T>("AlexNet-bins/conv4_0_bias.bin"));
    prms.weights = readBlobFromFile<T>("AlexNet-bins/conv4_1_weights.bin");
    auto c4_1 =
        ReLU(Convolution(s4[1], prms) + readBlobFromFile<T>("AlexNet-bins/conv4_1_bias.bin"));

    // 5th part
    //<convolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="3" kernel-y="3"
    //output="128" group="1"/>
    prms.num_output_planes = 128;
    prms.weights = readBlobFromFile<T>("AlexNet-bins/conv5_0_weights.bin");
    auto c5_0 =
        ReLU(Convolution(c4_0, prms) + readBlobFromFile<T>("AlexNet-bins/conv5_0_bias.bin"));
    prms.weights = readBlobFromFile<T>("AlexNet-bins/conv5_1_weights.bin");
    auto c5_1 =
        ReLU(Convolution(c4_1, prms) + readBlobFromFile<T>("AlexNet-bins/conv5_1_bias.bin"));
    auto c5 = Concat({c5_0, c5_1});

    //<pooling_data kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" stride-x="2" stride-y="2"
    //rounding-type="ceil" pool-method="max"/>
    auto p5 = Pooling(c5, {3, 3}, {2, 2}, {0, 0}, PoolingLayer::MAX);

    // fc6
    auto m = readBlobFromFile<T>("AlexNet-bins/fc6_weights.bin");
    auto fc6 = ReLU(m * Reshape({1, 9216}, p5) + readBlobFromFile<T>("AlexNet-bins/fc6_bias.bin"));

    // fc7
    m = readBlobFromFile<T>("AlexNet-bins/fc7_weights.bin");
    auto fc7 = ReLU(m * fc6 + readBlobFromFile<T>("AlexNet-bins/fc7_bias.bin"));
    // fc8
    m = readBlobFromFile<T>("AlexNet-bins/fc8_weights.bin");
    auto fc8 = m * fc7 + readBlobFromFile<T>("AlexNet-bins/fc8_bias.bin");

    const OutputPort &output = Softmax(fc8);
    doc.addOutput(output);
    return;
}

inline void prompt(const char *msg = "press [ENTER] to contine") {
#ifdef WIN32
    std::cout << msg << std::endl;
    std::cin.get();
#endif
}

bool testAlexNet() {
    std::string tmpStr;
    IRDocument doc("my-alexnet");

    tmpStr = FileUtils::GetCWD();
    std::cout << "starting from: " << tmpStr << std::endl;
    try {
#ifdef ENABLE_MYRIAD
        createAlexNet<short>(doc);
#elif ENABLE_MKLDNN
        createAlexNet<float>(doc);
#endif
    } catch (const InferenceEngine::details::InferenceEngineException &ex) {
        std::cerr << "IE Expection: " << ex.what();
        prompt();
        return false;
    } catch (const std::exception &ex) {
        std::cerr << ex.what();
        prompt();
        return false;
    }

    std::fstream xml, bin;

    doc.save("test-alexnet");

    // try it out
    try {
#ifdef ENABLE_MYRIAD
        auto in = readBlobFromFile<short>("AlexNet-bins/input.bin", {1, 3, 227, 227},
                                          InferenceEngine::NCHW);
#elif ENABLE_MKLDNN
        auto in = readBlobFromFile<float>("AlexNet-bins/input.bin", {1, 3, 227, 227},
                                          InferenceEngine::NCHW);
#endif
        /*
                        InfEng eng(TargetDevice::eMYRIAD);

                        auto exe = eng.Load(doc);

                        auto ob = exe.Infer(in);
        */
        printf("aks initialize ExecuteNetwork\n");
#ifdef ENABLE_MYRIAD
        ExecuteNetwork executeNet(doc, TargetDevice::eMYRIAD);
#elif ENABLE_MKLDNN
        ExecuteNetwork executeNet(doc, TargetDevice::eCPU);
#endif
        executeNet.prepareInput();
        executeNet.prepareOutput();
        printf("load network\n");
        executeNet.loadNetwork();
        printf("infer network\n");
        auto ob = executeNet.Infer(in);

        dumpBlob("Input: ", 50, in);
        dumpBlob("Output: ", 50, ob);

#ifndef DONT
        std::vector<unsigned> top10;
        InferenceEngine::TopResults<float>(10, *ob, top10);

        const auto mem = ob->readOnly();

        // const float *pf = mem.as<const float*>();
        const float *pf = mem.as<const float *>();

        for (int i = 0; i < top10.size(); i++) {
            std::cout << "TOP " << i + 1 << ": " << pf[top10[i]] * 100 << "% at index " << top10[i]
                      << std::endl;
        }
#endif
    } catch (const std::exception &ex) {
        std::cerr << ex.what();
        return false;
    }
    return true;
}

bool testAffineLayer() {
    std::string tmpStr;
    try {
        std::string netName("TestNet");
        IRDocument doc("TestNet");

        size_t batch = 1;

        printf("create network input\n");
        vec<uint32_t> indims = {1, 5};

        auto input = doc.createInput("input", toDims(indims));  // 5 elements, batch of 1
        // auto input = doc.createInput("input", { 5 }); // 5 elements, batch of 1

        std::vector<float> wdata = {2.0f, 4.0f, 0.5f, 0.25f, 1.0f};
        float *buf = wdata.data();

        vec<uint32_t> wdims = {1, 5};
        // vec<uint32_t> wdims = { 5, 1 };
#ifdef ENABLE_MYRIAD
        TensorDesc td(InferenceEngine::Precision::FP16, toDims(wdims), Layout::NC);

        InferenceEngine::TBlob<short>::Ptr weightsBlob =
            std::make_shared<InferenceEngine::TBlob<short>>(td);
        weightsBlob->allocate();
        auto mem = weightsBlob->data();
        short *fp16Array = mem.as<short *>();

        // convert from [(float *)buf, len] to fp16Array,

        uint32_t nelem = getNumberOfElements(wdims);
        printf("weights nelement %u\n", nelem);

        f32tof16Arrays(fp16Array, (float *)buf,
                       nelem);  // void f32tof16Arrays(short *dst, const float *src, uint32_t&
                                // nelem, float scale = 1, float bias = 0)
#elif ENABLE_MKLDNN
        TensorDesc td(InferenceEngine::Precision::FP32, toDims(wdims), Layout::NC);

        InferenceEngine::TBlob<float>::Ptr weightsBlob =
            std::make_shared<InferenceEngine::TBlob<float>>(td);
        weightsBlob->set(wdata);
#endif

        // auto weights = make_shared_blob<short>(Precision::FP32, Layout::C,
        // std::vector<float>({ 2.0f,3.0f,0.1f,5.0f,1.0f })); auto weights =
        // make_shared_blob<short>(Precision::FP16, Layout::NC, std::vector<short>({ 2,3,1,5,1 }));

        // Create Bias blob
        std::vector<float> bData = {1.0f};

        float *buf1 = bData.data();

        vec<uint32_t> bdims = {1};
#ifdef ENABLE_MYRIAD
        TensorDesc td1(InferenceEngine::Precision::FP16, toDims(bdims), /*Layout::ANY*/ Layout::C);

        InferenceEngine::TBlob<short>::Ptr biasBlob =
            std::make_shared<InferenceEngine::TBlob<short>>(td1);
        biasBlob->allocate();
        auto mem1 = biasBlob->data();
        short *fp16Array1 = mem1.as<short *>();

        // convert from [(float *)buf, len] to fp16Array,
        uint32_t nelem1 = getNumberOfElements(bdims);
        // printf("bias nelement %u\n", nelem);

        f32tof16Arrays(fp16Array1, (float *)buf1,
                       nelem1);  // void f32tof16Arrays(short *dst, const float *src, uint32_t&
                                 // nelem, float scale = 1, float bias = 0)
#elif ENABLE_MKLDNN
        TensorDesc td1(InferenceEngine::Precision::FP32, toDims(bdims), /*Layout::ANY*/ Layout::C);

        InferenceEngine::TBlob<float>::Ptr biasBlob =
            std::make_shared<InferenceEngine::TBlob<float>>(td1);
        biasBlob->set(bData);
#endif

        // auto bias = make_shared_blob<short>(Precision::FP32, Layout::C, std::vector<float>({
        // -1.0f })); auto bias = make_shared_blob<short>(Precision::FP16, Layout::C,
        // std::vector<short>({ 1 }));

        printf("addOutput\n");
        auto indim = input->getDims();
        std::cout << "input indim[0] " << indim[0] << " indim[1] " << indim[1] << std::endl;
        auto inputdata = input->getInputData();
        auto indatadim = inputdata->getDims();
        std::cout << "inputdata indatadim[0] " << indatadim[0] << " indatadim[1] " << indatadim[1]
                  << std::endl;

        doc.addOutput(ReLU(weightsBlob * input->getInputData() + biasBlob));

        doc.buildNetwork();
        std::fstream dot;
        std::string graphfile("/data/graphtest-file");
        dot.open("/data/graphtest.dot", std::ios::out);
        // save it
        doc.save(graphfile);
        doc.crateDotFile(dot);
        dot.close();

        // save it
        // doc.save("TestNet");

        // test it
        std::cout << "create input blob" << std::endl;

        std::vector<float> invalue = {0.5f, 0.25f, 1.0f, 2.0f, 0.5f};
        // float* inbuf = invalue.data();

        TensorDesc intd(InferenceEngine::Precision::FP32, toDims(indims), Layout::ANY);

        InferenceEngine::TBlob<float>::Ptr inData =
            std::make_shared<InferenceEngine::TBlob<float>>(intd);
        inData->allocate();

        for (size_t i = 0; i < invalue.size(); i++) {
            inData->data()[i] = invalue.at(i);
        }

        // auto inData = make_shared_blob<float>(Precision::FP32, Layout::NC, std::vector<float>({
        // 0.1f,0.2f,0.3f,0.4f,0.5f }));

        auto network = doc.getNetwork();
        network->setBatchSize(batch);
        char networkName1[1024] = {};
        network->getName(networkName1, sizeof(networkName1));  // aks
        std::cout << "graphtest network name " << networkName1 << std::endl;

        printf("aks initialize ExecuteNetwork\n");
#ifdef ENABLE_MYRIAD
        ExecuteNetwork executeNet(doc, TargetDevice::eMYRIAD);
#elif ENABLE_MKLDNN
        ExecuteNetwork executeNet(doc, TargetDevice::eCPU);
#endif
        executeNet.prepareInput();
        executeNet.prepareOutput();
        printf("load network\n");
        executeNet.loadNetwork();
        printf("infer network\n");
        auto ob = executeNet.Infer(inData);

        // Check result
        float expectedResult = bData.data()[0];

        for (int i = 0; i < weightsBlob->size(); i++) {
            expectedResult += wdata.data()[i] * inData->readOnly()[i];
            // expectedResult += weightsBlob->readOnly().as<short*>()[i] * inData_mem16[i];
        }
        if (expectedResult < 0) expectedResult = 0;

        auto result = ob->readOnly()[0];
        std::cout << "Affine Layer output: size=" << ob->size() << " value: " << result
                  << std::endl;
        ALOGI("Affine Layer output: size= %d value %f ", ob->size(), result);
        printf("Affine Layer output: size= %d value %f ", ob->size(), result);
        if (fabsf(expectedResult - result) < 1E-3) {
            printf("expected: %f got result %f ", expectedResult, result);
            std::cout << "TEST OK!" << std::endl;
            ALOGI("TEST OK!");
        } else {
            std::cout << "TEST FAILED! expected:" << expectedResult << " Got: " << result
                      << std::endl;
            printf("TEST FAILED! expected: %f got result %f ", expectedResult, result);
        }
        return true;
    } catch (const std::exception &ex) {
        printf("exception\n");
        std::cerr << ex.what();

        prompt();
        return false;
    }
}

template <typename T>
bool testMKLBug() {
    std::string tmpStr;
    try {
        IRDocument doc("test_bug");

        auto input = doc.createInput("in1", {1, 3, 227, 227});

        ConvolutionParams prms;
        //<convolution_data stride-x="4" stride-y="4" pad-x="0" pad-y="0" kernel-x="11"
        //kernel-y="11" output="96" group="1"/>
        prms.weights = readBlobFromFile<T>("AlexNet-bins/conv1_weights.bin");
        prms.kernel = {11, 11};
        prms.stride = {4, 4};
        prms.num_output_planes = 96;

        auto b1 = readBlobFromFile<T>("AlexNet-bins/conv1_bias.bin");
        auto r1 = ReLU(Convolution(input->getInputData(), prms) + b1);
        //<norm_data alpha = "9.9999997e-05" beta = "0.75" local-size = "5" region = "across" / >
        auto n1 = LRN(r1, 0.0001f, 0.75f, 5, true);
        //<pooling_data kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" stride-x="2" stride-y="2"
        //rounding-type="ceil" pool-method="max"/>
        auto p1 = Pooling(n1, {3, 3}, {2, 2}, {0, 0}, PoolingLayer::MAX);

        /// 2nd part
        auto s1 = Split(p1, 2);
        //<convolution_data stride-x="1" stride-y="1" pad-x="2" pad-y="2" kernel-x="5" kernel-y="5"
        //output="128" group="1"/>
        prms.kernel = {5, 5};
        prms.stride = {1, 1};
        prms.pad_end = prms.pad_start = {2, 2};
        prms.num_output_planes = 128;

        prms.weights = readBlobFromFile<T>("AlexNet-bins/conv2_0_weights.bin");
        auto c2_0 = Convolution(s1[0], prms) + readBlobFromFile<T>("AlexNet-bins/conv2_0_bias.bin");
        prms.weights = readBlobFromFile<T>("AlexNet-bins/conv2_1_weights.bin");
        auto c2_1 = Convolution(s1[1], prms) + readBlobFromFile<T>("AlexNet-bins/conv2_1_bias.bin");
        auto conv2 = Concat({c2_0, c2_1});
        auto c2 = ReLU(conv2);
        auto n2 = LRN(c2, 0.0001f, 0.75f, 5, true);
        //<pooling_data kernel-x="3" kernel-y="3" pad-x="0" pad-y="0" stride-x="2" stride-y="2"
        //rounding-type="ceil" pool-method="max"/>
        auto p2 = Pooling(n2, {3, 3}, {2, 2}, {0, 0}, PoolingLayer::MAX);

        doc.addOutput(c2);
        doc.addOutput(p2);

        auto in = readBlobFromFile<T>("AlexNet-bins/input.bin", {1, 3, 227, 227}, Layout::NCHW);
        /*
                        InfEng eng(TargetDevice::eMYRIAD);

                        auto ob = eng.Load(doc).Infer(in);
        */
        printf("aks initialize ExecuteNetwork\n");
        ExecuteNetwork executeNet(doc, TargetDevice::eMYRIAD);
        executeNet.prepareInput();
        printf("load network\n");
        executeNet.loadNetwork();
        printf("infer network\n");
        auto ob = executeNet.Infer(in);
    } catch (const std::exception &ex) {
        std::cerr << ex.what();
        return false;
    }
    return true;
}

int main(int argc, const char *argv[]) {
    std::string inp;

#ifdef ENABLE_MYRIAD
    IRBuilder::g_layer_precision = InferenceEngine::Precision::FP16;
    // testAlexNet();
    // testMKLBug<short>();
#elif ENABLE_MKLDNN
    IRBuilder::g_layer_precision = InferenceEngine::Precision::FP32;
    // testAlexNet();
    // testMKLBug<float>();
#endif

    testAffineLayer();

    prompt("enter string to exit\n");
    return 0;
}
