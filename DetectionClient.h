#ifndef __DETECTION_CLIENT_H
#define __DETECTION_CLIENT_H

#include <string>
#include <grpcpp/grpcpp.h>
#include <fstream>
#include <android/log.h>
#include <log/log.h>
#include <android-base/logging.h>
#include "nnhal_object_detection.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientWriter;
using grpc::Status;
using objectDetection::DataTensor;
using objectDetection::Detection;
using objectDetection::ReplyDataTensors;
using objectDetection::ReplyStatus;
using objectDetection::RequestDataChunks;
using objectDetection::RequestDataTensors;
using objectDetection::RequestString;
using time_point = std::chrono::system_clock::time_point;

static std::string IR_XML("/data/vendor/neuralnetworks/ngraph_ir.xml");
static std::string IR_BIN("/data/vendor/neuralnetworks/ngraph_ir.bin");

class DetectionClient {
public:
    DetectionClient(std::shared_ptr<Channel> channel) : stub_(Detection::NewStub(channel)){}

    std::string prepare(bool& flag);

    Status sendFile(std::string fileName,
                    std::unique_ptr<ClientWriter<RequestDataChunks> >& writer);

    std::string sendIRs(bool& flag);

    void add_input_data(std::string label, const uint8_t* buffer, std::vector<size_t> shape, uint32_t size);
    void get_output_data(std::string label, uint8_t* buffer, std::vector<size_t> shape);
    void clear_data();
    std::string remote_infer();
    bool get_status();

private:
    std::unique_ptr<Detection::Stub> stub_;
    RequestDataTensors request;
    ReplyDataTensors reply;
    Status status;
};

#endif