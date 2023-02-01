#include "DetectionClient.h"

#undef LOG_TAG
#define LOG_TAG "DetectionClient"

std::string DetectionClient::prepare(bool& flag) {
    RequestString request;
    request.set_value("");
    ReplyStatus reply;
    ClientContext context;
    time_point deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(100);
    context.set_deadline(deadline);

    Status status = stub_->prepare(&context, request, &reply);

    if (status.ok()) {
        flag = reply.status();
        return (flag ? "status True" : "status False");
    } else {
        return std::string(status.error_message());
    }
}

Status DetectionClient::sendFile(std::string fileName,
                std::unique_ptr<ClientWriter<RequestDataChunks> >& writer) {
    RequestDataChunks request;
    uint32_t CHUNK_SIZE = 1024 * 1024;
    std::ifstream fin(fileName, std::ifstream::binary);
    std::vector<char> buffer(CHUNK_SIZE, 0);
    ALOGV("GRPC sendFile %s", fileName.c_str());
    ALOGI("GRPC sendFile %d sized chunks", CHUNK_SIZE);

    if (!fin.is_open()) ALOGE("GRPC sendFile file Open Error ");
    while (!fin.eof()) {
        fin.read(buffer.data(), buffer.size());
        std::streamsize s = fin.gcount();
        // ALOGI("GRPC sendFile read %d", s);
        request.set_data(buffer.data(), s);
        if (!writer->Write(request)) {
            ALOGE("GRPC Broken Stream ");
            break;
        }
    }

    writer->WritesDone();
    ALOGI("GRPC sendFile completed");
    return writer->Finish();
}

std::string DetectionClient::sendIRs(bool& flag) {
    ReplyStatus reply;
    ClientContext context;
    std::unique_ptr<ClientWriter<RequestDataChunks> > writerXml =
        std::unique_ptr<ClientWriter<RequestDataChunks> >(stub_->sendXml(&context, &reply));
    Status status = sendFile(IR_XML, writerXml);

    if (status.ok()) {
        ClientContext newContext;
        std::unique_ptr<ClientWriter<RequestDataChunks> > writerBin =
            std::unique_ptr<ClientWriter<RequestDataChunks> >(
                stub_->sendBin(&newContext, &reply));
        status = sendFile(IR_BIN, writerBin);
        if (status.ok()) {
            flag = reply.status();
            return (flag ? "status True" : "status False");
        }
    }
    return std::string(status.error_message());
}

void DetectionClient::add_input_data(std::string label, const uint8_t* buffer, std::vector<size_t> shape) {
    const float* src;
    size_t index;
    size_t size = 1;

    DataTensor* input = request.add_data_tensors();
    input->set_node_name(label);
    for (index = 0; index < shape.size(); index++) {
        input->add_tensor_shape(shape[index]);
        size *= shape[index];
    }
    input->set_data(buffer, size * sizeof(float));
}

void DetectionClient::get_output_data(std::string label, uint8_t* buffer, std::vector<size_t> shape) {
    std::string src;
    size_t index;
    size_t size = 1;

    for (index = 0; index < shape.size(); index++) {
        size *= shape[index];
    }
    for (index = 0; index < reply.data_tensors_size(); index++) {
        if (label.compare(reply.data_tensors(index).node_name()) == 0) {
            src = reply.data_tensors(index).data();
            memcpy(buffer, src.data(), src.length());
            break;
        }
    }
}

void DetectionClient::clear_data() {
    request.clear_data_tensors();
    reply.clear_data_tensors();
}

std::string DetectionClient::remote_infer() {
    ClientContext context;
    time_point deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(1000);
    context.set_deadline(deadline);

    status = stub_->getInferResult(&context, request, &reply);
    if (status.ok()) {
        if (reply.data_tensors_size() == 0) ALOGE("GRPC reply empty, ovms failure ?");
        return "Success";
    } else {
        ALOGE("GRPC Error code: %d, message: %s", status.error_code(),
                status.error_message().c_str());
        return std::string(status.error_message());
    }
}

bool DetectionClient::get_status() {
    if (status.ok() && (reply.data_tensors_size() > 0))
        return 1;
    else {
        return 0;
    }
}