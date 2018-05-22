#ifndef __NC_H_INCLUDED__
#define __NC_H_INCLUDED__

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
	NC_OK = 0,
	NC_BUSY = -1,                     // Device is busy, retry later
	NC_ERROR = -2,                    // Error communicating with the device
	NC_OUT_OF_MEMORY = -3,            // Out of memory
	NC_DEVICE_NOT_FOUND = -4,         // No device at the given index or name
	NC_INVALID_PARAMETERS = -5,       // At least one of the given parameters is wrong
	NC_TIMEOUT = -6,                  // Timeout in the communication with the device
	NC_MVCMD_NOT_FOUND = -7,          // The file to boot Myriad was not found
	NC_NOT_ALLOCATED = -8,            // The graph or device has been closed during the operation
	NC_UNAUTHORIZED = -9,             // Unauthorized operation
	NC_UNSUPPORTED_GRAPH_FILE = -10,  // The graph file version is not supported
	NC_UNSUPPORTED_CONFIGURATION_FILE = -11, // The configuration file version is not supported
	NC_UNSUPPORTED_FEATURE = -12,     // Not supported by this FW version
	NC_MYRIAD_ERROR = -13,            // An error has been reported by the device
                                      // use  NC_DEVICE_DEBUG_INFO or NC_GRAPH_DEBUG_INFO
} ncStatus_t;

typedef enum {
    NC_OPTION_CLASS0 = 0,
    NC_OPTION_CLASS1 = 1,
    NC_OPTION_CLASS2 = 2,
    NC_OPTION_CLASS3 = 3,
} ncOptionClass_t;

typedef enum {
	NC_RW_LOG_LEVEL = 0, // Log level, int, 0 = nothing, 1 = errors, 2 = verbose
    NC_RO_API_VER = 1,   // retruns API Version. string
} ncGlobalOptions_t;

typedef enum {
    NC_RO_GRAPH_STATE = 1000, // Returns graph state: INITIALIZED, READY, WAITING_FOR_INPUT, RUNNING
    NC_RO_GRAPH_TIME_TAKEN = 1001,   // Return time taken for last inference (float *)
    NC_RO_GRAPH_INPUT_COUNT = 1002,  // Returns number of inputs, size of array returned
                                     // by NC_RO_INPUT_TENSOR_DESCRIPTORS, int
    NC_RO_GRAPH_OUTPUT_COUNT = 1003, // Returns number of outputs, size of array returned
                                     // by NC_RO_OUTPUT_TENSOR_DESCRIPTORS,int
    NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS = 1004,  // Return a tensorDescriptor pointer array
                                            // which describes the graph inputs in order.
                                            // Can be used for fifo creation.
                                            // The length of the array can be retrieved
                                            // using the NC_RO_INPUT_COUNT option

    NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS = 1005, // Return a tensorDescriptor pointer
                                            // array which describes the graph
                                            // outputs in order. Can be used for
                                            // fifo creation. The length of the
                                            // array can be retrieved using the
                                            // NC_RO_OUTPUT_COUNT option

    NC_RO_GRAPH_DEBUG_INFO = 1006,  // Return debug info, string
    NC_RO_GRAPH_NAME = 1007,        // Returns name of the graph, string
    NC_RO_GRAPH_OPTION_CLASS_LIMIT = 1008,  // return the highest option class supported
    NC_RO_GRAPH_VERSION = 1009,     // returns graph version, string
} ncGraphOptionsClass0_t;

typedef enum {
    NC_RW_GRAPH_BATCH_SIZE = 1108,  // configure batch size. 1 by default
    NC_RW_GRAPH_RUNTIME_CONFIG = 1109,  //blob config file with resource hints
    NC_RW_GRAPH_EXECUTORS_NUM = 1110,
} ncGraphOptionsClass1_t;

typedef enum {
    NC_DEVICE_INITIALIZED = 0,
    NC_DEVICE_OPENED = 1,
    NC_DEVICE_CLOSED = 2,
} ncDeviceState_t;

typedef enum {
    NC_GRAPH_INITIALIZED = 0,
    NC_GRAPH_READY = 1,
    NC_GRAPH_WAITING_FOR_INPUT = 2,
    NC_GRAPH_RUNNING = 3,
    NC_GRAPH_DEALLOCATED = 4,
} ncGraphState_t;

typedef enum {
    NC_FIFO_INITIALIZED = 0,
    NC_FIFO_CREATED = 1,
    NC_FIFO_DESTROYED = 2,
} ncFifoState_t;

typedef enum {
    NC_RO_DEVICE_THERMAL_STATS = 2000,  // Return temperatures, float *, not for general use
    NC_RO_DEVICE_OPTIMISATION_LIST = 2001,  // Return optimisations list, char *, not for general use
    NC_RO_DEVICE_THERMAL_THROTTLING_LEVEL = 2002,   // 1=TEMP_LIM_LOWER reached, 2=TEMP_LIM_HIGHER reached
    NC_RO_DEVICE_STATE = 2003,                  // Returns device state: INIT, OPEN, CLOSED,
    NC_RO_DEVICE_CURRENT_MEMORY_USED = 2004,    // Returns current device memory usage
    NC_RO_DEVICE_MEMORY_SIZE = 2005,    // Returns device memory size
    NC_RO_DEVICE_MAX_FIFO_NUM = 2006,   // return the maximum number of fifos supported
    NC_RO_DEVICE_ALLOCATED_FIFO_NUM = 2007,  // return the number of currently allocated fifos
    NC_RO_DEVICE_MAX_GRAPH_NUM = 2008,       // return the maximum number of graphs supported
    NC_RO_DEVICE_ALLOCATED_GRAPH_NUM = 2009, //  return the number of currently allocated graphs
    NC_RO_DEVICE_OPTION_CLASS_LIMIT = 2010,  //  return the highest option class supported
    NC_RO_DEVICE_FW_VER = 2011, // return device firmware version, string
    NC_RO_DEVICE_DEBUG_INFO = 2012,    // Return debug info, string
    NC_RO_DEVICE_MVTENSOR_VER = 2013,  // returns mv tensor version, string
    NC_RO_DEVICE_NAME = 2014, // returns device name as generated internally
} ncDeviceOptionsClass0;

typedef struct _devicePrivate_t devicePrivate_t;
typedef struct _graphPrivate_t graphPrivate_t;
typedef struct _fifoPrivate_t fifoPrivate_t;
typedef struct _ncTensorDescriptorPrivate_t ncTensorDescriptorPrivate_t;

struct fifoHandle_t {
    // keep place for public data here
    fifoPrivate_t* private_data;
};

struct graphHandle_t {
    // keep place for public data here
    graphPrivate_t* private_data;
};

struct deviceHandle_t {
    // keep place for public data here
    devicePrivate_t* private_data;
};

typedef enum {
    NC_FIFO_HOST_RW = 0, // fifo can be read/written through the API
                         // default for ease of use, not efficient (wasting memory)
    NC_FIFO_HOST_RO = 1, // fifo can be read through the API but can not be
                         // written ( graphs can read and write data )
    NC_FIFO_HOST_WO = 2, // fifo can be written through the API but can not be
                         // read (graphs can read but can not write)
    NC_FIFO_DEVICE_ONLY = 3, // fifo elements can be handled by graphs only
} ncFifoType_t;

typedef enum {
    NC_FIFO_FP16 = 0,
    NC_FIFO_FP32 = 1,
} ncFifoDatatype_t;

struct ncTensorDescriptor_t {
    unsigned int n;
    unsigned int c;
    unsigned int w;
    unsigned int h;
    unsigned int totalSize;
};

typedef enum {
    NC_RW_FIFO_TYPE = 0, // configure the fifo type to one type from ncFifoType_t
    NC_RW_FIFO_CONSUMER_COUNT = 1,  // The number of consumers of elements
                                    // (the number of times data must be read by
                                    // a graph or host before the element is removed.
                                    // Defaults to 1. Host can read only once always.
    NC_RW_FIFO_DATA_TYPE = 2, // 0 for fp16, 1 for fp32. If configured to fp32,
                              // the API will convert the data to the internal
                              // fp16 format automatically
    NC_RW_FIFO_DONT_BLOCK = 3,   // WriteTensor will return NC_OUT_OF_MEMORY instead
                            // of blocking, GetResult will return NO_DATA
    NC_RO_FIFO_CAPACITY = 4,     // return number of maximum elements in the buffer
    NC_RO_FIFO_READ_FILL_LEVEL = 5,   // return number of tensors in the read buffer
    NC_RO_FIFO_WRITE_FILL_LEVEL = 6,  // return number of tensors in a write buffer
    NC_RO_FIFO_TENSOR_DESCRIPTOR = 7, // return the tensor descriptor of the FIFO
    NC_RO_FIFO_STATE = 8, // return the device state
} ncFifoOption_t;


// Global
ncStatus_t ncGlobalSetOption(int option, const void *data,
                             unsigned int dataLength);
ncStatus_t ncGlobalGetOption(int option, void *data, unsigned int *dataLength);

// Device
ncStatus_t ncDeviceSetOption(struct deviceHandle_t *deviceHandle,
                             ncOptionClass_t opClass, int option,
                             const void *data, unsigned int dataLength);
ncStatus_t ncDeviceGetOption(struct deviceHandle_t *deviceHandle,
                             ncOptionClass_t opClass, int option,
                             void *data, unsigned int *dataLength);
ncStatus_t ncDeviceInit(int index, struct deviceHandle_t **deviceHandle);
ncStatus_t ncDeviceOpen(struct deviceHandle_t *deviceHandle);
ncStatus_t ncDeviceClose(struct deviceHandle_t *deviceHandle);

// Graph
ncStatus_t ncGraphInit(const char* name, struct graphHandle_t **graphHandle);
ncStatus_t ncGraphAllocate(struct deviceHandle_t *deviceHandle,
                           struct graphHandle_t *graphHandle,
                           const void *graphFile, unsigned int graphFileLength);
ncStatus_t ncGraphDeallocate(struct graphHandle_t *graphHandle);
ncStatus_t ncGraphSetOption(struct graphHandle_t *graphHandle,
                            ncOptionClass_t opClass, int option,
                            const void *data, unsigned int dataLength);
ncStatus_t ncGraphGetOption(struct graphHandle_t *graphHandle,
                            ncOptionClass_t opClass,
                            int option, void *data,
                            unsigned int *dataLength);
ncStatus_t ncGraphQueueInference(struct graphHandle_t *graphHandle,
                            struct fifoHandle_t** fifoIn,
                            struct fifoHandle_t** fifoOut);
ncStatus_t ncGraphQueueInferenceWithFifoElem(struct graphHandle_t *graphHandle,
                                        struct fifoHandle_t** fifoIn,
                                        struct fifoHandle_t** fifoOut, const void *inputTensor,
                                        struct ncTensorDescriptor_t *inputDesc, void *userParam);

// Fifo
ncStatus_t ncFifoInit(ncFifoType_t type, struct fifoHandle_t** fifo);
ncStatus_t ncFifoCreate(struct fifoHandle_t* fifo, struct deviceHandle_t* device,
                        struct ncTensorDescriptor_t* tensorDesc,
                        unsigned int numElem);
ncStatus_t ncFifoSetOption(struct fifoHandle_t* fifo, ncFifoOption_t option,
                           const void *data, unsigned int dataLength);
ncStatus_t ncFifoGetOption(struct fifoHandle_t* fifo, ncFifoOption_t option,
                           void *data, unsigned int *dataLength);


ncStatus_t ncFifoDelete(struct fifoHandle_t* fifo);
ncStatus_t ncFifoWriteElem(struct fifoHandle_t* fifo, const void *inputTensor,
                           struct ncTensorDescriptor_t *inputDesc, void *userParam);
ncStatus_t ncFifoReadElem(struct fifoHandle_t* fifo, void **outputData,
                          struct ncTensorDescriptor_t *outputDesc, void **userParam);
ncStatus_t ncFifoRemoveElem(struct fifoHandle_t* fifo);
#ifdef __cplusplus
}
#endif

#endif
