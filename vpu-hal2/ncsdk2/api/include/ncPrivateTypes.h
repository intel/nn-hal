///
/// @file
/// @copyright All code copyright Movidius Ltd 2012, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     MVNC host-device communication structures
///


// Includes
// ----------------------------------------------------------------------------

#ifndef _NC_PRIVATE_TYPES_H_
#define _NC_PRIVATE_TYPES_H_

#include <pthread.h>
#include <mvnc.h>

#define NC_MAX_NAME_SIZE        28

struct _devicePrivate_t {
    int backoff_time_normal, backoff_time_high, backoff_time_critical;
    int temperature_debug, throttle_happened;
    float temp_lim_upper, temp_lim_lower;
    float *thermal_stats;
    char *dev_addr;     // Device USB address as returned by usb_
    char *dev_file;     // Device filename in /dev directory
    char *optimisation_list;
    int mvtensor_ver;
    XLinkHandler_t *usb_link;
    struct _devicePrivate_t *next;  // Next device in chain
    struct _graphPrivate_t *graphs; // List of associated graphs
    struct _fifoPrivate_t *fifos; // List of associated fifos
    streamId_t device_mon_stream_id;
    streamId_t graph_monitor_stream_id;
    pthread_mutex_t dev_data_m;
    pthread_mutex_t dev_stream_m;
    pthread_mutex_t graph_streamm;
    deviceCapabilities_t dev_attr;
    ncDeviceState_t state;
} *devices;
struct _userParamPrivate_t {
    void* data;
    struct _userParamPrivate_t* next;
};
struct _graphPrivate_t {
    uint32_t id;
    int started;
    int batch_size;
    int executors_number;
    int have_data;
    int input_count;
    int output_count;
    struct ncTensorDescriptor_t input_tensor_desc;
    struct ncTensorDescriptor_t output_tensor_desc;
    unsigned nstages;
    struct _devicePrivate_t *dev;
    struct _graphPrivate_t *next;
    char *aux_buffer;
    char *debug_buffer;
    char name[NC_MAX_NAME_SIZE];
    float *time_taken;
    streamId_t graph_stream_id;
    ncGraphState_t state;
};

struct _fifoPrivate_t{
    ncFifoType_t type;
    int  consumer_cnt;
    uint32_t id;
    streamId_t streamId;
    struct ncTensorDescriptor_t tensor_desc;
    struct _devicePrivate_t *dev;
    struct _fifoPrivate_t *next;
    char name[16];
    struct _userParamPrivate_t *user_param_in; //used for write fifo
    struct _userParamPrivate_t *user_param_out; //used for read fifo
    int write_count;
    int consumed_by_graph;
    int num_elements;
    int api_read_element;
    int api_read_adjust;
    int datatype;
    int free_;
    int consumers_remaining;
    pthread_mutex_t fifo_mutex;
    ncFifoState_t state;
    void* output_data;
};
#endif
