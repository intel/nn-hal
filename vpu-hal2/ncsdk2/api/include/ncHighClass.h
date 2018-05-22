///
/// @file
/// @copyright All code copyright Movidius Ltd 2012, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     MVNC host-device communication structures
///


// Includes
// ----------------------------------------------------------------------------

#ifndef _NC_HIGH_CLASS_H_
#define _NC_HIGH_CLASS_H_
#include "mvnc.h"


typedef enum {
    GRAPH_2_NOT_IMPLEMENTED = 1200,
} ncGraphOptionsClass2_t; //TODO: not implemented or not for class0

typedef enum {
    GRAPH_3_NOT_IMPLEMENTED = 1300,
} ncGraphOptionsClass3_t; //TODO: not implemented or not for class0


typedef enum {
    NC_TEMP_LIM_LOWER = 2100,       // Temperature for short sleep, float, not for general use
    NC_TEMP_LIM_HIGHER = 2101,      // Temperature for long sleep, float, not for general use
    NC_BACKOFF_TIME_NORMAL = 2102,  // Normal sleep in ms, int, not for general use
    NC_BACKOFF_TIME_HIGH = 2103,    // Short sleep in ms, int, not for general use
    NC_BACKOFF_TIME_CRITICAL = 2104,// Long sleep in ms, int, not for general use
    NC_TEMPERATURE_DEBUG = 2105,    // Stop on critical temperature, int, not for general use
    NC_RO_DEVICE_MAX_EXECUTORS_NUM = 2106,  //Maximum number of executers per graph
} ncDeviceOptionsClass1;


typedef enum {
    NC_RW_DEVICE_SHELL_ENABLE = 2200, // Activate RTEMS shell.
    NC_RW_DEVICE_LOG_LEVEL = 2201, // Set/Get log level for the NC infrastructure on the device
    NC_RW_DEVICE_MVTENSOR_LOG_LEVEL = 2202, //Set/Get log level of MV_Tensor
    NC_RW_DEVICE_XLINK_LOG_LEVEL = 2203, //Set.Get XLink log level
    DEV_2_NOT_IMPLEMENTED = 2204
} ncDeviceOptionsClass2;

typedef enum {
    DEV_3_NOT_IMPLEMENTED = 2300
} ncDeviceOptionsClass3;

ncStatus_t setDeviceOptionClass1(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass1 option,
                                 const void *data, unsigned int dataLength);
ncStatus_t setDeviceOptionClass2(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass2 option,
                                 const void *data, unsigned int dataLength);
ncStatus_t setDeviceOptionClass3(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass3 option,
                                 const void *data, unsigned int dataLength);


ncStatus_t getDeviceOptionClass1(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass1 option,
                                 void *data, unsigned int* dataLength);
ncStatus_t getDeviceOptionClass2(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass2 option,
                                 void *data, unsigned int* dataLength);
ncStatus_t getDeviceOptionClass3(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass3 option,
                                 void *data, unsigned int* dataLength);

ncStatus_t getGraphOptionClass2(struct _graphPrivate_t *g,
                                ncGraphOptionsClass2_t option,
                                void *data, unsigned int* dataLength);
ncStatus_t getGraphOptionClass3(struct _graphPrivate_t *g,
                                ncGraphOptionsClass3_t option,
                                void *data, unsigned int* dataLength);
ncStatus_t setGraphOptionClass2(struct _graphPrivate_t *g,
                                ncGraphOptionsClass2_t option,
                                const void *data, unsigned int dataLength);
ncStatus_t setGraphOptionClass3(struct _graphPrivate_t *g,
                                ncGraphOptionsClass3_t option,
                                const void *data, unsigned int dataLength);

#endif
