/*
* Copyright 2017 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/

#define _GNU_SOURCE

#include <XLinkConsole.h>
#include <XLink.h>
#include <unistd.h>
#include "ncCommPrivate.h"
#include "ncPrivateTypes.h"
#include "ncHighClass.h"

#define MVLOG_UNIT_NAME ncAPIHighClass
#include "mvLog.h"

static void initShell(struct _devicePrivate_t *d, int portNum){
    initXlinkShell(d->usb_link, portNum);
}

static int sendOptData(struct _devicePrivate_t *d)
{
    int config[10];
    return NC_UNSUPPORTED_FEATURE;// TODO: this is class1 feature
    config[3] = d->temp_lim_upper;
    config[4] = d->temp_lim_lower;
    config[5] = d->backoff_time_normal;
    config[6] = d->backoff_time_high;
    config[7] = d->backoff_time_critical;
    config[8] = d->temperature_debug;

    if(XLinkWriteData(d->device_mon_stream_id, (const uint8_t*)config, sizeof(config)) != 0 )
        return NC_ERROR;

    return NC_OK;
}



ncStatus_t setDeviceOptionClass1(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass1 option,
                                 const void *data, unsigned int dataLength){
    switch (option) {
    case NC_TEMP_LIM_LOWER:
        d->temp_lim_lower = *(float *) data;
        break;
    case NC_TEMP_LIM_HIGHER:
        d->temp_lim_upper = *(float *) data;
        break;
    case NC_BACKOFF_TIME_NORMAL:
        d->backoff_time_normal = *(int *) data;
        break;
    case NC_BACKOFF_TIME_HIGH:
        d->backoff_time_high = *(int *) data;
        break;
    case NC_BACKOFF_TIME_CRITICAL:
        d->backoff_time_critical = *(int *) data;
        break;
    case NC_TEMPERATURE_DEBUG:
        d->temperature_debug = *(int *) data;
        break;
    default:
        return NC_INVALID_PARAMETERS;
    }
    return NC_OK;
}
ncStatus_t setDeviceOptionClass2(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass2 option,
                                 const void *data, unsigned int dataLength){
    deviceCommand_t config;
    config.optionClass = NC_OPTION_CLASS2;
    if (d->dev_attr.max_device_opt_class < NC_OPTION_CLASS2){
        return NC_UNAUTHORIZED;
    }
    switch(option){
    case NC_RW_DEVICE_SHELL_ENABLE:
        initShell(d, *(int*)data);
        config.type.c2 = CLASS2_START_SHELL;
        config.data = 0;
        break;
    case NC_RW_DEVICE_LOG_LEVEL:
        config.type.c2 = CLASS2_SET_LOG_LEVEL_GLOBAL;
        config.data = *(uint32_t*)data;
        break;
    case NC_RW_DEVICE_MVTENSOR_LOG_LEVEL:
        config.type.c2 = CLASS2_SET_LOG_LEVEL_FATHOM;
        config.data = *(uint32_t*)data;
        break;
    case NC_RW_DEVICE_XLINK_LOG_LEVEL:
        config.type.c2 = CLASS2_SET_LOG_LEVEL_XLINK;
        config.data = *(uint32_t*)data;
        break;
    default:
        return NC_INVALID_PARAMETERS;
        break;
    }
    //sleep(1);
    printf("0-----------------------\n");
    XLinkWriteData(d->device_mon_stream_id, &config, sizeof(config));
    return NC_OK;
}
ncStatus_t setDeviceOptionClass3(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass3 option,
                                 const void *data, unsigned int dataLength){
    return NC_UNSUPPORTED_FEATURE;
}


ncStatus_t getDeviceOptionClass1(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass1 option,
                                 void *data, unsigned int* dataLength){
    switch (option) {
    case NC_TEMP_LIM_LOWER:
        *(float *) data = d->temp_lim_lower;
        *dataLength = sizeof(int);
        break;
    case NC_TEMP_LIM_HIGHER:
        *(float *) data = d->temp_lim_upper;
        *dataLength = sizeof(int);
        break;
    case NC_BACKOFF_TIME_NORMAL:
        *(int *) data = d->backoff_time_normal;
        *dataLength = sizeof(int);
        break;
    case NC_BACKOFF_TIME_HIGH:
        *(int *) data = d->backoff_time_high;
        *dataLength = sizeof(int);
        break;
    case NC_BACKOFF_TIME_CRITICAL:
        *(int *) data = d->backoff_time_critical;
        *dataLength = sizeof(int);
        break;
    case NC_TEMPERATURE_DEBUG:
        *(int *) data = d->temperature_debug;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_MAX_EXECUTORS_NUM:
        *(int *) data = d->dev_attr.max_executors;
        *dataLength = sizeof(int);
        break;

    default:
        return NC_INVALID_PARAMETERS;
    }
    return NC_OK;
}

ncStatus_t getDeviceOptionClass2(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass2 option,
                                 void *data, unsigned int* dataLength){
    return NC_UNSUPPORTED_FEATURE;
}
ncStatus_t getDeviceOptionClass3(struct _devicePrivate_t *d,
                                 ncDeviceOptionsClass3 option,
                                 void *data, unsigned int* dataLength){
    return NC_UNSUPPORTED_FEATURE;
}
ncStatus_t getGraphOptionClass2(struct _graphPrivate_t *g,
                                ncGraphOptionsClass2_t option,
                                void *data, unsigned int* dataLength){
    return NC_UNSUPPORTED_FEATURE;
}
ncStatus_t getGraphOptionClass3(struct _graphPrivate_t *g,
                                ncGraphOptionsClass3_t option,
                                void *data, unsigned int* dataLength){
    return NC_UNSUPPORTED_FEATURE;
}
ncStatus_t setGraphOptionClass2(struct _graphPrivate_t *g,
                                ncGraphOptionsClass2_t option,
                                const void *data, unsigned int dataLength){
    return NC_UNSUPPORTED_FEATURE;
}
ncStatus_t setGraphOptionClass3(struct _graphPrivate_t *g,
                                ncGraphOptionsClass3_t option,
                                const void *data, unsigned int dataLength){
    return NC_UNSUPPORTED_FEATURE;
}
