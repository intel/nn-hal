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
#include <dlfcn.h>		// For dladdr
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <dirent.h>
#include <time.h>
#include "mvnc.h"
#include "fp16.h"

#include "XLink.h"
#include "ncCommPrivate.h"
#include "ncPrivateTypes.h"
#include "ncHighClass.h"

#include <android/log.h>
#include <cutils/log.h>

#define MVLOG_UNIT_NAME ncAPI
#include "mvLog.h"

#define THERMAL_BUFFER_SIZE 100
#define DEBUG_BUFFER_SIZE 	120

#define MAX_TENSORS_TO_LOAD (2)
#define BLOB_STREAM_SIZE 4096
#define TENSOR_STREAM_SIZE 320*1024   * MAX_TENSORS_TO_LOAD
#define OUTPUT_STREAM_SIZE 8 //read only from PC


#define MAX_OPTIMISATIONS 		40
#define OPTIMISATION_NAME_LEN 	50
#define OPTIMISATION_LIST_BUFFER_SIZE (MAX_OPTIMISATIONS * OPTIMISATION_NAME_LEN)
#define CONFIG_STREAM_SIZE OPTIMISATION_LIST_BUFFER_SIZE

#define MAX_PATH_LENGTH 		255
#define STATUS_WAIT_TIMEOUT     15

#define SLEEP_MS        250
#define MAX_ITERATIONS  20

static int initialized = 0;
static pthread_mutex_t globalMutex = PTHREAD_MUTEX_INITIALIZER;
static XLinkGlobalHandler_t ghandler;

/////////////////////////// Structs /////////////////////////////

static double timeInSeconds()
{
	static double s;
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	if (!s)
		s = ts.tv_sec + ts.tv_nsec * 1e-9;
	return ts.tv_sec + ts.tv_nsec * 1e-9 - s;
}
static char* getProductName(const char* name)
{
    char* p = strchr(name, '-');
    if (p == NULL)
        return "";
    return p;
}

static void resetAll()
{
    int index = 0;
    int stalled_count = 0;
    int iters = 0;
    int bootrom_count = 0;
    int after_reset_count = 0;
    char name[NC_MAX_NAME_SIZE] = "";
    XLinkError_t rc;


    while (1) {
        rc = XLinkGetDeviceName(index, name, NC_MAX_NAME_SIZE);
        if (rc != X_LINK_SUCCESS)
            break; //no more devices found

        if (strlen(getProductName(name)) == 1) { //name doesn't have product number
            //device is already booted, need to reset
            mvLog(MVLOG_DEBUG,"Found stalled device %s\n", name);
            XLinkHandler_t* handler = calloc(1, sizeof(XLinkHandler_t));

            if (!handler){
                mvLog(MVLOG_ERROR, "Memory allocation failed");
                break;
            }
            handler->devicePath = (char*)name;
            if (XLinkConnect(handler) != X_LINK_SUCCESS) {
                mvLog(MVLOG_ERROR," Failed to connect to stalled device\n");
            }
            stalled_count++;
            free(handler);

        } else {
            bootrom_count++;
        }
        index++;
    }

    if (stalled_count) {
        mvLog(MVLOG_INFO,"Stalled devices found, Reseting...");
        XLinkResetAll();

        iters = 0;

        while ((after_reset_count < bootrom_count + stalled_count) &&
                iters < MAX_ITERATIONS) {
            usleep(SLEEP_MS*1000);
            after_reset_count = 0;
            index = 0;
            while (1) {
                XLinkError_t rc = XLinkGetDeviceName(index, name, NC_MAX_NAME_SIZE);
                if (rc != X_LINK_SUCCESS)
                break; //no more devices found

                if (strlen(getProductName(name)) > 1) { //name has product number
                    after_reset_count++;
                }
                index++;
            }
            iters++;
            mvLog(MVLOG_INFO,"...");
            //mvLog(MVLOG_DEBUG,"total number of none-stalled devices %d", after_reset_count);
        }
        usleep(SLEEP_MS*1000);
    }
}


static void initialize()
{
    mvLogDefaultLevelSet(MVLOG_FATAL);
	// We sanitize the situation by trying to reset the devices that have been left open
	initialized = 1;
	int sc = XLinkInitialize(&ghandler); //need to be called once
	if (sc != X_LINK_SUCCESS) {
	    mvLog(MVLOG_ERROR," Initialization failed\n");
	}
#ifndef XLINK_NO_BOOT

    resetAll();
#endif
}

ncStatus_t ncDeviceInit(int index, struct deviceHandle_t **deviceHandle)
{
    mvLog(MVLOG_DEBUG, "ncDeviceInit index %d\n", index);
	if (index < 0 || !deviceHandle)
		return NC_INVALID_PARAMETERS;

	char name[NC_MAX_NAME_SIZE] = "";

	pthread_mutex_lock(&globalMutex);
	if (!initialized)
		initialize();

	XLinkError_t rc = XLinkGetDeviceName(index, name, NC_MAX_NAME_SIZE);
	pthread_mutex_unlock(&globalMutex);

	if (rc == X_LINK_SUCCESS) {
		struct deviceHandle_t *dH = calloc(1, sizeof(*dH));
		struct _devicePrivate_t *d = calloc(1, sizeof(*d));
		dH->private_data = d;
		d->state = NC_DEVICE_INITIALIZED;
		d->dev_addr = strdup(name);
		d->device_mon_stream_id = INVALID_LINK_ID;
		d->graph_monitor_stream_id = INVALID_LINK_ID;
		*deviceHandle = dH;
	}
    switch(rc) {
        case X_LINK_SUCCESS:
            return NC_OK;
        case X_LINK_DEVICE_NOT_FOUND:
            return NC_DEVICE_NOT_FOUND;
        case X_LINK_TIMEOUT:
            return NC_TIMEOUT;
        default:
            return NC_ERROR;
    }
}

static ncStatus_t getDevAttributes(struct _devicePrivate_t *d) {
    pthread_mutex_lock(&d->dev_stream_m);
    deviceCommand_t config;
    config.type.c0 = CLASS0_DEVICE_CAPABILITIES;
    config.optionClass = NC_OPTION_CLASS0;
    if(XLinkWriteData(d->device_mon_stream_id, (const uint8_t*)&config, sizeof(config)) != 0 ) {
        pthread_mutex_unlock(&d->dev_stream_m);
        return NC_ERROR;
    }
    streamPacketDesc_t* packet;
    if(XLinkReadData(d->device_mon_stream_id, &packet) != 0) {
        pthread_mutex_unlock(&d->dev_stream_m);
        return NC_ERROR;
    }
    pthread_mutex_unlock(&d->dev_stream_m);
    if(packet->length != sizeof(d->dev_attr)) {
        mvLog(MVLOG_WARN, "Broken protocol. DevData can't be read\n");
        XLinkReleaseData(d->device_mon_stream_id);
        return NC_ERROR;
    }
    d->dev_attr = *(deviceCapabilities_t*)packet->data;
    XLinkReleaseData(d->device_mon_stream_id);
    mvLog(MVLOG_INFO, "Device attributes\n");
    mvLog(MVLOG_INFO, "Device FW version: %x.%x.%x.%x\n", d->dev_attr.fw_version[0],
          d->dev_attr.fw_version[1], d->dev_attr.fw_version[2], d->dev_attr.fw_version[3]);
    mvLog(MVLOG_INFO, "Maximum graphs: %d\n", d->dev_attr.max_graphs);
    mvLog(MVLOG_INFO, "Maximum fifos: %d\n", d->dev_attr.max_fifos);
    mvLog(MVLOG_INFO, "Maximum graph option class: %d\n", d->dev_attr.max_graph_opt_class);
    mvLog(MVLOG_INFO, "Maximum device option class: %d\n", d->dev_attr.max_device_opt_class);
    mvLog(MVLOG_INFO, "Device memory capacity: %d\n", d->dev_attr.max_memory);
    return NC_OK;
}

static ncStatus_t getOptimisationList(struct _devicePrivate_t *d)
{
    int  i;
    char *p;

    if (d->optimisation_list)
        return NC_OK;

    d->optimisation_list = calloc(OPTIMISATION_LIST_BUFFER_SIZE, 1);
    if (!d->optimisation_list)
        return NC_OUT_OF_MEMORY;
    deviceCommand_t config;
    config.type.c0 = CLASS0_OPT_LIST;
    config.optionClass = NC_OPTION_CLASS0;
    pthread_mutex_lock(&d->dev_stream_m);
    if(XLinkWriteData(d->device_mon_stream_id, (const uint8_t*)&config, sizeof(config)) != 0 ) {
        pthread_mutex_unlock(&d->dev_stream_m);
        return NC_ERROR;
    }
    streamPacketDesc_t* packet;
    if(XLinkReadData(d->device_mon_stream_id, &packet) != 0) {
        pthread_mutex_unlock(&d->dev_stream_m);
        return NC_ERROR;
    }
    pthread_mutex_unlock(&d->dev_stream_m);
    memcpy(d->optimisation_list, packet->data, packet->length);
    XLinkReleaseData(d->device_mon_stream_id);

    for (i = 0; i < MAX_OPTIMISATIONS; i++) {
        p = strchr(d->optimisation_list + i * OPTIMISATION_NAME_LEN, '~');
        if (p)
            *p = 0;
    }
    return NC_OK;
}

static ncStatus_t getThermalStats(struct _devicePrivate_t *d){
    if (!d->thermal_stats){
        d->thermal_stats = calloc(THERMAL_BUFFER_SIZE, 1);
        if (!d->thermal_stats)
            return NC_OUT_OF_MEMORY;
    }
    deviceCommand_t config;
    config.type.c0 = CLASS0_THERMAL_STATS;
    config.optionClass = NC_OPTION_CLASS0;
    pthread_mutex_lock(&d->dev_stream_m);
    if(XLinkWriteData(d->device_mon_stream_id, (const uint8_t*)&config, sizeof(config)) != 0) {
        pthread_mutex_unlock(&d->dev_stream_m);
        return NC_ERROR;
    }
    streamPacketDesc_t* packet;

    if(XLinkReadData(d->device_mon_stream_id, &packet) != 0) {
        pthread_mutex_unlock(&d->dev_stream_m);
        return NC_ERROR;
    }
    pthread_mutex_unlock(&d->dev_stream_m);
    if( packet->length != (THERMAL_BUFFER_SIZE + sizeof(float))) {
        return NC_ERROR;
    }
    memcpy(d->thermal_stats, packet->data, packet->length);
    XLinkReleaseData(d->device_mon_stream_id);
    return NC_OK;
}
static ncStatus_t deviceGetDeviceMemory(struct _devicePrivate_t *d, uint32_t *mem) {
    deviceCommand_t config;
    config.type.c0 = CLASS0_DEVICE_USED_MEMORY;
    config.optionClass = NC_OPTION_CLASS0;
    pthread_mutex_lock(&d->dev_stream_m);
    if(XLinkWriteData(d->device_mon_stream_id, (const uint8_t*)&config, sizeof(config)) != 0) {
        pthread_mutex_unlock(&d->dev_stream_m);
        return NC_ERROR;
    }
    streamPacketDesc_t* packet;

    if(XLinkReadData(d->device_mon_stream_id, &packet) != 0) {
        pthread_mutex_unlock(&d->dev_stream_m);
        return NC_ERROR;
    }
    pthread_mutex_unlock(&d->dev_stream_m);
    if( packet->length != (sizeof(uint32_t))) {
        return NC_ERROR;
    }
    memcpy(mem, packet->data, packet->length);
    XLinkReleaseData(d->device_mon_stream_id);
    return NC_OK;
}
static int isDeviceOpened(const char *name)
{
	struct _devicePrivate_t *d = devices;
	while (d) {
		if (strcmp(d->dev_addr, name) == 0)
			return 0;
		d = d->next;
	}
	return -1;
}

ncStatus_t ncDeviceOpen(struct deviceHandle_t *deviceHandle) {
	char mv_cmd_file_path[MAX_PATH_LENGTH], *p;
	//char mv_cmd_file_name[40] = "mvnc/MvNCAPI-maXXXX.mvcmd";
	char mv_cmd_file_name[40] = "/system/vendor/etc/MvNCAPI-ma2450.mvcmd";


	char name2[NC_MAX_NAME_SIZE] = "";

	if (!deviceHandle || !deviceHandle->private_data){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
		return NC_INVALID_PARAMETERS;
	}
	struct _devicePrivate_t *d = deviceHandle->private_data;

	pthread_mutex_lock(&globalMutex);
	if (!initialized)
		initialize();

	// Search the mvnc executable in the same directory of this library, under mvnc
	// in the future there will ideally be one FW file for all, for now they are seperate

/*	sprintf(mv_cmd_file_name, "mvnc/MvNCAPI%s.mvcmd", getProductName(d->dev_addr));


	Dl_info info;
	dladdr(ncDeviceOpen, &info);
	strncpy(mv_cmd_file_path, info.dli_fname, sizeof(mv_cmd_file_path) - 40);
	p = strrchr(mv_cmd_file_path, '/');
	if (p)
		strcpy(p + 1, mv_cmd_file_name);
	else */
		strcpy(mv_cmd_file_path, mv_cmd_file_name);

	mvLog(MVLOG_DEBUG, "File path %s\n", mv_cmd_file_path);

	int rc = XLinkBootRemote(d->dev_addr, mv_cmd_file_path);
	if (rc)
		mvLog(MVLOG_WARN, "%s() XLinkBootRemote returned error %d\n", __func__, rc);
	else
		mvLog(MVLOG_INFO, "%s() XLinkBootRemote returned success %d\n", __func__, rc);

	double waittm = timeInSeconds() + STATUS_WAIT_TIMEOUT;
	while (timeInSeconds() < waittm && rc == 0) {
		XLinkHandler_t* handler = calloc(1, sizeof(XLinkHandler_t));

		handler->devicePath = (char*)d->dev_addr;
		rc = XLinkConnect(handler);
		if (rc != X_LINK_SUCCESS) {
			//we might fail in case name changed after boot
			int count = 0;
			while (1) {
				name2[0] = '\0';
				rc = XLinkGetDeviceName(count, name2, NC_MAX_NAME_SIZE);
				if (rc != X_LINK_SUCCESS)
					break;
				handler->devicePath = (char*) name2;
				rc = XLinkConnect(handler);
				if (isDeviceOpened(name2) < 0 && rc == X_LINK_SUCCESS) {
					break;
				}
				count++;
			}
		}

		if (rc != X_LINK_SUCCESS) {
			mvLog(MVLOG_WARN, "failed to find device\n");
			return NC_ERROR;
		}
		mvLog(MVLOG_INFO, "XLinkConnect done - link Id %d\n", handler->linkId);

		if (strlen(name2) > 0) {
			free(d->dev_addr);
			d->dev_addr = strdup(name2);
		}

		d->usb_link = handler;
		d->next = devices;
		d->temp_lim_upper = 95;
		d->temp_lim_lower = 85;
		d->backoff_time_normal = 0;
		d->backoff_time_high = 100;
		d->backoff_time_critical = 10000;
		d->temperature_debug = 0;
		pthread_mutex_init(&d->dev_data_m, 0);
        pthread_mutex_init(&d->dev_stream_m, 0);
        pthread_mutex_init(&d->graph_streamm, 0);

		devices = d;

		mvLog(MVLOG_DEBUG, "done\n");
		mvLog(MVLOG_INFO, "Booted %s -> %s\n",
		           d->dev_addr,
		           d->dev_file ? d->dev_file : "VSC");
		pthread_mutex_unlock(&globalMutex);
		sleep(1); //TODO: fix the create stream
		streamId_t streamId = XLinkOpenStream(d->usb_link->linkId, "deviceMonitor", CONFIG_STREAM_SIZE);
		if (streamId == INVALID_STREAM_ID) {
			mvLog(MVLOG_WARN, "can't open stream\n");
			return NC_ERROR;
		}
		d->device_mon_stream_id = streamId;
		getDevAttributes(d);


		streamId = XLinkOpenStream(d->usb_link->linkId, "graphMonitor", BLOB_STREAM_SIZE);
		if (streamId == INVALID_STREAM_ID) {
		    mvLog(MVLOG_WARN, "can't open stream\n");
			return NC_ERROR;
		}
		d->graph_monitor_stream_id = streamId;
		d->state = NC_DEVICE_OPENED;
		return NC_OK;
	}

	pthread_mutex_unlock(&globalMutex);
	return NC_ERROR;
}

static int findDevice(struct _devicePrivate_t *deviceHandle)
{

	struct _devicePrivate_t *d = devices;

	while (d) {
		if (d == deviceHandle)
			return 0;
		d = d->next;
	}

	return -1;
}
static int deviceGetNumberOfGraphs(struct _devicePrivate_t *deviceHandle)
{
	if (deviceHandle == NULL)
		return 0;
	int num = 0;
	struct _graphPrivate_t *g = deviceHandle->graphs;
	while (g) {
		num++;
		g = g->next;
	}
	return num;
}

static int deviceGetNumberOfFifos(struct _devicePrivate_t *deviceHandle)
{
    if (deviceHandle == NULL)
        return 0;
    int num = 0;
    struct _fifoPrivate_t *f = deviceHandle->fifos;
    while (f) {
        num++;
        f = f->next;
    }
    return num;
}

static int findGraph(struct _graphPrivate_t *graphHandle)
{
	struct _devicePrivate_t *d = devices;

	while (d) {
		struct _graphPrivate_t *g = d->graphs;
		while (g) {
			if (g == graphHandle)
				return 0;
			g = g->next;
		}
		d = d->next;
	}

	return -1;
}

// Defined here as it will be used twice
static int deallocateGraph(struct _graphPrivate_t *g)
{
	int found = 0;

	// Remove it from the list of the associated device
	if (g->dev->graphs == g) {
		g->dev->graphs = g->next;
		found = 1;
	} else {
		struct _graphPrivate_t *gp = g->dev->graphs;
		while (gp->next) {
			if (gp->next == g) {
				found = 1;
				gp->next = gp->next->next;
				break;
			}
			gp = gp->next;
		}
	}

	// Free it with all its data
	if (found) {
		free(g->aux_buffer);
		g->dev->thermal_stats = 0;
		free(g);
	}

	return -!found;
}

static int deallocateFifo(struct _fifoPrivate_t *f)
{
	int found = 0;
	// Remove it from the list of the associated device
	if (f->dev->fifos == f) {
		f->dev->fifos = f->next;
		found = 1;
	} else {
		struct _fifoPrivate_t *fp = f->dev->fifos;
		while (fp->next) {
			if (fp->next == f) {
				found = 1;
				fp->next = fp->next->next;
				break;
			}
			fp = fp->next;
		}
	}

	// Free it with all its data
	if (found) {
		// even if we are deallocating the item, somebody could try to use it
		// by mistake, good to modify this data.
		f->state = NC_FIFO_DESTROYED;

		//deallocate on device
		XLinkCloseStream(f->streamId);
		free(f->output_data);
		struct _userParamPrivate_t* temp;
		while (f->user_param_in) {
			temp = f->user_param_in;
			f->user_param_in =  f->user_param_in->next;
			free(temp);
		}
		while (f->user_param_out) {
			temp = f->user_param_out;
			f->user_param_out =  f->user_param_out->next;
			free(temp);
		}
	}

	return -!found;
}
ncStatus_t ncDeviceClose(struct deviceHandle_t *deviceHandle){
	int found = 0;

	if (!deviceHandle){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
		return NC_INVALID_PARAMETERS;
	}

	pthread_mutex_lock(&globalMutex);
	if (findDevice(deviceHandle->private_data)) {
		pthread_mutex_unlock(&globalMutex);
		return NC_INVALID_PARAMETERS;
	}
	mvLog(MVLOG_INFO, "closing device\n");

	struct _devicePrivate_t *d = deviceHandle->private_data;
	// Remove it from our list
	if (devices == d) {
		devices = d->next;
		found = 1;
	} else {
		struct _devicePrivate_t *dp = devices;
		while (dp->next) {
			if (dp->next == d) {
				found = 1;
				dp->next = dp->next->next;
				break;
			}
			dp = dp->next;
		}
	}

	if (!found) {
		pthread_mutex_unlock(&globalMutex);
		return NC_INVALID_PARAMETERS;
	}
	// Deallocate all associated graphs
	pthread_mutex_lock(&d->dev_data_m);
	while (d->graphs){
		deallocateGraph(d->graphs);
	}
	// Deallocate all associated fifos
	while (d->fifos) {
		deallocateFifo(d->fifos);
	}
	if (d->device_mon_stream_id != INVALID_LINK_ID)
		XLinkCloseStream(d->device_mon_stream_id);
	if (d->graph_monitor_stream_id != INVALID_LINK_ID)
		XLinkCloseStream(d->graph_monitor_stream_id);

	// Reset
	XLinkResetRemote(d->usb_link->linkId);
//	usblink_resetmyriad(d->usb_link);
//	usblink_close(d->usb_link);
	if (d->optimisation_list)
		free(d->optimisation_list);
	d->state = NC_DEVICE_CLOSED;
	free(d->dev_addr);
	free(d->dev_file);
	pthread_mutex_unlock(&d->dev_data_m);
	pthread_mutex_destroy(&d->dev_data_m);
	free(d);
	free(deviceHandle);
	pthread_mutex_unlock(&globalMutex);

	usleep(500000);//TODO: verify if this is needed. Add comment if so.
	return NC_OK;
}

ncStatus_t ncGraphInit(const char* name, struct graphHandle_t **graphHandle) {

    struct graphHandle_t *gH = calloc(1, sizeof(*gH));
    struct _graphPrivate_t *g = calloc(1, sizeof(*g));
    gH->private_data = g;
    strncpy(g->name, name, NC_MAX_NAME_SIZE);
    g->batch_size = 1;
    g->executors_number = 1;
    g->dev = NULL;
    *graphHandle = gH;
    return NC_OK;
}

ncStatus_t sendGraphMonitorRequest(streamId_t graphMonStream, graphMonCommand_t *cmd) {
    if(XLinkWriteData(graphMonStream, (uint8_t*)cmd, sizeof(*cmd)) != 0)
    {
        return NC_ERROR;
    }
    return NC_OK;
}
ncStatus_t checkGraphMonitorResponse(streamId_t graphMonStream) {
    streamPacketDesc_t *ack;
    if(XLinkReadData(graphMonStream, &ack) != 0 ) {
        mvLog(MVLOG_ERROR, "XLink error");
        return NC_ERROR;
    }
    int value = *((int*)ack->data);
    if(XLinkReleaseData(graphMonStream) != 0 ) {
        mvLog(MVLOG_ERROR, "XLink error");
        return NC_ERROR;
    }
    if (value != 0){
        mvLog(MVLOG_WARN, "Graph monitor request returned error");

        return NC_MYRIAD_ERROR;
    }

    return NC_OK;
}

ncStatus_t ncGraphAllocate(struct deviceHandle_t *deviceHandle,
                           struct graphHandle_t *graphHandle,
                           const void *graphFile, unsigned int graphFileLength) {
    ncStatus_t rc = NC_OK;

		mvLog(MVLOG_INFO, "Starting Graph allocation sequence\n");
		ALOGE("Starting Graph allocation sequence");




    if (!graphHandle || !graphFile || !deviceHandle){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }

    //TODO: fix graph version
//    if (graph[VERSION_OFFSET] != GRAPH_VERSION)
//        return NC_UNSUPPORTED_GRAPH_FILE;


    static int graphIdCount = 0;
	struct _graphPrivate_t *g = graphHandle->private_data;

	struct _devicePrivate_t *d = devices;
	if (graphFileLength > d->dev_attr.max_memory){
	    mvLog(MVLOG_ERROR, "The graph file is bigger than the device memory");
	    return NC_UNSUPPORTED_GRAPH_FILE;
	}


	pthread_mutex_lock(&globalMutex);
	while (d) {
		if (d == deviceHandle->private_data)
			break;
		d = d->next;
	}
	//TODO: review lists of devices and graphs internally.
	//TODO: check if the graph is not already on the device
	if (!d) {
        pthread_mutex_unlock(&globalMutex);
        mvLog(MVLOG_ERROR, "Device not found!");
		return NC_INVALID_PARAMETERS;
	}
    pthread_mutex_unlock(&globalMutex);

//	if (d->graphs) {
//		pthread_mutex_unlock(&mm);
//		return NC_BUSY;
//	}
    g->id = graphIdCount++;
    streamId_t streamId;
    if (g->executors_number > d->dev_attr.max_executors)
    {
        mvLog(MVLOG_ERROR, "nce number is greater than max allowed!");
        return NC_INVALID_PARAMETERS;
    }

    graphMonCommand_t cmd;
    cmd.cmdClass = GRAPH_MON_CLASS_GRAPH_CMD;
    cmd.cmd.graphCmd.type  = GRAPH_ALLOCATE_CMD;
    snprintf(cmd.cmd.graphCmd.streamName, 16, "graphFile%d", g->id);
    streamId = XLinkOpenStream(d->usb_link->linkId, cmd.cmd.graphCmd.streamName, graphFileLength);
    if (streamId == INVALID_STREAM_ID){
        mvLog(MVLOG_WARN, "can't open stream for graphFile transmission");
        return NC_ERROR;
    }
    cmd.cmd.graphCmd.id  = g->id;
    cmd.cmd.graphCmd.executors_number = g->executors_number;

    pthread_mutex_lock(&d->graph_streamm);

    if(sendGraphMonitorRequest(d->graph_monitor_stream_id, &cmd)){
        mvLog(MVLOG_WARN, "can't send graph allocation command");
        pthread_mutex_lock(&d->graph_streamm);
        return NC_ERROR;
    }
    if(XLinkWriteData(streamId, graphFile, graphFileLength) != 0 ){
        mvLog(MVLOG_WARN, "can't send graph data to device");
        pthread_mutex_unlock(&d->graph_streamm);
        return NC_ERROR;
    }
    mvLog(MVLOG_INFO, "Sent graph");
		printf("Sent graph\n");
    streamPacketDesc_t * tensorDescIn;
    streamPacketDesc_t * tensorDescOut;
    streamPacketDesc_t * nstages;


    XLinkReadData(streamId, &tensorDescIn);
    XLinkReadData(streamId, &tensorDescOut);
    XLinkReadData(streamId, &nstages);
		mvLog(MVLOG_INFO, "XLinkReadData done");
    //for now, supoprt only count 1
    if(!tensorDescIn ||
        tensorDescIn->length % sizeof(struct tensorDescriptor_t) ||
        tensorDescIn->length / sizeof(struct tensorDescriptor_t) > 1) {
        mvLog(MVLOG_ERROR, "Input tensor descriptors of the graph are invalid\n");
        mvLog(MVLOG_ERROR, "Received data from graph %d\n", *(int*)tensorDescIn->data);
        rc = NC_MYRIAD_ERROR;
    }
    //for now, supoprt only count 1
    if(!tensorDescOut ||
        tensorDescOut->length % sizeof(struct tensorDescriptor_t) ||
        tensorDescOut->length / sizeof(struct tensorDescriptor_t) > 1) {
        mvLog(MVLOG_ERROR, "Output tensor descriptors of the graph are invalid\n");
        rc = NC_MYRIAD_ERROR;
    }
    if (rc == NC_OK){
			  mvLog(MVLOG_INFO, "set input/output/stages count");
        g->input_count = tensorDescIn->length / sizeof(struct tensorDescriptor_t);
        memcpy(&g->input_tensor_desc, tensorDescIn->data,
               sizeof(struct tensorDescriptor_t));
        g->output_count = tensorDescOut->length / sizeof(struct tensorDescriptor_t);
        memcpy(&g->output_tensor_desc, tensorDescOut->data,
               sizeof(struct tensorDescriptor_t));
        g->nstages = *(uint32_t*)nstages->data;
				mvLog(MVLOG_INFO, "set input/output/stages count done");
    }

    XLinkReleaseData(streamId);
    XLinkReleaseData(streamId);
    XLinkReleaseData(streamId);
		mvLog(MVLOG_INFO, "XLinkReleaseData done");
    g->graph_stream_id = streamId;
    if(checkGraphMonitorResponse(d->graph_monitor_stream_id)) {
        mvLog(MVLOG_WARN, "The device didn't accept the graph\n");
        uint32_t memory_used;
        uint32_t length;
        ncDeviceGetOption(deviceHandle, NC_OPTION_CLASS0, NC_RO_DEVICE_CURRENT_MEMORY_USED, &memory_used, &length);
        uint32_t remaining_memory = d->dev_attr.max_memory - memory_used;
        mvLog(MVLOG_INFO, "Remaining device memory %d\n", remaining_memory);

        if(remaining_memory < 2 * graphFileLength){
            mvLog(MVLOG_WARN, "Remaining device memory (%d) is not enough for graph file (%d)\n", remaining_memory, graphFileLength);
        }
        pthread_mutex_unlock(&d->graph_streamm);
        return NC_ERROR;
    }
    if (rc){
        return rc;
    }
    //TODO: this will go away once graph options are handled properly

    // aux_buffer
    g->aux_buffer = calloc(1, 224 + g->nstages * sizeof(*g->time_taken));
    if (!g->aux_buffer) {
        return NC_OUT_OF_MEMORY;
    }
    // output_data

    //TODO: this will go away once Buffer is implemented
    g->debug_buffer = g->aux_buffer;
    g->time_taken = (float *) (g->aux_buffer + 120);
    pthread_mutex_unlock(&d->graph_streamm);

    pthread_mutex_lock(&globalMutex);
    g->dev = d; //TODO: support allocating graph to multiple devices!

    if (d->graphs)
        g->next = d->graphs;
    d->graphs = g;
    d->state = NC_GRAPH_READY;
    pthread_mutex_unlock(&globalMutex);
    mvLog(MVLOG_INFO, "Graph allocation completed successfully\n");

	return NC_OK;
}

ncStatus_t ncGraphDeallocate(struct graphHandle_t *graphHandle) {
	if (!graphHandle){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
		return NC_INVALID_PARAMETERS;
	}
    struct _graphPrivate_t *g = graphHandle->private_data;
	pthread_mutex_lock(&globalMutex);
	if (findGraph(g)) {
		pthread_mutex_unlock(&globalMutex);
        mvLog(MVLOG_ERROR, "This graph is corrupt");
		return NC_INVALID_PARAMETERS;
	}

    pthread_mutex_unlock(&globalMutex);
	struct _devicePrivate_t *d = (graphHandle->private_data)->dev;

    graphMonCommand_t cmd;
    cmd.cmdClass = GRAPH_MON_CLASS_GRAPH_CMD;
    cmd.cmd.graphCmd.type  = GRAPH_DEALLOCATE_CMD;
    cmd.cmd.graphCmd.id  = g->id;
    pthread_mutex_lock(&d->graph_streamm);
    if (sendGraphMonitorRequest(d->graph_monitor_stream_id, &cmd)) {
        return NC_ERROR;
    }
    if (checkGraphMonitorResponse(d->graph_monitor_stream_id)) {
        return NC_ERROR;
    }
    XLinkCloseStream(g->graph_stream_id);
    pthread_mutex_unlock(&d->graph_streamm);
    graphHandle->private_data->state = NC_GRAPH_DEALLOCATED;
    pthread_mutex_lock(&d->dev_data_m);
    if (deallocateGraph(graphHandle->private_data)) {
        pthread_mutex_unlock(&d->dev_data_m);
        return NC_INVALID_PARAMETERS;
    }
    pthread_mutex_unlock(&d->dev_data_m);
    return NC_OK;
}

static ncStatus_t setGraphOptionClass1(struct _graphPrivate_t *g,
                                        ncGraphOptionsClass1_t option,
                                        const void *data, unsigned int dataLength) {
    switch (option) {
        case NC_RW_GRAPH_EXECUTORS_NUM:
            if (g->state != NC_GRAPH_INITIALIZED)
            {
                mvLog(MVLOG_ERROR, "Can't set NCE number after graph allocation");
                return NC_UNAUTHORIZED;
            }
            g->executors_number = *(int *) data;;
            break;
        case NC_RW_GRAPH_BATCH_SIZE :
        case NC_RW_GRAPH_RUNTIME_CONFIG:
            return NC_UNSUPPORTED_FEATURE;
        default:
            mvLog(MVLOG_ERROR, "There is no such option in class 1");
            return NC_INVALID_PARAMETERS;
        }
    return NC_OK;
}

ncStatus_t ncGraphSetOption(struct graphHandle_t *graphHandle,
                            ncOptionClass_t opClass, int option,
                            const void *data, unsigned int dataLength) {

	mvLog(MVLOG_INFO, "ncs2 ncGraphSetOption\n");

	if (!graphHandle || !data || dataLength != 4){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
	}

	struct _graphPrivate_t *g = graphHandle->private_data;
	if(findGraph(g)){
		mvLog(MVLOG_ERROR, "aks This graph is corrupt");
	}
	pthread_mutex_lock(&globalMutex);
	if (option != NC_RW_GRAPH_EXECUTORS_NUM && findGraph(g)) {
		pthread_mutex_unlock(&globalMutex);
        mvLog(MVLOG_ERROR, "This graph is corrupt");
		return NC_INVALID_PARAMETERS;
	}
	pthread_mutex_unlock(&globalMutex);
    //we check what we can at this point, later we might fail if
    //user set a class that was not permitted
    if (g->dev != NULL && opClass > g->dev->dev_attr.max_graph_opt_class){
        mvLog(MVLOG_ERROR, "This device FW does not support NC_OPTION_CLASS%d", opClass);
        return NC_UNAUTHORIZED;
    }
    ncStatus_t rc;
    switch (opClass) {
    case NC_OPTION_CLASS0:
        mvLog(MVLOG_ERROR, "Class 0 options are read-only");
        rc = NC_UNAUTHORIZED; // option class 0 consists of read-only value
        break;
    case NC_OPTION_CLASS1:
				mvLog(MVLOG_INFO, "ncGraphSetOption NC_OPTION_CLASS1\n");
        rc = setGraphOptionClass1(g, option, data, dataLength);
        break;
    case NC_OPTION_CLASS2:
        rc = setGraphOptionClass2(g, option, data, dataLength);
        break;
    case NC_OPTION_CLASS3:
        rc = setGraphOptionClass3(g, option, data, dataLength);
        break;
    default:
        mvLog(MVLOG_ERROR, "There is no such option class");
        rc = NC_INVALID_PARAMETERS;
        break;
    }
	return rc;
}

static ncStatus_t getGraphOptionClass0(struct _graphPrivate_t *g,
                                        ncGraphOptionsClass0_t option,
                                        void *data, unsigned int* dataLength) {
    graphMonCommand_t cmd;
    streamPacketDesc_t* pack;
    cmd.cmdClass = GRAPH_MON_CLASS_GET_CLASS0;

    switch (option) {
    case NC_RO_GRAPH_STATE:
        *(int *) data = g->state;
        *dataLength = sizeof(int);
        //TODO: some graph states must be read from myriad (maybe we can keep track here)
        break;
    case NC_RO_GRAPH_INPUT_COUNT:
        *(int *) data = g->input_count;
        *dataLength = sizeof(int);
        break;
    case NC_RO_GRAPH_OUTPUT_COUNT:
        *(int *) data = g->output_count;
        *dataLength = sizeof(int);
        break;
    case NC_RO_GRAPH_TIME_TAKEN:
        cmd.cmd.optionCmd.id  = g->id;
        cmd.cmd.optionCmd.type.c0  = CLASS0_TIMING_DATA;
        pthread_mutex_lock(&g->dev->graph_streamm); //TODO: this mutex is shared with device option stream, could be separated
        //TODO: create buffer to other function
        if (sendGraphMonitorRequest(g->dev->graph_monitor_stream_id, &cmd)){
            pthread_mutex_unlock(&g->dev->graph_streamm);
            return NC_ERROR;
        }
        if(XLinkReadData(g->dev->graph_monitor_stream_id, &pack))
        {
            pthread_mutex_unlock(&g->dev->graph_streamm);
            return NC_ERROR;
        }
        if(pack->length != sizeof(*g->time_taken) * g->nstages){
            pthread_mutex_unlock(&g->dev->graph_streamm);

            XLinkReleaseData(g->dev->graph_monitor_stream_id);
            return NC_ERROR;
        }
        //Need to copy data before we check the response, since checkGraphMonitorResponse
        //calls releaseData
        memcpy(g->time_taken, pack->data, pack->length);
        XLinkReleaseData(g->dev->graph_monitor_stream_id);

        if (checkGraphMonitorResponse(g->dev->graph_monitor_stream_id)){
            pthread_mutex_unlock(&g->dev->graph_streamm);
            return NC_ERROR;
        }

        pthread_mutex_unlock(&g->dev->graph_streamm);
        *(float **) data = g->time_taken;
        *dataLength = sizeof(*g->time_taken) * g->nstages;

        break;
    case NC_RO_GRAPH_DEBUG_INFO: //TODO: copy paste here.. would be good to rething the whole get/set
        cmd.cmd.optionCmd.type.c0  = CLASS0_DEBUG_DATA;
        cmd.cmd.optionCmd.id  = g->id;
        //TODO: create buffer to other function
        pthread_mutex_lock(&g->dev->graph_streamm);
        if(XLinkWriteData(g->dev->graph_monitor_stream_id, (const uint8_t*)&cmd, sizeof(cmd)) != 0 )
        {
            pthread_mutex_unlock(&g->dev->graph_streamm);
            return NC_ERROR;
        }
        if(XLinkReadData(g->dev->graph_monitor_stream_id, &pack))
        {
            pthread_mutex_unlock(&g->dev->graph_streamm);
            return NC_ERROR;
        }
        if(pack->length != DEBUG_BUFFER_SIZE){
            pthread_mutex_unlock(&g->dev->graph_streamm);
            XLinkReleaseData(g->dev->graph_monitor_stream_id);
            return NC_ERROR;
        }
        memcpy(g->debug_buffer, pack->data, pack->length);
        XLinkReleaseData(g->dev->graph_monitor_stream_id);
        if (checkGraphMonitorResponse(g->dev->graph_monitor_stream_id)){
            pthread_mutex_unlock(&g->dev->graph_streamm);
            return NC_ERROR;
        }
        pthread_mutex_unlock(&g->dev->graph_streamm);

        *(char **) data = g->debug_buffer;
        *dataLength = DEBUG_BUFFER_SIZE;
        break;
    case NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS:
        *(struct ncTensorDescriptor_t**) data = &g->input_tensor_desc; //TODO: should we malloc here instead
        *dataLength = sizeof(&g->input_tensor_desc);
        break;
    case NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS:
        *(struct ncTensorDescriptor_t**) data = &g->output_tensor_desc; //TODO: should we malloc here instead
        *dataLength = sizeof(&g->output_tensor_desc);
        break;
    case NC_RO_GRAPH_NAME:
        *(char**) data = g->name;
        *dataLength = strlen(g->name) + 1;
        break;
    case NC_RO_GRAPH_OPTION_CLASS_LIMIT:
        *(int *) data = g->dev->dev_attr.max_graph_opt_class;
        *dataLength = sizeof(int);
        break;
    case NC_RO_GRAPH_VERSION:
        return NC_UNSUPPORTED_FEATURE;
    default:
        mvLog(MVLOG_ERROR, "There is no such option in class 0");
        return NC_INVALID_PARAMETERS;
    }
    return NC_OK;
}
static ncStatus_t getGraphOptionClass1(struct _graphPrivate_t *g,
                                        ncGraphOptionsClass1_t option,
                                        void *data, unsigned int* dataLength){
    switch (option) {
    case NC_RW_GRAPH_EXECUTORS_NUM:
        *(int*) data = g->executors_number;
        *dataLength = sizeof(int);
				mvLog(MVLOG_INFO, "executors_number %d ",g->executors_number);
				return NC_OK;
        //break;
    case NC_RW_GRAPH_BATCH_SIZE :
        *(int *) data = g->batch_size;
         *dataLength = sizeof(int);
         return NC_OK;
    case NC_RW_GRAPH_RUNTIME_CONFIG:
        return NC_UNSUPPORTED_FEATURE;
    default:
        mvLog(MVLOG_ERROR, "There is no such option in class 1");
        return NC_INVALID_PARAMETERS;
    }
}

ncStatus_t ncGraphGetOption(struct graphHandle_t *graphHandle,
                            ncOptionClass_t class,
                            int option, void *data,
                            unsigned int *dataLength) {
	if (!graphHandle || !data || !dataLength) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
		return NC_INVALID_PARAMETERS;
	}
	struct _graphPrivate_t *g = graphHandle->private_data;
	if (g->dev != NULL && class > g->dev->dev_attr.max_graph_opt_class) {
	    mvLog(MVLOG_ERROR, "This device FW does not support NC_OPTION_CLASS%d", class);
	    return NC_UNAUTHORIZED;
	}
	pthread_mutex_lock(&globalMutex);
	if (findGraph(g)) {
        mvLog(MVLOG_ERROR, "This graph is corrupt");
				mvLog(MVLOG_INFO, "This graph is corrupt");
		pthread_mutex_unlock(&globalMutex);
		return NC_INVALID_PARAMETERS;
	}

	pthread_mutex_unlock(&globalMutex);
	ncStatus_t rc;
    switch (class) {
    case NC_OPTION_CLASS0:
        rc = getGraphOptionClass0(g, option, data, dataLength);
        break;
    case NC_OPTION_CLASS1:
        rc = getGraphOptionClass1(g, option, data, dataLength);
        break;
    case NC_OPTION_CLASS2:
        rc = getGraphOptionClass2(g, option, data, dataLength);
        break;
    case NC_OPTION_CLASS3:
        rc = getGraphOptionClass3(g, option, data, dataLength);
        break;
    default:
        mvLog(MVLOG_ERROR, "There is no such option class");
        rc = NC_INVALID_PARAMETERS;
        break;
    }
    return rc;
}

ncStatus_t ncGlobalSetOption(int option, const void *data,
			       unsigned int dataLength)
{
	if (!data){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
		return NC_INVALID_PARAMETERS;
	}
	if (dataLength != sizeof(int)){
	    mvLog(MVLOG_ERROR, "The dataLength must be %d", sizeof(int));
	    return NC_INVALID_PARAMETERS;
	}
	switch (option) {
	case NC_RW_LOG_LEVEL:
	    mvLogLevelSet(*(mvLog_t *) data);
		break;
	case NC_RO_API_VER:
	    mvLog(MVLOG_ERROR, "API version is read-only");
	    return NC_UNAUTHORIZED;
	    break;
	default:
        mvLog(MVLOG_ERROR, "No such option");
		return NC_INVALID_PARAMETERS;
	}

	return NC_OK;
}

ncStatus_t ncGlobalGetOption(int option, void *data, unsigned int *dataLength)
{
	if (!data || !dataLength) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
		return NC_INVALID_PARAMETERS;
	}
	switch (option) {
	case NC_RW_LOG_LEVEL:
		*(int *) data = mvLogLevel_ncAPI;
		*dataLength = sizeof(mvLogLevel_ncAPI);
		break;
	case NC_RO_API_VER:
	    return NC_UNSUPPORTED_FEATURE;
	    break;
	default:
        mvLog(MVLOG_ERROR, "No such option");
		return NC_INVALID_PARAMETERS;
	}
	return NC_OK;
}

static ncStatus_t getDeviceOptionClass0(struct _devicePrivate_t *d,
                                          ncDeviceOptionsClass0 option,
                                          void *data, unsigned int* dataLength) {
    ncStatus_t rc = NC_OK;

    switch (option) {
    case NC_RO_DEVICE_THERMAL_STATS:
        rc = getThermalStats(d);
        if (rc) {
            return rc;
        }
        *(float **) data = &d->thermal_stats[1];
        *dataLength = THERMAL_BUFFER_SIZE;
        break;
    case NC_RO_DEVICE_THERMAL_THROTTLING_LEVEL:
        rc = getThermalStats(d); // TODO: thermal stats and thermal throttle are partially merged. We can do both with a single read
        if (rc) {
            return rc;
        }
        d->throttle_happened = d->thermal_stats[0];
        *(int *) data = d->throttle_happened;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_OPTIMISATION_LIST:
        rc = getOptimisationList(d);
        if (rc) {
            return rc;
        }
        *(char **) data = d->optimisation_list;
        *dataLength = OPTIMISATION_LIST_BUFFER_SIZE;
        break;
    case NC_RO_DEVICE_STATE:
        *(int *) data = d->state;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_ALLOCATED_GRAPH_NUM:
        *(int *) data = deviceGetNumberOfGraphs(d);
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_ALLOCATED_FIFO_NUM:
        *(int *) data = deviceGetNumberOfFifos(d);
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_MEMORY_SIZE:
        *(int *) data = d->dev_attr.max_memory;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_MAX_FIFO_NUM:
        *(int *) data = d->dev_attr.max_fifos;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_MAX_GRAPH_NUM:
        *(int *) data = d->dev_attr.max_graphs;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_OPTION_CLASS_LIMIT:
        *(int *) data = d->dev_attr.max_device_opt_class;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_NAME:
        *(char**) data = d->dev_addr;
        *dataLength = strlen(d->dev_addr) + 1;
        break;
    case NC_RO_DEVICE_FW_VER:
        *(int **) data = d->dev_attr.fw_version;
        *dataLength = sizeof(int*);
        break;
    case NC_RO_DEVICE_CURRENT_MEMORY_USED:{
        uint32_t mem;
        if (deviceGetDeviceMemory(d, &mem)){
            rc = NC_ERROR;
            break;
        }
        *(int *) data = mem;
        *dataLength = sizeof(uint32_t);
        break;
    }
    case NC_RO_DEVICE_MVTENSOR_VER:
    case NC_RO_DEVICE_DEBUG_INFO:
        return NC_UNSUPPORTED_FEATURE;
//aks
/*
		case NC_RO_DEVICE_PLATFORM:
        *(int**) data = "2450";
        *dataLength = strlen(data) + 1;
        break;
*/
    default:
        mvLog(MVLOG_ERROR, "No such option");
        return NC_INVALID_PARAMETERS;
    }
    return rc;
}

ncStatus_t ncDeviceSetOption(struct deviceHandle_t *deviceHandle,
                             ncOptionClass_t opClass, int option,
                             const void *data, unsigned int dataLength){
    ncStatus_t rc = NC_OK;

	if (!deviceHandle || !data){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
		return NC_INVALID_PARAMETERS;
	}
    if (dataLength != sizeof(int)){
        mvLog(MVLOG_ERROR, "The dataLength must be %d", sizeof(int));
        return NC_INVALID_PARAMETERS;
    }
	struct _devicePrivate_t *d = deviceHandle->private_data;
	pthread_mutex_lock(&globalMutex);
	if (findDevice(d)) {
        mvLog(MVLOG_ERROR, "This device handle is corrupt");
		pthread_mutex_unlock(&globalMutex);

		return NC_INVALID_PARAMETERS;
	}
	pthread_mutex_unlock(&globalMutex);
	if (opClass > d->dev_attr.max_device_opt_class){
	    mvLog(MVLOG_ERROR, "This device FW does not support NC_OPTION_CLASS%d", opClass);
	    return NC_UNAUTHORIZED;
	}
	switch (opClass) {
	case NC_OPTION_CLASS0:
        mvLog(MVLOG_ERROR, "Class 0 options are read-only");
	    rc = NC_UNAUTHORIZED; // option class 0 consists of read-only value
	    break;
	case NC_OPTION_CLASS1:
	    rc = setDeviceOptionClass1(d, option, data, dataLength);
	    break;
	case NC_OPTION_CLASS2:
	    rc = setDeviceOptionClass2(d, option, data, dataLength);
	    break;
	case NC_OPTION_CLASS3:
	    rc = setDeviceOptionClass3(d, option, data, dataLength);
	    break;
	default:
	    rc = NC_INVALID_PARAMETERS;
	    break;
	}
	return rc;
}


ncStatus_t ncDeviceGetOption(struct deviceHandle_t *deviceHandle,
                             ncOptionClass_t opClass, int option,
                             void *data, unsigned int *dataLength) {
	ncStatus_t rc;

	if (!deviceHandle || !data || !dataLength){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
	    return NC_INVALID_PARAMETERS;
	}
	struct _devicePrivate_t *d = deviceHandle->private_data;
	if (d->dev_attr.max_device_opt_class < opClass) {
		mvLog(MVLOG_ERROR, "This device FW does not support NC_OPTION_CLASS%d", opClass);
		return NC_UNAUTHORIZED;
	}

	pthread_mutex_lock(&globalMutex);
	if (findDevice(d)) {
        mvLog(MVLOG_ERROR, "This device handle is corrupt");
		pthread_mutex_unlock(&globalMutex);
		return NC_INVALID_PARAMETERS;
	}
	pthread_mutex_unlock(&globalMutex);
	switch (opClass) {
	case NC_OPTION_CLASS0:
        rc = getDeviceOptionClass0(d, option, data, dataLength);
	    break;
	case NC_OPTION_CLASS1:
	    rc = getDeviceOptionClass1(d, option, data, dataLength);
	    break;
	case NC_OPTION_CLASS2:
	    rc = getDeviceOptionClass2(d, option, data, dataLength);
	    break;
	case NC_OPTION_CLASS3:
	    rc = getDeviceOptionClass3(d, option, data, dataLength);
	    break;
    default:
        rc = NC_INVALID_PARAMETERS;
        break;
	}
	return rc;
}

static int fifoWriteAccess( struct _fifoPrivate_t* fifo) {
    if (fifo->type == NC_FIFO_HOST_RW || fifo->type == NC_FIFO_HOST_WO) {
        return 1;
    }
    return 0;
}

static int fifoReadAccess( struct _fifoPrivate_t* fifo) {
    if(fifo->type == NC_FIFO_HOST_RW || fifo->type == NC_FIFO_HOST_RO) {
        return 1;
    }
    return 0;
}

ncStatus_t ncFifoInit(ncFifoType_t type, struct fifoHandle_t** fifo) {
    mvLog(MVLOG_INFO, "Init fifo");

    if (!fifo){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }
    static int fifoIdCounter = 0;
    *fifo = (struct fifoHandle_t*) malloc(sizeof(struct fifoHandle_t));
    mvLog(MVLOG_DEBUG, "Segfault without this message means wrong fifo pointer");

    if (!(*fifo)){
        mvLog(MVLOG_ERROR, "Memory allocation failed");
        return NC_OUT_OF_MEMORY;
    }
    struct _fifoPrivate_t* handle = (struct _fifoPrivate_t*) malloc(sizeof(struct _fifoPrivate_t));
    (*fifo)->private_data = handle;
    if (!handle){
        mvLog(MVLOG_ERROR, "Memory allocation failed");
        return NC_OUT_OF_MEMORY;
    }
    //TODO: check malloc rc

    handle->type = type; //default type
    if (handle->type != NC_FIFO_HOST_RW &&
        handle->type != NC_FIFO_HOST_RO &&
        handle->type != NC_FIFO_HOST_WO) {
        return NC_UNSUPPORTED_FEATURE;
    }
    handle->consumer_cnt = 1; //default consumers
    handle->state = NC_FIFO_INITIALIZED;
    pthread_mutex_init(&handle->fifo_mutex, NULL);
    handle->consumed_by_graph = 0;
    handle->write_count = 0;
    handle->user_param_in = NULL;
    handle->user_param_out = NULL;
    handle->api_read_element = 0;
    handle->api_read_adjust = 0;
    handle->id = fifoIdCounter++;
    handle->datatype = NC_FIFO_FP16;
    handle->num_elements = 0;
    snprintf(handle->name, 16, "FIFO%d", handle->id);
    return NC_OK;
}

int pushUserParam(struct _fifoPrivate_t* fH, void* user_param, int isIn)
{
    struct _userParamPrivate_t* new_user_param = calloc(1, sizeof(struct _userParamPrivate_t));
    new_user_param->next = NULL;
    if (!new_user_param) {
        mvLog(MVLOG_ERROR, "calloc failed!");
        return NC_OUT_OF_MEMORY;
    }
    new_user_param->data = user_param;
    if (isIn) {
        new_user_param->next = fH->user_param_in;
        fH->user_param_in = new_user_param;
    }
    else {
        new_user_param->next = fH->user_param_out;
        fH->user_param_out = new_user_param;
    }
    return NC_OK;
}
int popUserParam(struct _fifoPrivate_t* fH, void** user_param, int isIn)
{
    struct _userParamPrivate_t* prev = NULL;
    struct _userParamPrivate_t* curr = NULL;
    if (isIn)
        curr = fH->user_param_in;
    else
        curr = fH->user_param_out;

    if (curr == NULL) {
        *user_param = NULL;
        mvLog(MVLOG_ERROR, "Trying to read user param from an empty queue!");
        return NC_ERROR;
    }

    while (curr->next != NULL)
    {
        prev = curr;
        curr = curr->next;
    }

    *user_param = curr->data;

    if (prev)
        prev->next = NULL;
    else {
        if (isIn)
            fH->user_param_in = NULL;
        else
            fH->user_param_out = NULL;
    }
    free(curr);
    curr = NULL;
    return NC_OK;
}

ncStatus_t ncFifoCreate(struct fifoHandle_t* fifo, struct deviceHandle_t* device,
                        struct ncTensorDescriptor_t* tensor_desc, unsigned int numElem) {
    mvLog(MVLOG_INFO, "Creating fifo");
    if (!fifo || !device || !tensor_desc || !numElem){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }
    if(!tensor_desc->totalSize){
        mvLog(MVLOG_ERROR, "Tensor descriptor is invalid. Total size 0");
        return NC_INVALID_PARAMETERS;
    }
    struct _fifoPrivate_t* handle = fifo->private_data;
    if (handle->state != NC_FIFO_INITIALIZED) {
        mvLog(MVLOG_ERROR, "FIFO is not yet initialized. You can't allocate it");
        return NC_INVALID_PARAMETERS; //TODO: better error code
    }
    struct _devicePrivate_t *d = devices;
    pthread_mutex_lock(&globalMutex);
    while (d) {
        if (d == device->private_data)
            break;
        d = d->next;
    }
    if (!d) {
        pthread_mutex_unlock(&globalMutex);
        mvLog(MVLOG_ERROR, "Device not found!\n");
        return NC_INVALID_PARAMETERS;
    }
    pthread_mutex_unlock(&globalMutex);

    handle->tensor_desc = *tensor_desc;
    //TODO: for now, hardcode to sizeof(fp16), since tensor_descs don't yet have correct dimensions
    int sizeof_td_dt = 2; //tensor_desc->totalSize / (tensor_desc->n * tensor_desc->c * tensor_desc->w * tensor_desc->h);
    handle->output_data = calloc(1, tensor_desc->totalSize * numElem * sizeof(float) / sizeof_td_dt);	// allocate space for fp32
    handle->user_param_in = NULL;
    handle->user_param_out = NULL;
    handle->num_elements = numElem;
    handle->consumers_remaining = handle->consumer_cnt; //default consumers
    handle->dev = d;
    handle->next = NULL;

    if (d->fifos)
        handle->next = d->fifos;
    d->fifos = handle;

    graphMonCommand_t cmd;
    cmd.cmdClass = GRAPH_MON_CLASS_BUFFER_CMD;
    cmd.cmd.buffCmd.type  = BUFFER_ALLOCATE_CMD;
    struct tensorDescriptor_t privateDesc;
    privateDesc.c = tensor_desc->c;
    privateDesc.n = tensor_desc->n;
    privateDesc.h = tensor_desc->h;
    privateDesc.w = tensor_desc->w;
    privateDesc.totalSize = tensor_desc->totalSize;
    cmd.cmd.buffCmd.desc  = privateDesc;
    cmd.cmd.buffCmd.elemCnt = numElem;
    strncpy(cmd.cmd.buffCmd.name, handle->name, 16);
    cmd.cmd.buffCmd.id = handle->id;


    //TODO: buffer size to be transmitted;
    uint32_t writeSize;
    if (fifoWriteAccess(handle)) {
        writeSize = tensor_desc->totalSize * numElem;
        cmd.cmd.buffCmd.writeChannel = 1;
    } else {
        cmd.cmd.buffCmd.writeChannel = 0;
        writeSize = 8; // no write permission on this buffer, so we shouldn't bother allocating buffer on the device
    }
    if (fifoReadAccess(handle)) {
        cmd.cmd.buffCmd.readChannel = 1;
    } else {
        cmd.cmd.buffCmd.readChannel = 0;
    }
    streamId_t streamId = XLinkOpenStream(d->usb_link->linkId, handle->name, writeSize);
    if (streamId == INVALID_STREAM_ID) {
        mvLog(MVLOG_WARN, "can't open stream\n");
        return NC_ERROR;
    }
    handle->streamId = streamId;
    pthread_mutex_lock(&d->graph_streamm);

    if (sendGraphMonitorRequest(d->graph_monitor_stream_id, &cmd)) {
        pthread_mutex_unlock(&d->graph_streamm);
        mvLog(MVLOG_WARN, "can't send command\n");
        return NC_ERROR;
    }
    if (checkGraphMonitorResponse(d->graph_monitor_stream_id)) {
        pthread_mutex_unlock(&d->graph_streamm);
        mvLog(MVLOG_WARN, "myriad NACK\n");
        return NC_ERROR;
    }
    pthread_mutex_unlock(&d->graph_streamm);

    handle->state = NC_FIFO_CREATED;
    return NC_OK;

}

ncStatus_t ncFifoDelete(struct fifoHandle_t* fifo) {
    if (!fifo)
        return NC_INVALID_PARAMETERS;

    struct _fifoPrivate_t* handle = fifo->private_data;

    if (handle->state != NC_FIFO_CREATED) {
        mvLog(MVLOG_ERROR, "FIFO is not yet created.");
        return NC_INVALID_PARAMETERS; //TODO: better error code
    }
    //First write to the fifo to stop it's thread
    if (fifoWriteAccess(handle)) {
        int msg = 0xdead;
        if(XLinkWriteData(handle->streamId, &msg, sizeof(&msg)) != 0)
        {
            mvLog(MVLOG_ERROR, "Failed to write to fifo before deleting it!");
            return NC_ERROR;
        }
    }

    graphMonCommand_t cmd;
    cmd.cmdClass = GRAPH_MON_CLASS_BUFFER_CMD;
    cmd.cmd.buffCmd.type  = BUFFER_DEALLOCATE_CMD;
    cmd.cmd.buffCmd.id = handle->id;

    struct _devicePrivate_t *d = handle->dev;
    pthread_mutex_lock(&d->graph_streamm);
    if (sendGraphMonitorRequest(d->graph_monitor_stream_id, &cmd)) {
        pthread_mutex_unlock(&d->graph_streamm);
        mvLog(MVLOG_WARN, "can't send command\n");
        return NC_ERROR;
    }
    if (checkGraphMonitorResponse(d->graph_monitor_stream_id)) {
        pthread_mutex_unlock(&d->graph_streamm);
        mvLog(MVLOG_WARN, "myriad NACK\n");
        return NC_ERROR;
    }
    pthread_mutex_unlock(&d->graph_streamm);

    pthread_mutex_lock(&d->dev_data_m);
    if (deallocateFifo(handle)) {
        pthread_mutex_unlock(&d->dev_data_m);
        return NC_INVALID_PARAMETERS;
    }
    pthread_mutex_unlock(&d->dev_data_m);

    free(fifo->private_data);
    free(fifo);
    return NC_OK;

}

ncStatus_t ncFifoWriteElem(struct fifoHandle_t* fifo, const void *inputTensor,
                           struct ncTensorDescriptor_t *inputDesc, void *userParam) {
    if (!fifo)
        return NC_INVALID_PARAMETERS;
    struct _fifoPrivate_t* handle = (struct _fifoPrivate_t*) fifo->private_data;
    if (!fifoWriteAccess(handle)) {
        return NC_UNAUTHORIZED;
    }
    //default to the FIFO descriptor
    if (inputDesc == NULL){
        inputDesc = &handle->tensor_desc;
    }
    if (inputDesc->totalSize > handle->tensor_desc.totalSize){
        return NC_INVALID_PARAMETERS; // the tensor size given is bigger than the size supported by the FIFOs
    }
    unsigned int inputTensorLength = inputDesc->totalSize;
    // Convert fp32 to fp16
    if (handle->datatype == NC_FIFO_FP32){
        unsigned char *inputTensorFP16 = malloc(inputTensorLength);
        //TODO: for now, hardcode to sizeof(fp16), since tensor_descs don't yet have correct dimensions
        int sizeof_td_dt = 2; //inputDesc->totalSize / (inputDesc->n * inputDesc->c * inputDesc->w * inputDesc->h);
        unsigned int cnt = inputTensorLength / sizeof_td_dt;
        floattofp16(inputTensorFP16, (float *)inputTensor, cnt);
        inputTensor = (void *)inputTensorFP16;
    }
    if(XLinkWriteData(handle->streamId, inputTensor, inputTensorLength) != 0 )
    {
        if (handle->datatype == NC_FIFO_FP32)
            free(inputTensor);
        return NC_ERROR;
    }
    if (handle->datatype == NC_FIFO_FP32)
        free(inputTensor);
    pthread_mutex_lock(&handle->fifo_mutex);
    int rc = pushUserParam(handle, userParam , 1);
    if(rc != NC_OK) {
        pthread_mutex_unlock(&handle->fifo_mutex);
        return rc;
    }
    handle->write_count++;
    pthread_mutex_unlock(&handle->fifo_mutex);

    mvLog(MVLOG_DEBUG, "write count %d num_elements %d userparam %p\n",
            handle->write_count - 1, handle->num_elements,  userParam);
    return NC_OK;

}

ncStatus_t ncFifoReadElem(struct fifoHandle_t* fifo, void **outputData,
                          struct ncTensorDescriptor_t *outputDesc, void **userParam) {
    if (!fifo || !outputData || !outputDesc)
        return NC_INVALID_PARAMETERS;

    struct _fifoPrivate_t* handle = fifo->private_data;
    streamPacketDesc_t * packet;
    if (!fifoReadAccess(handle)){
        return NC_UNAUTHORIZED;
    }
    if (handle->api_read_element != 0){
        return NC_UNAUTHORIZED;
    }
    if (!XLinkReadData(handle->streamId, &packet))
    {
        // Convert fp16 to fp32
        if (handle->datatype == NC_FIFO_FP32){
            // TODO: for now, hardcode to sizeof(fp16), since tensor_descs don't yet have correct dimensions
            int sizeof_td_dt = 2; //outputDesc->totalSize / (outputDesc->n * outputDesc->c * outputDesc->w * outputDesc->h);
            int cnt = packet->length / sizeof_td_dt;
            fp16tofloat(handle->output_data, (unsigned char *)packet->data, cnt);
        }else{
            //TODO: memcpy for now, not needed in future
            memcpy(handle->output_data,packet->data, packet->length); //TODO: may need to keep track of the elements and give different memory each time
       }
        XLinkReleaseData(handle->streamId);
    }

    //As user should see an API read to be the same as Graph read, we need to wirte the element in 2 queues.
    //if we read it here, we will need to remove the element on the device side
    //to avoid sending a message just for this purpose, we can send it at the next trigger which touches this FIFO.

    pthread_mutex_lock(&handle->fifo_mutex);
    handle->api_read_element = 1;//TODO: protect with mutex
    handle->api_read_adjust++;

    handle->consumers_remaining--;
    if (handle->consumers_remaining == 0) {
        handle->api_read_element = 0;
        handle->consumers_remaining = handle->consumer_cnt;
        //no other action required when the element is consumed
    }
    popUserParam(handle, userParam ,0);
    pthread_mutex_unlock(&handle->fifo_mutex);

    *outputData = handle->output_data;
    *outputDesc = handle->tensor_desc;
    mvLog(MVLOG_DEBUG, "num_elements %d userparam %p output length %d\n",
            handle->num_elements,  userParam, outputDesc->totalSize);
    return NC_OK;

}

ncStatus_t ncFifoRemoveElem(struct fifoHandle_t* fifo) {
    if (!fifo)
        return NC_INVALID_PARAMETERS;
    struct _fifoPrivate_t* handle = (struct _fifoPrivate_t*) fifo->private_data;

    return NC_UNSUPPORTED_FEATURE;
}

ncStatus_t ncFifoSetOption(struct fifoHandle_t* fifo, ncFifoOption_t option,
                           const void *data, unsigned int dataLength) {
    if (!fifo)
        return NC_INVALID_PARAMETERS;
    if (fifo->private_data->state != NC_FIFO_INITIALIZED){
        return NC_UNAUTHORIZED; // after allocation, we can't set any option of FIFO
    }
	struct _fifoPrivate_t *f = (struct _fifoPrivate_t*) fifo->private_data;
    switch (option){
    case NC_RW_FIFO_TYPE:
        f->type = *(ncFifoType_t *) data;
        break;
    case NC_RW_FIFO_CONSUMER_COUNT:
        f->consumer_cnt = *(int *) data;
        break;
    case NC_RW_FIFO_DATA_TYPE:
        f->datatype = *(int *) data;
        break;
    case NC_RW_FIFO_DONT_BLOCK:
        return NC_UNSUPPORTED_FEATURE; //TODO: XLink support for this (fill level may be enough for it)
        break;
    case NC_RO_FIFO_CAPACITY:
    case NC_RO_FIFO_READ_FILL_LEVEL:
    case NC_RO_FIFO_WRITE_FILL_LEVEL:

    case NC_RO_FIFO_TENSOR_DESCRIPTOR:
    case NC_RO_FIFO_STATE:
        return NC_UNAUTHORIZED;
        break;
    default:
        return NC_INVALID_PARAMETERS;
        break;
    }
    return NC_OK;
}
ncStatus_t ncFifoGetOption(struct fifoHandle_t* fifo, ncFifoOption_t option,
                           void *data, unsigned int *dataLength) {
    if (!fifo)
        return NC_INVALID_PARAMETERS;
    switch (option){
    case NC_RW_FIFO_TYPE:
        *(ncFifoType_t *) data = fifo->private_data->type;
        *dataLength = sizeof(fifo->private_data->type);
        break;
    case NC_RW_FIFO_CONSUMER_COUNT:
        *(int *) data = fifo->private_data->consumer_cnt;
        *dataLength = sizeof(fifo->private_data->consumer_cnt);
        break;
    case NC_RW_FIFO_DATA_TYPE:
        *(int *) data = fifo->private_data->datatype;
        *dataLength = sizeof(fifo->private_data->datatype);
        break;
    case NC_RO_FIFO_CAPACITY:
        *(int *) data = fifo->private_data->num_elements;
        *dataLength = sizeof(fifo->private_data->num_elements);
        break;
    case NC_RO_FIFO_TENSOR_DESCRIPTOR:
        if (fifo->private_data->state != NC_FIFO_CREATED)
            return NC_UNAUTHORIZED; // before allocation, tensor_desc is NULL
        *(struct ncTensorDescriptor_t*) data = fifo->private_data->tensor_desc;
        *dataLength = sizeof(fifo->private_data->tensor_desc);
        break;
    case NC_RO_FIFO_READ_FILL_LEVEL:
        {
            struct _fifoPrivate_t* fi = fifo->private_data;
            if (fi->type == NC_FIFO_DEVICE_ONLY ||
                !fifoReadAccess(fi))
                return NC_UNAUTHORIZED;

            *dataLength = sizeof(int);
            if (fi->state != NC_FIFO_CREATED) {
                *(int*) data = 0;
                break;
            }
            int fillLevel;
            if (XLinkGetFillLevel(fi->streamId, 0 ,&fillLevel) == X_LINK_SUCCESS) {
                *(int*) data =  (fillLevel/fi->tensor_desc.totalSize);
            } else {
                return NC_UNAUTHORIZED;
            }

            break;
        }
    case NC_RO_FIFO_WRITE_FILL_LEVEL:
        {
            struct _fifoPrivate_t* fi = fifo->private_data;
            if (fi->type == NC_FIFO_DEVICE_ONLY || !fifoWriteAccess(fi))
                return NC_UNAUTHORIZED;

            *dataLength = sizeof(int);
            if (fi->state != NC_FIFO_CREATED) {
                *(int*) data = 0;
                break;
            }
            int fillLevel;
            if (XLinkGetFillLevel(fi->streamId,  1 ,&fillLevel) == X_LINK_SUCCESS) {
                *(int*) data =  (fillLevel/fi->tensor_desc.totalSize);
            } else {
                return NC_ERROR;
            }

            break;
        }
    case NC_RW_FIFO_DONT_BLOCK:
        return NC_UNSUPPORTED_FEATURE; //TODO: XLink support for this (fill level may be enough for it)
        break;
    case NC_RO_FIFO_STATE:
        *(int*) data = fifo->private_data->state;
        *dataLength = sizeof(int);
        break;
    default:
        return NC_INVALID_PARAMETERS;
        break;
    }
    return NC_OK;
}

static ncStatus_t tensorCompatibility(struct ncTensorDescriptor_t* tens1,
                                      struct ncTensorDescriptor_t* tens2) {
    if(tens1->totalSize != tens2->totalSize ||
                    tens1->n != tens2->n ||
                    tens1->c != tens2->c ||
                    tens1->h != tens2->h ||
                    tens1->w != tens2->w)
        return NC_ERROR;
    return NC_OK;
}

ncStatus_t ncGraphQueueInference(struct graphHandle_t *graphHandle,
                            struct fifoHandle_t** fifoIn,
                            struct fifoHandle_t** fifoOut) {
    mvLog(MVLOG_INFO, "trigger start\n");
    if (!graphHandle || !fifoIn || !fifoIn[0] || !fifoOut || !fifoOut[0])
        return NC_INVALID_PARAMETERS;
    struct _fifoPrivate_t* fi = fifoIn[0]->private_data;
    struct _fifoPrivate_t* fo = fifoOut[0]->private_data;
    struct _graphPrivate_t * g = graphHandle->private_data;
    ncStatus_t rc;
    if (fi->state != NC_FIFO_CREATED|| fo->state != NC_FIFO_CREATED)
        return NC_ERROR; //TODO: could add specific error code
    //WO fifos have no graph access
    if (fo->type == NC_FIFO_HOST_WO){
        //graphs have no access to one of the fifos
        return NC_INVALID_PARAMETERS;
    }
//    if (fi->type == NC_FIFO_HOST_RO && fi->consumer_cnt == 1){
//        // if the FIFO is read only, and there is only one consumer, that should be XLink. Other FIFO types can be used for this usecase.
//        return NC_INVALID_PARAMETERS;
//    }//TODO: we may decide to add this limitation later.
    if (tensorCompatibility(&fi->tensor_desc, &g->input_tensor_desc) != NC_OK ||
                    tensorCompatibility(&fo->tensor_desc, &g->output_tensor_desc)!= NC_OK ) {
        mvLog(MVLOG_WARN, "Input/Output tensor shape is not compatible with graph");
        return NC_ERROR; //TODO: should add specific error code
    }

    graphMonCommand_t cmd;
    cmd.cmdClass = GRAPH_MON_CLASS_GRAPH_CMD;
    cmd.cmd.graphCmd.type = GRAPH_TRIGGER_CMD;
    cmd.cmd.graphCmd.id = g->id;
    cmd.cmd.graphCmd.buffId1 = fi->id;
    cmd.cmd.graphCmd.buffId2 = fo->id;

    void* user_param;
    pthread_mutex_lock(&fi->fifo_mutex);
    fi->consumers_remaining--;
    cmd.cmd.graphCmd.releaseElemBuff1 = fi->api_read_adjust;
    fi->api_read_adjust = 0;
    if (fi->consumer_cnt == 0) {
        if (!fi->api_read_element && fifoReadAccess(fi)) {//the element was entirely consumed by graphs. This means we need to free it up from XLink
            streamPacketDesc_t* packet;
            XLinkReadData(fi->streamId, &packet);
            XLinkReleaseData(fi->streamId);
        }
        fi->consumers_remaining = fi->consumer_cnt;
        fi->api_read_element = 0;
    }
    popUserParam(fi, &user_param , 1);
    if (fi->write_count <= fi->consumed_by_graph){
        mvLog(MVLOG_WARN, "No point on triggering graph. There are no more elements in the input FIFO");
        pthread_mutex_unlock(&fi->fifo_mutex);
        return NC_UNAUTHORIZED;
    }
    fi->consumed_by_graph++;
    pthread_mutex_unlock(&fi->fifo_mutex);

    pthread_mutex_lock(&fo->fifo_mutex);
    cmd.cmd.graphCmd.releaseElemBuff2 = fo->api_read_adjust;
    fo->api_read_adjust = 0;
    rc = pushUserParam(fo, user_param , 0);
    if(rc != NC_OK) {
        pthread_mutex_unlock(&fo->fifo_mutex);
        return rc;
    }
    fo->write_count++;
    pthread_mutex_unlock(&fo->fifo_mutex);

    pthread_mutex_lock(&g->dev->graph_streamm);

    if(sendGraphMonitorRequest(g->dev->graph_monitor_stream_id, &cmd)) {
        pthread_mutex_unlock(&g->dev->graph_streamm);
        mvLog(MVLOG_WARN, "Can't send trigger request");
        return NC_ERROR;
    }
    if(checkGraphMonitorResponse(g->dev->graph_monitor_stream_id)) {
        pthread_mutex_unlock(&g->dev->graph_streamm);
        return NC_ERROR;
    }
    pthread_mutex_unlock(&g->dev->graph_streamm);

    mvLog(MVLOG_INFO, "trigger end\n");
    return NC_OK;
}

ncStatus_t ncGraphQueueInferenceWithFifoElem(struct graphHandle_t *graphHandle,
                                        struct fifoHandle_t** fifoIn,
                                        struct fifoHandle_t** fifoOut,
                                        const void *inputTensor,
                                        struct ncTensorDescriptor_t *inputDesc,
                                        void *userParam) {
	ncStatus_t rc = ncFifoWriteElem(fifoIn[0], inputTensor, inputDesc, userParam);
	if (rc != NC_OK)
		return rc;

	return ncGraphQueueInference(graphHandle, fifoIn, fifoOut);
}
