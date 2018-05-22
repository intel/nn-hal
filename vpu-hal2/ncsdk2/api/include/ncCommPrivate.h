///
/// @file
/// @copyright All code copyright Movidius Ltd 2012, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     MVNC host-device communication structures
///


// Includes
// ----------------------------------------------------------------------------
#ifndef _MVNC_COMM_H_
#define _MVNC_COMM_H_

struct tensorDescriptor_t {
    uint32_t n;
    uint32_t c;
    uint32_t w;
    uint32_t h;
    uint32_t totalSize;
};//If this structure is equivalent with the API structure, it is just a coincidence. Don't make any assumptions based on this in the code.

typedef enum {
    GRAPH_MON_CLASS_GRAPH_CMD = 0,
    GRAPH_MON_CLASS_BUFFER_CMD = 1,
    GRAPH_MON_CLASS_GET_CLASS0 = 2,
    GRAPH_MON_CLASS_GET_CLASS1 = 3,
    GRAPH_MON_CLASS_GET_CLASS2 = 4,
    GRAPH_MON_CLASS_GET_CLASS3 = 5,
    GRAPH_MON_CLASS_SET_CLASS0 = 6,
    GRAPH_MON_CLASS_SET_CLASS1 = 7,
    GRAPH_MON_CLASS_SET_CLASS2 = 8,
    GRAPH_MON_CLASS_SET_CLASS3 = 9,
    //TODO: expandable for set/getoption classes
}graphMonClass_t;

typedef enum {
    GRAPH_ALLOCATE_CMD = 0,
    GRAPH_DEALLOCATE_CMD = 1,
    GRAPH_TRIGGER_CMD = 2,

}graphCommandType_t;

typedef enum {
    CLASS0_TIMING_DATA = 0,
    CLASS0_DEBUG_DATA = 1,
}graphOptionClass0_t;
typedef enum {
    CLASS1_GR_NI = 0,
}graphOptionClass1_t;
typedef enum {
    CLASS2_GR_NI = 0,
}graphOptionClass2_t;
typedef enum {
    CLASS3_GR_NI = 0,
}graphOptionClass3_t;

typedef enum {
    BUFFER_ALLOCATE_CMD = 0,
    BUFFER_DEALLOCATE_CMD = 1,

}bufferCommandType_t;

typedef struct {
    graphCommandType_t type;
    uint32_t id;
    char streamName[16];
    uint32_t buffId1;
    uint32_t buffId2;
    uint32_t releaseElemBuff1;
    uint32_t releaseElemBuff2;
    uint32_t executors_number;
    uint8_t laterUse[16];
}graphCommand_t;

typedef struct {
    bufferCommandType_t type;
    char name[16];
    uint32_t id;
    uint32_t elemCnt;
    struct tensorDescriptor_t desc;
    uint8_t readChannel;
    uint8_t writeChannel;
    uint8_t laterUse[10];
}bufferCommand_t;

typedef struct {
    union{
        graphOptionClass0_t c0;
        graphOptionClass1_t c1;
        graphOptionClass2_t c2;
        graphOptionClass3_t c3;
    }type;
    uint32_t id;
}graphOptionSet_t;

typedef struct{
    graphMonClass_t cmdClass;
    union{
        graphCommand_t graphCmd;
        bufferCommand_t buffCmd;
        graphOptionSet_t optionCmd;
    }cmd;
}graphMonCommand_t;

typedef enum {
    CLASS0_OPT_LIST = 0,
    CLASS0_THERMAL_STATS = 1,
    CLASS0_DEVICE_CAPABILITIES = 2,
    CLASS0_DEVICE_USED_MEMORY = 3,
}deviceOptionClass0;

typedef enum {
    CLASS1_DEV_NI
}deviceOptionClass1;
typedef enum {
    CLASS2_START_SHELL = 0,
    CLASS2_SET_LOG_LEVEL_GLOBAL,
    CLASS2_SET_LOG_LEVEL_FATHOM,
    CLASS2_SET_LOG_LEVEL_XLINK,
}deviceOptionClass2; //TODO: move to separate header
typedef enum {
    CLASS3_DEV_NI
}deviceOptionClass3;

typedef struct {
    union{
        deviceOptionClass0 c0;
        deviceOptionClass1 c1;
        deviceOptionClass2 c2;
        deviceOptionClass3 c3;
    }type;
    uint32_t optionClass;
    uint32_t data;
}deviceCommand_t;

typedef struct {
    uint32_t max_graphs;
    uint32_t max_fifos;
    uint32_t max_memory;
    uint32_t max_device_opt_class;
    uint32_t max_graph_opt_class;
    uint32_t max_executors;
    uint32_t fw_version[4];

}deviceCapabilities_t;


#endif
