LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := libmvnc
#LOCAL_SRC_FILES := inference-engine/temp/myriad/lib/libmvnc.so
LOCAL_SRC_FILES := lib/libmvnc.so
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
#LOCAL_MODULE_SUFFIX := .so
#LOCAL_MODULE_CLASS := SHARED_LIBRARIES

#include $(BUILD_PREBUILT)
include $(PREBUILT_SHARED_LIBRARY)
##################################################
include $(CLEAR_VARS)

LOCAL_MODULE := MvNCAPI.mvcmd
LOCAL_SRC_FILES := inference-engine/temp/myriad/lib/mvnc/MvNCAPI.mvcmd
LOCAL_PROPRIETARY_MODULE := true
LOCAL_MODULE_OWNER := intel
LOCAL_MODULE_CLASS := ETC

include $(BUILD_PREBUILT)
##################################################
#include $(CLEAR_VARS)
#LOCAL_MODULE := libpugixml
#LOCAL_PROPRIETARY_MODULE := true
#LOCAL_MODULE_SUFFIX := .a
#LOCAL_MODULE_CLASS := STATIC_LIBRARIES
#LOCAL_MODULE_STEM := libpugixml
#LOCAL_MODULE_OWNER := intel
#LOCAL_MULTILIB := 64
#LOCAL_SRC_FILES_64 := lib/libpugixml.a
#include $(BUILD_PREBUILT)
#################################################
