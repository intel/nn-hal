LOCAL_PATH:= $(call my-dir)

MY_LOCAL_PATH := $(LOCAL_PATH)
include $(LOCAL_PATH)/ie.mk

include $(MY_LOCAL_PATH)/graph-trans.mk
include $(MY_LOCAL_PATH)/myriad_plugin.mk
include $(MY_LOCAL_PATH)/mvnc.mk

include $(MY_LOCAL_PATH)/mkldnn_plugin.mk

include $(MY_LOCAL_PATH)/mkldnn.mk
include $(MY_LOCAL_PATH)/cpu_extension.mk

#include $(MY_LOCAL_PATH)/format_reader.mk
#include $(LOCAL_PATH)/prebuild.mk

#include $(MY_LOCAL_PATH)/cldnn_plugin.mk

#include $(MY_LOCAL_PATH)/cldnn.mk
include $(MY_LOCAL_PATH)/gna_plugin.mk