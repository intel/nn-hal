#!/bin/bash

## Update the BOARD based on the testing requirement. ##
## Currently the following boards are supported:      ##
## - volteer                                          ##
BOARD=volteer
ACTION=$1

red=`tput setaf 1`
green=`tput setaf 2`
blue=`tput setaf 4`
reset=`tput sgr0`

# Function to run command.
runCmd() {
        echo "${blue}RUN: \"${cmd}\"${reset}"
        ${cmd}
        status=$?
        if [ ${status} -eq 0 ]; then
                echo "${green}SUCCESS: \"${cmd}\"${reset}"
        else
                echo "${red}FAIL: \"${cmd}\"${reset}"
                exit 1
        fi
}

# Function to return correct status for ml_cmdline based on output log.
# This function is required because, by default, ml_cmdline
# always returns status 0 irrespective of whether it failed or not.
mlCmdline() {
	sub_cmd="ssh root@${IPADDRESS} ml_cmdline --nnapi"
	if ! ${sub_cmd} | grep "Status: OK" ; then
		return 1
	else
		return 0
	fi
}

# Function to get DUT IP address from config file based on BOARD.
getBoardAddr() {
	IPADDRESS=$(awk -v key=${BOARD} -F "=" 'BEGIN{/key/} {print $2}' boards.ini | tr -d ' ' | sed -r '/^\s*$/d')
	echo ${IPADDRESS}
	if [ -z "${IPADDRESS}" ]; then
		echo "${red}ERROR: Unsupported BOARD=${BOARD}.${reset}"
		exit 1
	fi
}

getBoardAddr

echo ${ACTION}
if [ "${ACTION}" = "build" ]; then

	cmd="cros_sdk --enter"
	runCmd

	cmd="cros_sdk -- cros_workon-${BOARD} start intel-nnhal"
	runCmd

	cmd="cros_sdk USE=\"vendor-nnhal\" -- emerge-${BOARD} intel-nnhal"
	runCmd

	cmd="cros_sdk -- cros_workon_make --board=${BOARD} --install intel-nnhal"
	runCmd

	cmd="cros_sdk -- cros deploy ssh://${IPADDRESS} intel-nnhal"
	runCmd

elif [ "${ACTION}" = "functional" ]; then

	# Run nnapi ml test
	cmd=mlCmdline
	runCmd

	# Run required cts tests
	cmd="ssh root@${IPADDRESS} export ANDROID_LOG_TAGS=\"*:f\" && cros_nnapi_cts --gtest_filter=-Validation*:TestGenerated*:TestRandom*:Generated*:UnknownCombinations*"
	runCmd

        # Run subset of nnapi vts 1_0 tests
        cmd="ssh root@${IPADDRESS} export ANDROID_LOG_TAGS=\"*:f\" && cros_nnapi_vts_1_0 --gtest_filter=-Validation*:TestGenerated*:TestRandom*:Generated*:UnknownCombinations*"
	runCmd

        # Run subset of nnapi vts 1_1 tests
        cmd="ssh root@${IPADDRESS} export ANDROID_LOG_TAGS=\"*:f\" && cros_nnapi_vts_1_1 --gtest_filter=-Validation*:TestGenerated*:TestRandom*:Generated*:UnknownCombinations*"
	runCmd

        # Run subset of nnapi vts 1_2 tests
        cmd="ssh root@${IPADDRESS} export ANDROID_LOG_TAGS=\"*:f\" && cros_nnapi_vts_1_2 --gtest_filter=-Validation*:TestGenerated*:TestRandom*:Generated*:UnknownCombinations*"
	runCmd

        # Run subset of  nnapi vts 1_3 tests
	cmd="ssh root@${IPADDRESS} export ANDROID_LOG_TAGS=\"*:f\" && cros_nnapi_vts_1_3 --gtest_filter=-Validation*:TestGenerated*:TestRandom*:Generated*:UnknownCombinations*"
	runCmd

elif [ "${ACTION}" = "regression" ]; then
	# Copy test script to DUT
	scp cts-vts.py root@${IPADDRESS}:~/

	# Run nnapi cts tests
	cmd="ssh root@${IPADDRESS} export ANDROID_LOG_TAGS=\"*:f\" && python cts-vts.py --cts"
	runCmd
	scp root@${IPADDRESS}:~/cts_*.csv . # Copy test result to host server
	ssh root@${IPADDRESS} rm -f cts_*.csv # Delete test result to save space in DUT

	# Run nnapi vts_1_0 tests
	cmd="ssh root@${IPADDRESS} python cts-vts.py --vts10"
	runCmd
        scp root@${IPADDRESS}:~/vts10_*.csv . # Copy test result to host server
        ssh root@${IPADDRESS} rm -f vts10_*.csv # Delete test result to save space in DUT

        # Run nnapi vts_1_1 tests
        cmd="ssh root@${IPADDRESS} python cts-vts.py --vts11"
        runCmd
        scp root@${IPADDRESS}:~/vts11_*.csv . # Copy test result to host server
        ssh root@${IPADDRESS} rm -f vts11_*.csv # Delete test result to save space in DUT

        # Run nnapi vts_1_2 tests
        cmd="ssh root@${IPADDRESS} python cts-vts.py --vts12"
        runCmd
        scp root@${IPADDRESS}:~/vts12_*.csv . # Copy test result to host server
        ssh root@${IPADDRESS} rm -f vts12_*.csv # Delete test result to save space in DUT

        # Run nnapi vts_1_3 tests
        cmd="ssh root@${IPADDRESS} python cts-vts.py --vts13"
        runCmd
        scp root@${IPADDRESS}:~/vts13_*.csv . # Copy test result to host server
        ssh root@${IPADDRESS} rm -f vts13_*.csv # Delete test result to save space in DUT
fi
