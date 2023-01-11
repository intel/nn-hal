#!/bin/bash

## Update the BOARD based on the testing requirement. ##
## Currently the following boards are supported:      ##
## - rex                                              ##
BOARD=rex
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
	sub_cmd="ssh root@${DUT} ml_cmdline --nnapi"
	if ! ${sub_cmd} | grep "Status: OK" ; then
		return 1
	else
		return 0
	fi
}

# Function to get DUT IP address from config file based on BOARD.
getBoardAddr() {
	DUT=$(awk -v key=${BOARD} -F "=" 'BEGIN{/key/} {print $2}' boards.ini | tr -d ' ' | sed -r '/^\s*$/d')
	echo ${DUT}
	if [ -z "${DUT}" ]; then
		echo "${red}ERROR: Invalid BOARD \"${BOARD}\" and/or DUT \"${DUT}\".${reset}"
		exit 1
	fi
}

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

elif [ "${ACTION}" = "deploy" ]; then
        getBoardAddr # Get DUT IP address

        cmd="cros_sdk -- cros deploy ${DUT} intel-nnhal"
        runCmd

elif [ "${ACTION}" = "functional" ]; then

	# Run nnapi ml test
	cmd=mlCmdline
	runCmd

	# Run required cts tests
	cmd="ssh root@${DUT} export ANDROID_LOG_TAGS=\"*:f\" && cros_nnapi_cts --gtest_filter=-Validation*:TestGenerated*:TestRandom*:Generated*:UnknownCombinations*"
	runCmd

        # Run subset of nnapi vts 1_0 tests
        cmd="ssh root@${DUT} export ANDROID_LOG_TAGS=\"*:f\" && cros_nnapi_vts_1_0 --gtest_filter=-Validation*:TestGenerated*:TestRandom*:Generated*:UnknownCombinations*"
	runCmd

        # Run subset of nnapi vts 1_1 tests
        cmd="ssh root@${DUT} export ANDROID_LOG_TAGS=\"*:f\" && cros_nnapi_vts_1_1 --gtest_filter=-Validation*:TestGenerated*:TestRandom*:Generated*:UnknownCombinations*"
	runCmd

        # Run subset of nnapi vts 1_2 tests
        cmd="ssh root@${DUT} export ANDROID_LOG_TAGS=\"*:f\" && cros_nnapi_vts_1_2 --gtest_filter=-Validation*:TestGenerated*:TestRandom*:Generated*:UnknownCombinations*"
	runCmd

        # Run subset of  nnapi vts 1_3 tests
	cmd="ssh root@${DUT} export ANDROID_LOG_TAGS=\"*:f\" && cros_nnapi_vts_1_3 --gtest_filter=-Validation*:TestGenerated*:TestRandom*:Generated*:UnknownCombinations*"
	runCmd

elif [ "${ACTION}" = "regression" ]; then
	# Copy test script to DUT
	scp cts-vts.py root@${DUT}:~/

	# Run nnapi cts tests
	cmd="ssh root@${DUT} export ANDROID_LOG_TAGS=\"*:f\" && python cts-vts.py --cts"
	runCmd
	scp root@${DUT}:~/cts_*.csv . # Copy test result to host server
	ssh root@${DUT} rm -f cts_*.csv # Delete test result to save space in DUT

	# Run nnapi vts_1_0 tests
	cmd="ssh root@${DUT} python cts-vts.py --vts10"
	runCmd
        scp root@${DUT}:~/vts10_*.csv . # Copy test result to host server
        ssh root@${DUT} rm -f vts10_*.csv # Delete test result to save space in DUT

        # Run nnapi vts_1_1 tests
        cmd="ssh root@${DUT} python cts-vts.py --vts11"
        runCmd
        scp root@${DUT}:~/vts11_*.csv . # Copy test result to host server
        ssh root@${DUT} rm -f vts11_*.csv # Delete test result to save space in DUT

        # Run nnapi vts_1_2 tests
        cmd="ssh root@${DUT} python cts-vts.py --vts12"
        runCmd
        scp root@${DUT}:~/vts12_*.csv . # Copy test result to host server
        ssh root@${DUT} rm -f vts12_*.csv # Delete test result to save space in DUT

        # Run nnapi vts_1_3 tests
        cmd="ssh root@${DUT} python cts-vts.py --vts13"
        runCmd
        scp root@${DUT}:~/vts13_*.csv . # Copy test result to host server
        ssh root@${DUT} rm -f vts13_*.csv # Delete test result to save space in DUT
fi
