# This scripts runs cts/vts tests and writes results to a timestamped csv file.
# Run "python cts_vts.py --help" for usage details.
import os
import sys
import getopt
import time
import datetime
import subprocess
from joblib import Parallel, delayed
import json

count = 0
total = 0
passed = 0
skipped = 0
failed = 0
error = 0
hang = 0
shared_result = set()
test_suite_binary = ""

def main(argv):
	global total
	global test_suite_binary
	try:
		opts, args = getopt.getopt(argv, "" , ["help", "cts", "vts10", "vts11", "vts12", "vts13"])
	except getopt.GetoptError as err:
		print(err.msg)
		print_usage()

	if not opts: # if opts is empty
		print_usage()

	for opt, arg in opts:
		if opt == "--help":
			print_usage()
		elif opt == "--cts":
			test_suite_binary = "cros_nnapi_cts"
		elif opt == "--vts10":
			test_suite_binary = "cros_nnapi_vts_1_0"
		elif opt == "--vts11":
			test_suite_binary = "cros_nnapi_vts_1_1"
		elif opt == "--vts12":
			test_suite_binary = "cros_nnapi_vts_1_2"
		elif opt == "--vts13":
			test_suite_binary = "cros_nnapi_vts_1_3"
		else:
			print_usage()

	now = datetime.datetime.now()
	print("Start time: {}".format(now))
	begin = time.time()

	tests = get_tests()
	print("Running {0} {1} tests. Please wait..".format(total, test_suite_binary))

	# Run tests concurrently. (n_cpus + 1 + n_jobs) are used.
	Parallel(n_jobs=-2, require='sharedmem')(delayed(run_test)(test) for test in tests)
	#for ltest in list_tests:
	#	run_test(ltest)

	output_file = opt.lstrip("-") + "_" + now.strftime("%Y%m%d_%H%M%S") + ".csv"
	print("Writing results to output file {}. Please wait..".format(output_file))
	with open(output_file, "w") as out:
		heading = "VTS/CTS Command" + "," + "Result"
		print(heading, file=out)
		for result in shared_result:
			print(result, file=out)

	end = time.time()
	summary()

	print("Duration (sec): {}".format(end - begin))

def print_usage():
		print("USAGE    : python cts-vts.py --<options>")
		print("OPTIONS  :")
		print("  --help : Provides usage details")
		print("  --cts  : Runs the cros_nnapi_cts tests")
		print("  --vts10: Runs the cros_nnapi_vts_1_0 tests")
		print("  --vts11: Runs the cros_nnapi_vts_1_1 tests")
		print("  --vts12: Runs the cros_nnapi_vts_1_2 tests")
		print("  --vts13: Runs the cros_nnapi_vts_1_3 tests")
		print("Results are written to output file <options>_<timestamp>.csv")
		sys.exit(0)

def get_tests():
	global test_suite_binary
	global total
	tests = []

	get_tests_file = test_suite_binary + ".json"
	get_tests_cmd = test_suite_binary + " " + "--gtest_list_tests" + " " + "--gtest_output=json:" + get_tests_file
	subprocess.getoutput(get_tests_cmd)
	time.sleep(5)

	with open(get_tests_file) as json_file:
			data = json.load(json_file)
			total = data["tests"]
			for ts in data["testsuites"]:
					test_suite = ts["name"]
					test_name = ""
					for t in ts["testsuite"]:
						test_name = test_suite + "." + t["name"]
						tests.append(test_name)

			if os.path.exists(get_tests_file):
				os.remove(get_tests_file)
			else:
				print("Could not remove file {}. The file does not exist.".format(get_tests_file))

	return tests

def run_test(ltest):
	status = ""
	global count
	global total
	global passed
	global skipped
	global failed
	global error
	global hang
	global test_suite_binary

	test = ltest.strip().split(" ", 1)[0]
	test_cmd = test_suite_binary + " " + " --gtest_filter=" + test

	proc = subprocess.Popen(test_cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
	try:
		out, err = proc.communicate(timeout=10)
	except subprocess.TimeoutExpired:
		proc.kill()
		hang = hang + 1
		status = "HANG"
		result = test_cmd + "," + status
		shared_result.add(result)
		return

	if "[  SKIPPED ]" in str(out):
		skipped = skipped + 1
		status = "SKIPPED"
	elif "[  FAILED ]" in str(out):
		failed = failed + 1
		status = "FAILED"
	elif "[  PASSED  ]" in str(out):
		passed = passed + 1
		status = "PASSED"
	else:
		error = error + 1
		status = "ERROR " + "(" + str(proc.returncode) + ")"

	result = test_cmd + "," + status
	os.system("rm -rf /var/spool/crash/*")

	shared_result.add(result)

	count = count + 1
	if count % 1000 == 0:
                print("Completed [{}/{}] tests..".format(count, total))

	return

def summary():
	print("*******SUMMARY*******")
	print(" PASSED  = %d" %(passed))
	print(" SKIPPED = %d" %(skipped))
	print(" FAILED  = %d" %(failed))
	print(" ERROR   = %d" %(error))
	print(" HANG    = %d" %(hang))
	print(" TOTAL   = %d" %(total))
	print("*********END*********")

if __name__ == "__main__":
	main(sys.argv[1:])
