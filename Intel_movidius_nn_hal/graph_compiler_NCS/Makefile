
.PHONY: test
test:
	@echo "\n making test "
	g++ test.cpp \
	    Blob.cpp \
			android_stage_dummy.cpp \
			input_stage.cpp \
			stage_logistic.cpp \
			stage_tanh.cpp \
			stage_relu.cpp -o test

clean: clean
	@echo "\nmaking clean";
	rm -f test;
