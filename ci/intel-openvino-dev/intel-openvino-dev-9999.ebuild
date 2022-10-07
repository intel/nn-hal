# Copyright 1999-2018 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=7

inherit cmake-utils git-r3 flag-o-matic cros-workon

DESCRIPTION="Intel OpenVino Toolkit"
HOMEPAGE="https://github.com/openvinotoolkit/openvino"

CMAKE_BUILD_TYPE="Debug"
LICENSE="BSD-Google"
KEYWORDS="-* amd64"
IUSE="+clang"
SLOT="0"

CROS_WORKON_PROJECT="third_party/intel-openvino-dev"
CROS_WORKON_LOCALNAME="third_party/intel-openvino-dev"

RDEPEND="
        dev-libs/protobuf
        media-libs/opencv
"

DEPEND="
	${RDEPEND}
"
src_preapre() {
    eapply_user
    cmake-utils_src_prepare
}

src_configure() {
	cros_enable_cxx_exceptions
	append-flags "-Wno-error -frtti -msse4.2 -fvisibility=default -Wno-macro-redefined"
	CPPFLAGS="-I${S}/inference-engine/gna/include -I${S}/inference-engine/omp/include -I${S}/ngraph/src -I${S}/inference_engine/ngraph_ops ${CPPFLAGS}"

	local mycmakeargs=(
                -DCMAKE_INSTALL_PREFIX="/usr/local/"
		-DCMAKE_BUILD_TYPE=Debug
		-DENABLE_CLDNN=OFF
		-DENABLE_GNA=OFF
		-DENABLE_NGRAPH=ON
                -DENABLE_FUNCTIONAL_TESTS=OFF
		-DTHREADING=SEQ
		-DENABLE_MKL_DNN=ON
		-DTARGET_OS="CHROMEOS"
		-DENABLE_OPENCV=OFF
		-DENABLE_SAMPLES=ON
                -DENABLE_TESTS=OFF
		-DBUILD_SHARED_LIBS=ON
                -DENABLE_PROTOC=OFF
                -DNGRAPH_ONNX_IMPORT_ENABLE=OFF
                -DNGRAPH_TEST_UTIL_ENABLE=OFF
                -DENABLE_MYRIAD=OFF
                -DENABLE_VPU=ON
                -DENABLE_SPEECH_DEMO=OFF
                -DGFLAGS_INSTALL_HEADERS=OFF
		-DNGRAPH_ONNX_IMPORT_ENABLE=OFF
                -DNGRAPH_ONNX_FRONTEND_ENABLE=OFF
                -DNGRAPH_PDPD_FRONTEND_ENABLE=OFF
	)
	cmake-utils_src_configure
}

src_install() {
	cmake-utils_src_install

        exeinto /usr/local/bin
        doexe ${S}/bin/intel64/Debug/hello_query_device
}
