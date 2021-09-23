# Copyright 2020 The Chromium OS Authors. All rights reserved.
# Distributed under the terms of the GNU General Public License v2

EAPI=7

CROS_WORKON_PROJECT=("chromiumos/platform2" "third_party/intel-nnhal-dev" "third_party/intel-openvino-dev")
CROS_WORKON_LOCALNAME=("platform2" "third_party/intel-nnhal-dev" "third_party/intel-openvino-dev")
CROS_WORKON_DESTDIR=("${S}/platform2" "${S}/platform2/intel-nnhal-dev" "${S}/platform2/intel-openvino-dev")
CROS_WORKON_SUBTREE=("common-mk intel-nnhal-dev .gn" "" "")

PLATFORM_SUBDIR="intel-nnhal-dev"

inherit cros-debug cros-workon platform

DESCRIPTION="Intel NNAPI HAL"
HOMEPAGE="https://github.com/intel/nn-hal"

LICENSE="BSD-Google"
KEYWORDS="*"
SLOT="0/0"

RDEPEND="
	chromeos-base/aosp-frameworks-ml-nn
	chromeos-base/intel-openvino
"

DEPEND="
	>=dev-libs/openssl-1.0.1:0
	${RDEPEND}
"
RESTRICT="strip"

src_prepare() {
	append-cxxflags "-g -O2 -ggdb"

	cros_enable_cxx_exceptions
	eapply_user
}

src_configure() {
	if use x86 || use amd64; then
		append-cppflags "-D_Float16=__fp16"
		append-cxxflags "-Xclang -fnative-half-type"
		append-cxxflags "-Xclang -fallow-half-arguments-and-returns"
	fi
	platform_src_configure
}

src_install() {
	dolib.so "${OUT}/lib/libvendor-nn-hal.so"
	dolib.so "${OUT}/lib/libintel_nnhal.so"
	#dostrip -x "${OUT}/lib/libintel_nnhal.so"
}
