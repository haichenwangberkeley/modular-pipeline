#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--force]" >&2
  exit 1
fi

SANDBOX_DIR="${HHXYY_QUICKFIT_SANDBOX:-${PROJECT_ROOT}/.cache/hhxyy_tools/quickfit-xcheck}"
REFERENCE_ROOT="${HHXYY_REFERENCE_ROOT:-${HOME}/disk/hhxyy}"
SRC_DIR="${SANDBOX_DIR}/src"
BUILD_DIR="${SANDBOX_DIR}/build"
INSTALL_DIR="${SANDBOX_DIR}/install"
LOG_DIR="${SANDBOX_DIR}/logs"
SETUP_FILE="${SANDBOX_DIR}/setup.sh"
VERSIONS_FILE="${SANDBOX_DIR}/versions.txt"

LOCAL_QUICKFIT_SOURCE="${LOCAL_QUICKFIT_SOURCE:-${REFERENCE_ROOT}/statistics-xcheck/tools/quickfit-xcheck/src/quickFit}"
LOCAL_ROOFITEXTENSIONS_SOURCE="${LOCAL_ROOFITEXTENSIONS_SOURCE:-${REFERENCE_ROOT}/statistics-xcheck/tools/quickfit-xcheck/src/RooFitExtensions}"
QUICKFIT_URL="${QUICKFIT_URL:-https://gitlab.cern.ch/atlas_higgs_combination/software/quickFit.git}"
ROOFITEXTENSIONS_URL="${ROOFITEXTENSIONS_URL:-https://gitlab.cern.ch/atlas_higgs_combination/software/RooFitExtensions.git}"
ROOT_VIEW="${ROOT_VIEW:-/cvmfs/sft.cern.ch/lcg/views/dev3/latest/x86_64-el9-gcc14-opt/setup.sh}"
CC_BIN="${CC_BIN:-/opt/cray/pe/gcc-native/14/bin/gcc}"
CXX_BIN="${CXX_BIN:-/opt/cray/pe/gcc-native/14/bin/g++}"
BUILD_JOBS="${BUILD_JOBS:-4}"

mkdir -p "${SRC_DIR}" "${BUILD_DIR}" "${INSTALL_DIR}" "${LOG_DIR}"

if [[ "${FORCE}" -eq 1 ]]; then
  rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"
  mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"
fi

if [[ -x "${INSTALL_DIR}/quickFit/bin/quickFit" && -x "${SETUP_FILE}" && "${FORCE}" -eq 0 ]]; then
  echo "quickFit sandbox already available at ${SANDBOX_DIR}"
  exit 0
fi

if [[ ! -r "${ROOT_VIEW}" ]]; then
  echo "ROOT view not found: ${ROOT_VIEW}" >&2
  exit 1
fi
if [[ ! -x "${CC_BIN}" || ! -x "${CXX_BIN}" ]]; then
  echo "Native GCC toolchain not found: ${CC_BIN} / ${CXX_BIN}" >&2
  exit 1
fi

prepare_source() {
  local name="$1"
  local local_source="$2"
  local remote_url="$3"
  local dest="${SRC_DIR}/${name}"

  if [[ -d "${dest}/.git" && "${FORCE}" -eq 0 ]]; then
    return
  fi
  rm -rf "${dest}"
  if [[ -d "${local_source}/.git" ]]; then
    git clone --no-local "${local_source}" "${dest}" >"${LOG_DIR}/${name}-source.log" 2>&1
  elif [[ -d "${local_source}" ]]; then
    mkdir -p "${dest}"
    cp -a "${local_source}/." "${dest}/"
  else
    git clone --depth 1 "${remote_url}" "${dest}" >"${LOG_DIR}/${name}-source.log" 2>&1
  fi
}

prepare_source "quickFit" "${LOCAL_QUICKFIT_SOURCE}" "${QUICKFIT_URL}"
prepare_source "RooFitExtensions" "${LOCAL_ROOFITEXTENSIONS_SOURCE}" "${ROOFITEXTENSIONS_URL}"

set +u
source "${ROOT_VIEW}" >/dev/null 2>&1
set -u

export CC="${CC_BIN}"
export CXX="${CXX_BIN}"

cmake -S "${SRC_DIR}/RooFitExtensions" \
  -B "${BUILD_DIR}/RooFitExtensions" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/RooFitExtensions" \
  >"${LOG_DIR}/roofitextensions-cmake.log" 2>&1
cmake --build "${BUILD_DIR}/RooFitExtensions" -j "${BUILD_JOBS}" \
  >"${LOG_DIR}/roofitextensions-build.log" 2>&1
cmake --install "${BUILD_DIR}/RooFitExtensions" \
  >"${LOG_DIR}/roofitextensions-install.log" 2>&1

RooFitExtensions_DIR="${INSTALL_DIR}/RooFitExtensions/cmake" \
cmake -S "${SRC_DIR}/quickFit" \
  -B "${BUILD_DIR}/quickFit" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}/quickFit" \
  >"${LOG_DIR}/quickfit-cmake.log" 2>&1
cmake --build "${BUILD_DIR}/quickFit" -j "${BUILD_JOBS}" \
  >"${LOG_DIR}/quickfit-build.log" 2>&1
cmake --install "${BUILD_DIR}/quickFit" \
  >"${LOG_DIR}/quickfit-install.log" 2>&1

cat >"${SETUP_FILE}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
set +u
source "${ROOT_VIEW}" >/dev/null 2>&1
set -u
export QUICKFIT_XCHECK_HOME="${SANDBOX_DIR}"
export RooFitExtensions_DIR="${INSTALL_DIR}/RooFitExtensions/cmake"
export PATH="${INSTALL_DIR}/quickFit/bin:\${PATH}"
export LD_LIBRARY_PATH="${INSTALL_DIR}/quickFit/lib:${INSTALL_DIR}/RooFitExtensions/lib:\${LD_LIBRARY_PATH:-}"
EOF
chmod +x "${SETUP_FILE}"

cat >"${VERSIONS_FILE}" <<EOF
quickFit_source=$(git -C "${SRC_DIR}/quickFit" rev-parse --show-toplevel 2>/dev/null || printf '%s' "${SRC_DIR}/quickFit")
quickFit_commit=$(git -C "${SRC_DIR}/quickFit" rev-parse HEAD 2>/dev/null || printf 'unknown')
RooFitExtensions_source=$(git -C "${SRC_DIR}/RooFitExtensions" rev-parse --show-toplevel 2>/dev/null || printf '%s' "${SRC_DIR}/RooFitExtensions")
RooFitExtensions_commit=$(git -C "${SRC_DIR}/RooFitExtensions" rev-parse HEAD 2>/dev/null || printf 'unknown')
root_config=$(command -v root-config)
root_version=$(root-config --version)
cc=${CC_BIN}
cxx=${CXX_BIN}
EOF

echo "quickFit sandbox ready at ${SANDBOX_DIR}"
echo "Source it with: source ${SETUP_FILE}"
