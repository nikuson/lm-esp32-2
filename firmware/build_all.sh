#!/bin/bash
# builds every firmware variant into prebuilt/<name>/
set -e
. ~/lab/env.sh
cd "$(dirname "$0")"
build() {  # name, defaults...
  local name=$1; shift
  local defs="sdkconfig.defaults"; for d in "$@"; do defs="$defs;sdkconfig.$d"; done
  FW_NAME=$name python $IDF_PATH/tools/idf.py -B build_$name -D SDKCONFIG=build_$name/sdkconfig -D SDKCONFIG_DEFAULTS="$defs" build > build_$name.log 2>&1 || { tail -30 build_$name.log; exit 1; }
  mkdir -p prebuilt/$name
  cp build_$name/bootloader/bootloader.bin build_$name/partition_table/partition-table.bin build_$name/esp_llm15.bin prebuilt/$name/
  echo "built $name: $(stat -c %s build_$name/esp_llm15.bin) bytes"
}
for v in ${@:-qemu quad_c32 quad_c64 quad_c64f120 octal_c32 octal_c64}; do
  case $v in
    qemu) build qemu quad c32 qemu ;;
    quad_c32) build quad_c32 quad c32 ;;
    quad_c64) build quad_c64 quad c64 ;;
    quad_c64f120) build quad_c64f120 quad c64 f120 ;;
    octal_c32) build octal_c32 octal c32 ;;
    octal_c64) build octal_c64 octal c64 ;;
  esac
done
