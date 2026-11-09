idf_component_register(SRCS "main3.c" "llm3.c" "mv_pie3.S"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES esp_partition esp_timer driver spi_flash esp_hw_support esp_rom soc esp_mm
                       LDFRAGMENTS "linker.lf")
target_compile_definitions(${COMPONENT_LIB} PRIVATE LLM_HAVE_PIE=1)
if(DEFINED ENV{FIRST_PIECES})
  target_compile_definitions(${COMPONENT_LIB} PRIVATE FIRST_PIECES=$ENV{FIRST_PIECES})
endif()
if(DEFINED ENV{FW_NAME})
  target_compile_definitions(${COMPONENT_LIB} PRIVATE FW_NAME="$ENV{FW_NAME}")
endif()
# float ops in exactly the order of the host simulator (no fused multiply-add)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off)
