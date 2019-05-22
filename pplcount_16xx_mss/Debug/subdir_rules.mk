################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
%.oer4f: ../%.c $(GEN_OPTS) | $(GEN_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"C:/ti/ccs901/ccs/tools/compiler/ti-cgt-arm_18.12.1.LTS/bin/armcl" -mv7R4 --code_state=16 --float_support=VFPv3D16 -me -O3 --include_path="C:/Users/Bernard/CCSworkspace_v9/pplcount_16xx_mss" --include_path="C:/ti/mmwave_sdk_02_01_00_04" --include_path="C:/ti/mmwave_sdk_02_01_00_04/packages" --include_path="C:/ti/mmwave_industrial_toolbox_3_3_1/labs/lab0011-pplcount/lab0011_pplcount_pjt/radarDemo" --include_path="C:/ti/mmwave_industrial_toolbox_3_3_1/labs/lab0011-pplcount/lab0011_pplcount_pjt/radarDemo/chains/RadarReceiverPeopleCounting/mmw_PCDemo/gtrack" --include_path="C:/ti/ccs901/ccs/tools/compiler/ti-cgt-arm_18.12.1.LTS/include" --define=SOC_XWR16XX --define=SUBSYS_MSS --define=DOWNLOAD_FROM_CCS --define=DebugP_ASSERT_ENABLED --define=_LITTLE_ENDIAN --define=MMWAVE_L3RAM_SIZE=0x40000 -g --c99 --diag_warning=225 --diag_wrap=off --display_error_number --gen_func_subsections=on --enum_type=int --abi=eabi --obj_extension=.oer4f --preproc_with_compile --preproc_dependency="$(basename $(<F)).d_raw" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

build-1242173169:
	@$(MAKE) --no-print-directory -Onone -f subdir_rules.mk build-1242173169-inproc

build-1242173169-inproc: ../mss_mmw.cfg
	@echo 'Building file: "$<"'
	@echo 'Invoking: XDCtools'
	"C:/ti/xdctools_3_50_08_24_core/xs" --xdcpath="C:/ti/bios_6_73_01_01/packages;" xdc.tools.configuro -o configPkg -t ti.targets.arm.elf.R4F -p ti.platforms.cortexR:IWR16XX:false:200 -r release -c "C:/ti/ccs901/ccs/tools/compiler/ti-cgt-arm_18.12.1.LTS" --compileOptions "--enum_type=int " "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

configPkg/linker.cmd: build-1242173169 ../mss_mmw.cfg
configPkg/compiler.opt: build-1242173169
configPkg/: build-1242173169


