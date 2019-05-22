################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Add inputs and outputs from these tool invocations to the build variables 
CFG_SRCS += \
../mss_mmw.cfg 

CMD_SRCS += \
../mss_mmw_linker.cmd \
../r4f_linker.cmd 

C_SRCS += \
../cli.c \
../cycle_measure.c \
../gtrackAlloc.c \
../gtrackLog.c \
../mss_main.c \
../radarOsal_malloc.c \
../task_app.c \
../task_mbox.c 

GEN_CMDS += \
./configPkg/linker.cmd 

GEN_FILES += \
./configPkg/linker.cmd \
./configPkg/compiler.opt 

GEN_MISC_DIRS += \
./configPkg/ 

C_DEPS += \
./cli.d \
./cycle_measure.d \
./gtrackAlloc.d \
./gtrackLog.d \
./mss_main.d \
./radarOsal_malloc.d \
./task_app.d \
./task_mbox.d 

GEN_OPTS += \
./configPkg/compiler.opt 

OBJS += \
./cli.oer4f \
./cycle_measure.oer4f \
./gtrackAlloc.oer4f \
./gtrackLog.oer4f \
./mss_main.oer4f \
./radarOsal_malloc.oer4f \
./task_app.oer4f \
./task_mbox.oer4f 

GEN_MISC_DIRS__QUOTED += \
"configPkg\" 

OBJS__QUOTED += \
"cli.oer4f" \
"cycle_measure.oer4f" \
"gtrackAlloc.oer4f" \
"gtrackLog.oer4f" \
"mss_main.oer4f" \
"radarOsal_malloc.oer4f" \
"task_app.oer4f" \
"task_mbox.oer4f" 

C_DEPS__QUOTED += \
"cli.d" \
"cycle_measure.d" \
"gtrackAlloc.d" \
"gtrackLog.d" \
"mss_main.d" \
"radarOsal_malloc.d" \
"task_app.d" \
"task_mbox.d" 

GEN_FILES__QUOTED += \
"configPkg\linker.cmd" \
"configPkg\compiler.opt" 

C_SRCS__QUOTED += \
"../cli.c" \
"../cycle_measure.c" \
"../gtrackAlloc.c" \
"../gtrackLog.c" \
"../mss_main.c" \
"../radarOsal_malloc.c" \
"../task_app.c" \
"../task_mbox.c" 


