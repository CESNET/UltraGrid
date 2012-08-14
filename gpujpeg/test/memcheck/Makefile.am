NVCC=@CUDA_INSTALL_PATH@/bin/nvcc
CUDA_MEMCHECK=@CUDA_INSTALL_PATH@/bin/cuda-memcheck

TESTS=memcheck_local.sh memcheck_cuda.sh
check_PROGRAMS = ${TESTS} memcheck

all-local: tests
tests: check-TESTS

memcheck:
	${NVCC} memcheck.cu -o memcheck

memcheck_local.sh: memcheck
	@echo "echo '+-------------------------------------------------------+'" > memcheck_local.sh
	@echo "echo '| Running without cuda memory checking (should pass)... |'" >> memcheck_local.sh
	@echo "echo '+-------------------------------------------------------+'" >> memcheck_local.sh
	@echo './memcheck' >> memcheck_local.sh
	@chmod +x memcheck_local.sh

memcheck_cuda.sh:
	@echo "echo '+-----------------------------------------------------+'" > memcheck_cuda.sh
	@echo "echo '| Running with cuda memory checking (may not pass)... |'" >> memcheck_cuda.sh
	@echo "echo '+-----------------------------------------------------+'" >> memcheck_cuda.sh
	@echo "${CUDA_MEMCHECK} ./memcheck || exit 0" >> memcheck_cuda.sh
	@chmod +x memcheck_cuda.sh
