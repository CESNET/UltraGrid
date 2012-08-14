build: memcheck
	@echo "+-------------------------------------------------------+"
	@echo "| Running without cuda memory checking (should pass)... |"
	@echo "+-------------------------------------------------------+"
	@./memcheck
	@echo ""
	@echo "+-----------------------------------------------------+"
	@echo "| Running with cuda memory checking (may not pass)... |"
	@echo "+-----------------------------------------------------+"
	@/usr/local/cuda/bin/cuda-memcheck ./memcheck

clean:
	rm -f *.o memcheck

memcheck:
	/usr/local/cuda/bin/nvcc memcheck.cu -o memcheck
