[executable]
path=./build/bicgstab-mpi-gpu
arguments=-m data/sample320/in/A.mtx -p data/sample320/in/row_part_2 -g data/sample320/in/is_gpu.txt -o data/sample320/out/X_mpi_gpu.txt -y data/sample320/in/B.txt -x data/sample320/in/X_init.txt -n 20
ask_directory=1
