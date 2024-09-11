CC = nvcc
CFLAGS = -O2 -g
LD = mpicxx
# CUDA_HOME = /apps/t4/rhel9/cuda/12.3.2
# on TSUBAME, CUDA_HOME is set by module load cuda
LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart
LIBS = 

APP = mm
OBJS = $(APP).o

all: $(APP)

$(APP): $(OBJS)
	$(LD) $^ $(LIBS) -o $@ $(LDFLAGS)

%.o : %.cu
	$(CC) $(CFLAGS) -c $< -o $*.o

clean:
	rm -f *.o
	rm -f $(APP)
