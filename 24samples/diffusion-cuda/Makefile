CC = nvcc
CFLAGS = -O3 -g 
LDFLAGS = 
LIBS = 

all: diffusion

mm: diffusion.o
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

%.o : %.cu
	$(CC) $(CFLAGS) -c $< -o $*.o

clean:
	rm -f *.o
	rm -f *~
	rm -f diffusion
