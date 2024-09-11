CC = nvcc
CFLAGS = -O3 -g
LDFLAGS = 
LIBS = 

all: mm

mm: mm.o
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

%.o : %.cu
	$(CC) $(CFLAGS) -c $< -o $*.o

clean:
	rm -f *.o
	rm -f *~
	rm -f mm
