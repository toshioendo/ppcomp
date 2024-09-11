APP=mpibcast

all: $(APP)

$(APP): $(APP).c
	mpicc -O -o $@ $<

clean:
	rm -f $(APP)
	rm -f *.o

