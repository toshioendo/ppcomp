CC = gcc
CFLAGS = -O3 -g -fopenmp
LDFLAGS = -lm -fopenmp
LIBS = 

APP = pi
OBJS = $(APP).o

all: $(APP)

$(APP): $(OBJS)
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $*.o

clean:
	rm -f *.o
	rm -f $(APP)
	rm -f *~
