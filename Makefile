bloom: main.c bloom.c
	gcc -g -o bloom main.c bloom.c

clean:
	rm bloom