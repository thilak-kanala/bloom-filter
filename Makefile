bloom: main.c bloom.c
	gcc -g -o bloom.out main.c bloom.c

clean:
	rm bloom.out