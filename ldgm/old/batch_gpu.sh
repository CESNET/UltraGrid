#1
for I in {1..30}
do 
    ./ldgm-encode -m 3072 -k 4096 -p 1024 -f data/data1.csv -g -t matrix-gen/rfc1.bin
done

#2
for I in {1..30}
do 
    ./ldgm-encode -m 2048 -k 4096 -p 1024 -f data/data2.csv -g -t matrix-gen/rfc2.bin
done

#2
for I in {1..30}
do 
    ./ldgm-encode -m 1024 -k 4096 -p 1024 -f data/data3.csv -g -t matrix-gen/rfc3.bin
done

#4
for I in {1..30}
do 
    ./ldgm-encode -m 1536 -k 2048 -p 2048 -f data/data4.csv -g -t matrix-gen/rfc4.bin
done

#5
for I in {1..30}
do 
    ./ldgm-encode -m 1024 -k 2048 -p 2048 -f data/data5.csv -g -t matrix-gen/rfc5.bin
done

#6
for I in {1..30}
do 
    ./ldgm-encode -m 512 -k 2048 -p 2048 -f data/data6.csv -g -t matrix-gen/rfc6.bin
done

#7
for I in {1..30}
do 
    ./ldgm-encode -m 768 -k 1024 -p 4096 -f data/data7.csv -g -t matrix-gen/rfc7.bin
done

#8
for I in {1..30}
do 
    ./ldgm-encode -m 512 -k 1024 -p 4096 -f data/data8.csv -g -t matrix-gen/rfc8.bin
done

#9
for I in {1..30}
do 
    ./ldgm-encode -m 256 -k 1024 -p 4096 -f data/data9.csv -g -t matrix-gen/rfc9.bin
done

#10
for I in {1..30}
do 
    ./ldgm-encode -m 384 -k 512 -p 8192 -f data/data10.csv -g -t matrix-gen/rfc10.bin
done

#11
for I in {1..30}
do 
    ./ldgm-encode -m 256 -k 512 -p 8192 -f data/data11.csv -g -t matrix-gen/rfc11.bin
done

#12
for I in {1..30}
do 
    ./ldgm-encode -m 128 -k 512 -p 8192 -f data/data12.csv -g -t matrix-gen/rfc12.bin
done

#13
for I in {1..30}
do 
    ./ldgm-encode -m 192 -k 256 -p 16384 -f data/data13.csv -g -t matrix-gen/rfc13.bin
done

#14
for I in {1..30}
do 
    ./ldgm-encode -m 128 -k 256 -p 16384 -f data/data14.csv -g -t matrix-gen/rfc14.bin
done

#15
for I in {1..30}
do 
    ./ldgm-encode -m 64 -k 256 -p 16384 -f data/data15.csv -g -t matrix-gen/rfc15.bin
done




