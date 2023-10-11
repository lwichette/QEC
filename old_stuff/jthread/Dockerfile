FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
COPY . /QEC
WORKDIR /QEC

RUN apt-get install make

RUN make

CMD ["./test"]