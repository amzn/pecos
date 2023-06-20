CXX=g++
CXXFLAGS=-fopenmp -O3 -std=c++14 -fPIC -DNDEBUG -Wall -DUSEOMP -DVECGRAD -fno-math-errno #-DDETERMINISTIC -DBENCHMARK  -DVERBOSE   -fopt-info-vec-optimized
INCLUDE_FLAGS=-I. -I../../pecos/core/ 
LIB_FLAGS=-lopenblas
ARCHFLAG=-mavx #-march=native

all: fm_train fm_embgen

fm_train: fm_train_driver.cpp 
	${CXX} -o fm_train ${CXXFLAGS} -g fm_train_driver.cpp ${INCLUDE_FLAGS} ${ARCHFLAG} ${LIB_FLAGS}

fm_embgen: fm_generate_embs.cpp
	${CXX} -o fm_embgen ${CXXFLAGS} -g fm_generate_embs.cpp ${INCLUDE_FLAGS} ${ARCHFLAG} ${LIB_FLAGS}

clean:
	rm -rf *.so *.o fm_train fm_embgen
