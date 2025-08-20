#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <stdint.h>
#include <string.h>
#include "rlwe_sife.h"

#define SEC_LEVEL 1
#define UNKNOWN		16
#define PERF 1

uint64_t CLOCK1, CLOCK2;

void polynomial(uint32_t terms[TERMS][2], double number) {
	int integ = (int)number;
	double decim = number - integ;
    
    if (number >= 0) {
		terms[0][0] = integ;
		terms[0][1] = 0;
	} else {
		terms[0][0] = 0;
		terms[0][1] = integ * -1;
	}

    for (int t = 1; t < TERMS; t++) {
        decim = (decim - (int)decim) * UNKNOWN;

        if (number >= 0) {
            terms[t][0] = (int)fabs(decim);
            terms[t][1] = 0;
        } else {
            terms[t][0] = 0;
            terms[t][1] = (int)fabs(decim);
        }
    }
}

void makeKeys(uint32_t* mpk, uint32_t* msk) {
	uint32_t (*mpk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])mpk;
	uint32_t (*msk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])msk;
	rlwe_sife_setup(mpk_t, msk_t);
}

void loadSecInput1x1(uint32_t* encryptedImage, double* image, int* imageSize, int stride, uint32_t* mpk) {
	#ifdef PERF
		uint64_t CLOCK3;
		uint64_t CLOCK_SUM_1 = 0;
		CLOCK1 = 0;
		CLOCK2 = 0;
		CLOCK3 = 0;
		CLOCK1=cpucycles();
	#endif
		
	uint32_t (*mpk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])mpk;

	int inputBatch = imageSize[0];
	int inputChannel = imageSize[1];
	int inputWidth = imageSize[2];
	int inputHeight = imageSize[3];

	int outputWidth = floor((inputWidth - 1) / stride) + 1;
	int outputHeight = floor((inputHeight - 1) / stride) + 1;

	// uint32_t m[SIFE_L] = {0};
	// uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N] = {0};
	uint32_t m[TERMS][2][SIFE_L] = {0};
	// uint32_t c[outputHeight][outputWidth][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N];

	uint32_t (*encryptedImage_t)[outputHeight][outputWidth][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N] = (uint32_t (*)[outputHeight][outputWidth][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N])encryptedImage;

	int input_W_dot_H = inputWidth * inputHeight;

	uint32_t polyInput[TERMS][2] = {0};
	uint32_t* slicedInput = (uint32_t*)malloc(1 * inputChannel * TERMS * 2 * sizeof(uint32_t));

	for (int b = 0; b < inputBatch; b++) {
		for (int h = 0; h < outputHeight; h++) {
			for (int w = 0; w < outputWidth; w++) {
				for (int ich = 0; ich < inputChannel; ich++) {
					polynomial(polyInput, image[b * inputChannel * input_W_dot_H + ich * input_W_dot_H + inputWidth * h * stride + w * stride]);

					for (int poly = 0; poly < TERMS; poly++) {
						for (int s = 0; s < 2; s++) {
							m[poly][s][ich] = polyInput[poly][s];
						}
					}
				}

				#ifdef PERF
					CLOCK3 = cpucycles();
				#endif
				rlwe_sife_encrypt_gui((uint32_t*)m, mpk_t, (uint32_t*)encryptedImage_t[b][h][w], 2*TERMS);
				#ifdef PERF
					CLOCK_SUM_1 += cpucycles() - CLOCK3;
				#endif

			}
		}
	}


	free(slicedInput);
	
	#ifdef PERF
		CLOCK2=cpucycles();
		printf("loadSecInput1x1 ended!:%ld\n", CLOCK2 - CLOCK1);
		printf("enc GPU Function in loadSecInput1x1 ended!:%lf\n", (double)CLOCK_SUM_1 / (inputBatch * outputHeight * outputWidth));
	#endif
	
}

void convolution1x1(double* output, uint32_t* secImage, int* imageSize, double* filter, int* filterSize, int stride, uint32_t* msk) {

	#ifdef PERF
		uint64_t CLOCK3;
		uint64_t CLOCK_SUM = 0;
		CLOCK1 = 0;
		CLOCK2 = 0;
		CLOCK3 = 0;
		CLOCK1=cpucycles();
	#endif

	int inputBatch = imageSize[0];
	int inputChannel = imageSize[1];
	int inputWidth = imageSize[2];
	int inputHeight = imageSize[3];

	int filterCount = filterSize[0];
	int filterLength = filterSize[1];

	int outputWidth = floor((inputWidth - 1) / stride) + 1;
	int outputHeight = floor((inputHeight - 1) / stride) + 1;

	uint32_t (*msk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])msk;
	uint32_t (*secImage_t)[outputHeight][outputWidth][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N] = (uint32_t (*)[outputHeight][outputWidth][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N])secImage;

	mpz_t dy[SIFE_N];
	uint32_t dy2[SIFE_NMODULI][SIFE_N];
	uint32_t* d_y = (uint32_t*)malloc(TERMS*2*TERMS*2*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));

	double term[TERMS*TERMS][4] = {0};

	uint32_t y[TERMS][2][SIFE_L] = {0};

	uint32_t* slicedFilter = (uint32_t*)malloc(1 * filterLength * TERMS * 2 * sizeof(uint32_t));

	uint32_t sk_y[TERMS][2][SIFE_NMODULI][SIFE_N] = {0};

	int pointer = 0;

	uint32_t polyFilter[TERMS][2] = {0};

	for(int i=0;i<SIFE_N;i++) {
		mpz_init(dy[i]);
	}

	for (int b = 0; b < inputBatch; b++) {
		for (int fc = 0; fc < filterCount; fc++) {
			printf("batch:%d/%d filter:%d/%d\n", b+1, inputBatch, fc+1, filterCount);
			for (int j = 0; j < outputHeight; j++) {
				for (int i = 0; i < outputWidth; i++) {
					for (int ich = 0; ich < inputChannel; ich++) {
						polynomial(polyFilter, filter[fc * filterLength + ich]);
						for (int poly = 0; poly < TERMS; poly++) {
							slicedFilter[poly * inputChannel + ich] = polyFilter[poly][0];
							slicedFilter[(poly + TERMS) * inputChannel + ich] = polyFilter[poly][1];
						}
					}

					double outPix = 0;

					for (int ft = 0; ft < TERMS * 2; ft++) {
						int fs = (int)floor((double)ft/(double)TERMS);
						for (int ich = 0; ich < inputChannel; ich++) {
							y[ft%TERMS][fs][ich] = slicedFilter[ft * inputChannel + ich];
						}
					}

					rlwe_sife_keygen_gui((uint32_t*)y, msk_t, (uint32_t*)sk_y, TERMS*2);

					#ifdef PERF
						CLOCK3 = cpucycles();
					#endif

					rlwe_sife_decrypt_gmp_gui3((uint32_t*)secImage_t[b][j][i], (uint32_t*)y, (uint32_t*)sk_y, (uint32_t*)d_y, TERMS*2, TERMS*2);

					#ifdef PERF
						CLOCK_SUM += cpucycles() - CLOCK3;
					#endif

					for (int ft = 0; ft < TERMS; ft++) {
						for (int fs = 0; fs < 2; fs++) {
							for (int it = 0; it < TERMS; it++) {
								for (int is = 0; is < 2; is++) {
									memcpy(dy2, d_y + (it * 2 * TERMS * 2 + is * TERMS * 2 + ft * 2 + fs) * SIFE_NMODULI * SIFE_N, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
									term[TERMS*it+ft][2*is+fs] = round_extract_gmp2(dy2);
									int sign = 1;
									if ((fs == 0 && is == 1) || (fs == 1 && is == 0)) {
										sign = -1;
									}
									if (term[TERMS*it+ft][2*is+fs] != 50241) {
										outPix += (term[TERMS*it+ft][2*is+fs]) / pow(UNKNOWN, (ft+it)) * sign;
									}
								}
							}
							
						}
					}
					
					output[pointer++] = outPix;
				}
			}
			printf("\033[1A");
			printf("\033[2K");
		}
		printf("\n");
	}

	for (int i = 0; i < SIFE_N; i++) {
		mpz_clear(dy[i]);
	}

	free(slicedFilter);

	#ifdef PERF
		CLOCK2=cpucycles();
		printf("convolution1x1 ended!:%ld\n", CLOCK2 - CLOCK1);
		printf("convolution1x1 GPU functions ended!:%lf, (%d, %d, %d, %d)\n", (double)CLOCK_SUM / (inputBatch * filterCount * outputHeight * outputWidth), inputBatch, filterCount, outputHeight, outputWidth);
	#endif
}