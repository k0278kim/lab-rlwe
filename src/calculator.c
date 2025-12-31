#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <stdint.h>
#include <string.h>
#include "rlwe_sife.h"

#define SEC_LEVEL 1
#define UNKNOWN                16
#define PERF 1
#define DEC_GPU 16 // 4,16

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
        uint32_t (*mpk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])mpk;

        int inputBatch = imageSize[0];
        int inputChannel = imageSize[1];
        int inputWidth = imageSize[2];
        int inputHeight = imageSize[3];

        int channelSplit = ceil((double)inputChannel / (double)SIFE_L);
        // printf("channelSplit: %d / inputChannel: %d / sifel: %d\n", channelSplit, inputChannel, SIFE_L);
        
        int splitted = SIFE_L; // SIFE_L 크기 고정

        int outputWidth = floor((inputWidth - 1) / stride) + 1;
        int outputHeight = floor((inputHeight - 1) / stride) + 1;

        uint32_t m[TERMS][2][SIFE_L] = {0};

        uint32_t (*encryptedImage_t)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N] = (uint32_t (*)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N])encryptedImage;

        int input_W_dot_H = inputWidth * inputHeight;

        uint32_t polyInput[TERMS][2] = {0};

        for (int b = 0; b < inputBatch; b++) {
                for (int h = 0; h < outputHeight; h++) {
                        for (int w = 0; w < outputWidth; w++) {
                                for (int cs = 0; cs < channelSplit; cs++) {
                                        
                                        memset(m, 0, sizeof(m)); // (채널 스플릿 수정 1)

                                        int channels_in_this_split = (cs == channelSplit - 1) ? (inputChannel - cs * SIFE_L) : SIFE_L; // (채널 스플릿 수정 2)

                                        for (int ich = 0; ich < channels_in_this_split; ich++) { // (채널 스플릿 수정 3)
                                                polynomial(polyInput, image[b * inputChannel * input_W_dot_H + (cs * SIFE_L + ich) * input_W_dot_H + inputWidth * h * stride + w * stride]);

                                                for (int poly = 0; poly < TERMS; poly++) {
                                                        for (int s = 0; s < 2; s++) {
                                                                m[poly][s][ich] = polyInput[poly][s];
                                                        }
                                                }
                                        }
                                        rlwe_sife_encrypt_gui((uint32_t*)m, mpk_t, (uint32_t*)encryptedImage_t[b][h][w][cs], 2*TERMS);
                                };
                        }
                }
        }
        // printf("loadsecinput1x1 ended!\n");
}


void convolution1x1_dec16(double* output, uint32_t* secImage, int* imageSize, double* filter, int* filterSize, int stride, uint32_t* msk) {

        int inputBatch = imageSize[0];
        int inputChannel = imageSize[1];
        int inputWidth = imageSize[2];
        int inputHeight = imageSize[3];

        int channelSplit = ceil((double)inputChannel / (double)SIFE_L);
        int splitted = SIFE_L; // SIFE_L 크기 고정

        int filterCount = filterSize[0];
        int filterLength = filterSize[1];

        int outputWidth = floor((inputWidth - 1) / stride) + 1;
        int outputHeight = floor((inputHeight - 1) / stride) + 1;

        uint32_t (*msk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])msk;
        uint32_t (*secImage_t)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N] = (uint32_t (*)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N])secImage;

        mpz_t dy[SIFE_N];
        uint32_t dy2[SIFE_NMODULI][SIFE_N];
        uint32_t* d_y = (uint32_t*)malloc(TERMS*2*TERMS*2*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));

        double term[TERMS*TERMS][4] = {0};

        uint32_t y[TERMS][2][SIFE_L] = {0};
        uint32_t* slicedFilter = (uint32_t*)malloc(1 * SIFE_L * TERMS * 2 * sizeof(uint32_t));
        uint32_t sk_y[TERMS][2][SIFE_NMODULI][SIFE_N] = {0};

        int pointer = 0;
        int test = 0;

        uint32_t polyFilter[TERMS][2] = {0};

        for(int i=0;i<SIFE_N;i++) {
                mpz_init(dy[i]);
        }

        for (int b = 0; b < inputBatch; b++) {
                for (int fc = 0; fc < filterCount; fc++) {
                        printf("batch:%d/%d filter:%d/%d\n", b+1, inputBatch, fc+1, filterCount);
                        for (int j = 0; j < outputHeight; j++) {
                                for (int i = 0; i < outputWidth; i++) {
                                        double outPix = 0;

                                        for (int cs = 0; cs < channelSplit; cs++) {
                                                
                                                int channels_in_this_split = (cs == channelSplit - 1) ? (inputChannel - cs * SIFE_L) : SIFE_L; // (채널 스플릿 수정)

                                                // (채널 스플릿 수정) 이전 데이터 오염 방지
                                                memset(slicedFilter, 0, 1 * SIFE_L * TERMS * 2 * sizeof(uint32_t));
                                                memset(y, 0, sizeof(y));

                                                for (int ich = 0; ich < channels_in_this_split; ich++) { // (채널 스플릿 수정)
                                                        polynomial(polyFilter, filter[fc * filterLength + (cs * SIFE_L + ich)]);
                                                        for (int poly = 0; poly < TERMS; poly++) {
                                                                slicedFilter[poly * splitted + ich] = polyFilter[poly][0];
                                                                slicedFilter[(poly + TERMS) * splitted + ich] = polyFilter[poly][1];
                                                        }
                                                }

                                                for (int ft = 0; ft < TERMS * 2; ft++) {
                                                        int fs = (int)floor((double)ft/(double)TERMS);
                                                        for (int ich = 0; ich < splitted; ich++) {
                                                                y[ft%TERMS][fs][ich] = slicedFilter[ft * splitted + ich];
                                                        }
                                                }

                                                rlwe_sife_keygen_gui((uint32_t*)y, msk_t, (uint32_t*)sk_y, TERMS*2);
                                                
                                                // (채널 스플릿 수정) d_y 버퍼 초기화
                                                memset(d_y, 0, (size_t)TERMS*2*TERMS*2*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));

                                                rlwe_sife_decrypt_gmp_gui3_x16((uint32_t*)secImage_t[b][j][i][cs], (uint32_t*)y, (uint32_t*)sk_y, (uint32_t*)d_y, TERMS*2, TERMS*2);

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
        free(d_y);
}

// (수정됨) dec4 함수 전체 수정
void convolution1x1_dec4(double* output, uint32_t* secImage, int* imageSize, double* filter, int* filterSize, int stride, uint32_t* msk) {

	int inputBatch = imageSize[0];
	int inputChannel = imageSize[1];
	int inputWidth = imageSize[2];
	int inputHeight = imageSize[3];

    // (수정) channelSplit 로직 추가
	int channelSplit = ceil((double)inputChannel / (double)SIFE_L);
    int splitted = SIFE_L; // SIFE_L 크기 고정

	int filterCount = filterSize[0];
	int filterLength = filterSize[1];

	int outputWidth = floor((inputWidth - 1) / stride) + 1;
	int outputHeight = floor((inputHeight - 1) / stride) + 1;

	uint32_t (*msk_t)[SIFE_NMODULI][SIFE_N] = (uint32_t (*)[SIFE_NMODULI][SIFE_N])msk;
    // (수정) secImage_t 캐스팅에 channelSplit 차원 추가
	uint32_t (*secImage_t)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N] = (uint32_t (*)[outputHeight][outputWidth][channelSplit][TERMS][2][SIFE_L+1][SIFE_NMODULI][SIFE_N])secImage;

	mpz_t dy[SIFE_N];
	uint32_t dy2[SIFE_NMODULI][SIFE_N];
	uint32_t* d_y = (uint32_t*)malloc(TERMS*2*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));

	double term[TERMS*TERMS][4] = {0};

	uint32_t y[TERMS][2][SIFE_L] = {0};

    // (수정) slicedFilter 크기를 SIFE_L 기준으로 변경
	uint32_t* slicedFilter = (uint32_t*)malloc(1 * SIFE_L * TERMS * 2 * sizeof(uint32_t));

	uint32_t sk_y[TERMS][2][SIFE_NMODULI][SIFE_N] = {0};

	int pointer = 0;

	uint32_t polyFilter[TERMS][2] = {0};

	for(int i=0;i<SIFE_N;i++){
		mpz_init(dy[i]);
	}

	for (int b = 0; b < inputBatch; b++) {
		for (int fc = 0; fc < filterCount; fc++) {
			// printf("batch:%d/%d filter:%d/%d \n", b+1, inputBatch, fc+1, filterCount);
			for (int j = 0; j < outputHeight; j++) {
				for (int i = 0; i < outputWidth; i++) {
					
                    double outPix = 0;

                    // (수정) cs 루프 추가 (dec16 로직과 동일)
                    for (int cs = 0; cs < channelSplit; cs++) {

                        int channels_in_this_split = (cs == channelSplit - 1) ? (inputChannel - cs * SIFE_L) : SIFE_L;

                        memset(slicedFilter, 0, 1 * SIFE_L * TERMS * 2 * sizeof(uint32_t));
                        memset(y, 0, sizeof(y));

                        for (int ich = 0; ich < channels_in_this_split; ich++) {
                            polynomial(polyFilter, filter[fc * filterLength + (cs * SIFE_L + ich)]);
                            for (int poly = 0; poly < TERMS; poly++) {
                                slicedFilter[poly * splitted + ich] = polyFilter[poly][0];
                                slicedFilter[(poly + TERMS) * splitted + ich] = polyFilter[poly][1];
                            }
                        }

                        for (int ft = 0; ft < TERMS * 2; ft++) {
                            int fs = (int)floor((double)ft/(double)TERMS);
                            for (int ich = 0; ich < splitted; ich++) { // splitted(SIFE_L) 만큼 루프
                                y[ft%TERMS][fs][ich] = slicedFilter[ft * splitted + ich];
                            }
                        }

                        rlwe_sife_keygen_gui((uint32_t*)y, msk_t, (uint32_t*)sk_y, TERMS*2);

                        for (int ft = 0; ft < TERMS; ft++) {
                            for (int fs = 0; fs < 2; fs++) {
                                
                                // (수정) 상태 오염 방지를 위해 d_y 버퍼 초기화
                                memset(d_y, 0, (size_t)TERMS*2*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));

                                // (수정) 올바른 인덱스(b, cs) 사용
                                rlwe_sife_decrypt_gmp_gui3_x4((uint32_t*)secImage_t[b][j][i][cs], (uint32_t*)y[ft][fs], (uint32_t*)sk_y[ft][fs], (uint32_t*)d_y, TERMS*2);
                                
                                for (int it = 0; it < TERMS; it++) {
                                    for (int is = 0; is < 2; is++) {
                                        memcpy(dy2, d_y + (it * 2 + is) * SIFE_NMODULI * SIFE_N, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
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
                    } // cs 루프 종료
					output[pointer++] = outPix;
				}
			}
			// printf("\033[1A");
			// printf("\032[K");
		}
		
	}

	for (int i = 0; i < SIFE_N; i++) {
		mpz_clear(dy[i]);
	}

	free(slicedFilter);
    free(d_y); // (수정) 메모리 누수 방지
}

void convolution1x1(double* output, uint32_t* secImage, int* imageSize, double* filter, int* filterSize, int stride, uint32_t* msk) {
	#if DEC_GPU == 16
		convolution1x1_dec16(output, secImage, imageSize, filter, filterSize, stride, msk);
	#endif
	#if DEC_GPU == 4
        // (수정됨) 이제 dec4 함수 시그니처가 double*을 받으므로 타입 일치
		convolution1x1_dec4(output, secImage, imageSize, filter, filterSize, stride, msk);
	#endif
}