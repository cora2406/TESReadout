#include <iostream>
#include <Dense>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <random>

using namespace Eigen;
using namespace std;

#define FILTERS 4
#define VERBOSE 1
#define REPEAT 3
#define SAMPLESPERWINDOW 100
#define WINDOWS_IN_BUFFER 10
#define MAXSAMPLESIZE WINDOWS_IN_BUFFER*SAMPLESPERWINDOW
#define MAXFILTERS 3
#define NUMEVENTS 100000
#define NPIXEL 250

const uint32_t CASE1 = 16843008;
const uint32_t CASE2 = 65792;
const uint32_t CASE3 = 16842753;
const uint32_t CASE4 = 65537;
const uint32_t CASE5 = 16842752;
const uint32_t CASE6 = 65536;

void make_events(vector<uint8_t> hits[NPIXEL], int total, float threshold=0.5){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist;
        for (int p=0; p<NPIXEL; p++) {
            for (int i=0; i<total; i++) {
                double random = dist(gen);
                if (random > threshold) {
                    hits[p][i] = 1;
                }
                else {
                    hits[p][i] = 0;
                }
            }
        }
}

int make_data(MatrixXi *data, int new_samples = -1, int start_index=0){
    //fills the array with random data. can specify width
    // returns new write index
        if (new_samples == -1){
        new_samples = (*data).cols();
        cout << "size : " << new_samples <<endl;
        }

        if (start_index+new_samples >= (*data).cols()){
            int partial_samples = (*data).cols()-start_index;
            (*data).block(0,start_index,NPIXEL,partial_samples) = MatrixXi::Random(NPIXEL, partial_samples);
            partial_samples = new_samples-partial_samples;
            (*data).block(0,0,NPIXEL,partial_samples) = MatrixXi::Random(NPIXEL, partial_samples);
        }
        else {
            (*data).block(0,start_index,NPIXEL, new_samples) = MatrixXi::Random(NPIXEL, new_samples);
        }

        return (start_index+new_samples)%((*data).cols());
}


int main(int argc, char** argv)
{
    MatrixXi data(NPIXEL, MAXSAMPLESIZE);
    std::vector<uint8_t> hits[NPIXEL];
    for (int i=0; i<NPIXEL; i++) {
        hits[i].resize(MAXSAMPLESIZE);
    }
    make_events(hits, MAXSAMPLESIZE);
    int write_index = make_data(&data);
    int read_index=0;
    bool wraparound=true;

    std::vector<VectorXf> filters1;
    std::vector<VectorXf> filters2;
    int lengths[] = {2, 3, 3, 4, 3, 4};
    for (int i=0; i<6; i++) {
        filters1.push_back(VectorXf::Random(lengths[i]*SAMPLESPERWINDOW));
        filters2.push_back(VectorXf::Random(lengths[i]*SAMPLESPERWINDOW));
    }

    VectorXf transformation_constant(1,1);
    transformation_constant(0,0) = 2.0;

    //cout << filters1[4].rows() << endl;

    float sum=0;

    std::chrono::duration<double, std::milli> t_acc(0);

    auto t1 = std::chrono::high_resolution_clock::now();
    read_index = 3*SAMPLESPERWINDOW;
    for (int i = 3; i<NUMEVENTS-1; i+=WINDOWS_IN_BUFFER){
        if (wraparound == false && (write_index - read_index <= 2*WINDOWS_IN_BUFFER)){
            write_index = make_data(&data, (WINDOWS_IN_BUFFER/2) * SAMPLESPERWINDOW, write_index);
            #if VERBOSE > 1
            cout << "Index updated to : " << write_index << endl;
            #endif
            if (read_index > write_index) { wraparound = true; }
        }
        if (wraparound == true && (data.cols()-(read_index - write_index) <= 2*WINDOWS_IN_BUFFER)){
            write_index = make_data(&data, (WINDOWS_IN_BUFFER/2) * SAMPLESPERWINDOW, write_index);
            make_events(hits, MAXSAMPLESIZE);
            #if VERBOSE > 1
            cout << "Index updated to : " << write_index << endl;
            #endif
        }
        int length=1;
        int start=0;
        VectorXf v1;
        VectorXf v2;

        auto t1_algo = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for shared(data, filters1, filters2, hits) private(v1,v2, length, start) reduction(+:sum)

            for (int p=0; p<NPIXEL; p++) {
                // main loop over time
                for (int j=2; j<WINDOWS_IN_BUFFER-1; j++) {
                    if (hits[p][j]) {
                        if (hits[p][j-1]) hits[p][j-2] = 0;
                        uint32_t mask = *reinterpret_cast<uint32_t*>(&(hits[p][j - 2]));
                        // the 6 different cases
                        #if VERBOSE > 2
                        cout << mask << endl;
                        #endif
                        switch(mask) {
                            case CASE1:
                                #if VERBOSE > 2
                                printf("case 1\n");
                                #endif
                                start = (j - 1)*SAMPLESPERWINDOW;
                                v1 = filters1[0];
                                v2 = filters2[0];
                                length = 2;
                                break;

                            case CASE2:
                                #if VERBOSE > 2
                                printf("case 2\n");
                                #endif
                                start = (j - 1)*SAMPLESPERWINDOW;
                                v1 = filters1[1];
                                v2 = filters2[1];
                                length = 3;
                                break;

                            case CASE3:
                                #if VERBOSE > 2
                                printf("case 3\n");
                                #endif
                                start = (j - 2)*SAMPLESPERWINDOW;
                                v1 = filters1[2];
                                v2 = filters2[2];
                                length = 3;
                                break;

                            case CASE4:
                                #if VERBOSE > 2
                                printf("case 4\n");
                                #endif
                                start = (j - 2)*SAMPLESPERWINDOW;
                                v1 = filters1[3];
                                v2 = filters2[3];
                                length = 4;
                                break;

                            case CASE5:
                                #if VERBOSE > 2
                                printf("case 5\n");
                                #endif
                                start = (j - 2)*SAMPLESPERWINDOW;
                                v1 = filters1[4];
                                v2 = filters2[4];
                                length = 3;
                                break;

                            case CASE6:
                                #if VERBOSE > 3
                                printf("case 6\n");
                                #endif
                                start = (j - 2)*SAMPLESPERWINDOW;
                                v1 = filters1[5];
                                v2 = filters2[5];
                                length = 4;
                                break;

                            default:
                                printf("Unkown case!!\n");
                                break;
                        }

                        //VectorXf dataf = data.block<1, Dynamic>(p,start,1,length*SAMPLESPERWINDOW).cast<float> ();
                        //#if VERBOSE > 1
                        //cout << "Size data:" << data.block<1, Dynamic>(p,start,1,length*SAMPLESPERWINDOW).rows() << " x " << data.block<1, Dynamic>(p,start,1,length*SAMPLESPERWINDOW).cols() << endl;
                        //cout << "Size v1:" << v1.rows() << " x " << v1.cols() << endl;
                        //cout << "Size v2:" << v2.rows() << " x " << v2.cols() << endl;
                        //#endif
                        //sum += (dataf * v1 + (1.0f / (2.0f - dataf)) * v2).sum();
                        // printf("energy %.2f\n", energy);
                        sum+= v1.dot(data.block<1, Dynamic>(p,start,1,length*SAMPLESPERWINDOW).cast<float> ()) +
                            v2.dot(3.5*((data.block<1, Dynamic>(p,start,1,length*SAMPLESPERWINDOW).cast<float> ()).colwise() - transformation_constant));
                    }
                }
            }
            t_acc += std::chrono::high_resolution_clock::now() - t1_algo;

            //Increment the data reader
            read_index+= SAMPLESPERWINDOW;
            if (read_index>=data.cols()){
                read_index=0;
                wraparound=false;
            }
            //cout << read_index << endl;
        }



    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;

    #if VERBOSE > 0
    cout << "The sum is : " << sum << endl;
    cout << "The entire process took " << fp_ms.count() << "ms" << endl;
    cout << "The algorithm took " << t_acc.count() << "ms" << endl;
    cout << "Rate of " << double(NUMEVENTS) / t_acc.count() << " kHz" << endl;
    #endif

}
