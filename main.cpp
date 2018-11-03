#include <iostream>
#include <Dense>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <random>

using namespace Eigen;
using namespace std;

#define FILTERS 4
#define VERBOSE 2
#define REPEAT 3
#define SAMPLESPERWINDOW 100
#define WINDOWS_IN_BUFFER 100
#define MAXSAMPLESIZE WINDOWS_IN_BUFFER*SAMPLESPERWINDOW
#define MAXFILTERS 3
#define NUMEVENTS 500
#define NPIXEL 250

const uint32_t CASE1 = 16843008;
const uint32_t CASE2 = 65792;
const uint32_t CASE3 = 16842753;
const uint32_t CASE4 = 65537;
const uint32_t CASE5 = 16842752;
const uint32_t CASE6 = 65536;

float evaluate_energy(const ArrayXi int_samples, const ArrayXXf filters, int windowsize){

    int size = SAMPLESPERWINDOW*windowsize;
    ArrayXf temp(size);
    ArrayXf data = (int_samples.cast<float>() / 1000).head(size);
    float energy = (data * filters.topLeftCorner(size,1)).sum();

    #if FILTERS > 1
    temp = 1/((data + 20 * ArrayXf::Constant(size,52))); // linear transformation
    energy = energy + (temp * filters.block(0,1,size,1)).sum();
    #endif

    #if FILTERS > 2
    temp = 1/((data + 40 * ArrayXf::Constant(size,25))); // linear transformation
    energy = energy + (temp * filters.block(0,2,size,1)).sum();
    #endif

    #if FILTERS > 3
    temp = 1/((data - 32 * ArrayXf::Constant(size,0.45))); // linear transformation
    energy = energy + (temp * filters.block(0,3,size,1)).sum();
    #endif

    return energy;

}

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

int make_data(ArrayXXi *data, int new_samples = -1, int start_index=0){
    //fills the array with random data. can specify width
    // returns new write index
        if (new_samples == -1){
        new_samples = (*data).cols();
        cout << "size : " << new_samples <<endl;
        }

        if (start_index+new_samples >= (*data).cols()){
            int partial_samples = (*data).cols()-start_index;
            (*data).block(0,start_index,NPIXEL,partial_samples) = ArrayXXi::Random(NPIXEL, partial_samples);
            partial_samples = new_samples-partial_samples;
            (*data).block(0,0,NPIXEL,partial_samples) = ArrayXXi::Random(NPIXEL, partial_samples);
        }
        else {
            (*data).block(0,start_index,NPIXEL, new_samples) = ArrayXXi::Random(NPIXEL, new_samples);
        }

        return (start_index+new_samples)%((*data).cols());
}

float get_energy(std::vector<uint8_t>& hits,
                 const ArrayXXi& data,
                 std::vector<ArrayXf> filters1[],
                 std::vector<ArrayXf> filters2[],
                 int i)
{
    // if photon at t-1 we dont care to differentiate what happened at t-2
    if (hits[i-1]) hits[i-2] = 0;
    uint32_t mask = *reinterpret_cast<uint32_t*>(&hits[i - 2]);
    int length;
    int start;
    ArrayXf* v1;
    ArrayXf* v2;
    // the 6 different cases
    switch(mask) {
        case CASE1:
            // printf("case 1\n");
            start = (i - 1)*SAMPLESPERWINDOW;
            v1 = filters1[0].data();
            v2 = filters2[0].data();
            length = 2;
            break;

        case CASE2:
            // printf("case 2\n");
            start = (i - 1)*SAMPLESPERWINDOW;
            v1 = filters1[1].data();
            v2 = filters2[1].data();
            length = 3;
            break;

        case CASE3:
            // printf("case 3\n");
            start = (i - 2)*SAMPLESPERWINDOW;
            v1 = filters1[2].data();
            v2 = filters2[2].data();
            length = 3;
            break;

        case CASE4:
            // printf("case 4\n");
            start = (i - 2)*SAMPLESPERWINDOW;
            v1 = filters1[3].data();
            v2 = filters2[3].data();
            length = 4;
            break;

        case CASE5:
            // printf("case 5\n");
            start = (i - 2)*SAMPLESPERWINDOW;
            v1 = filters1[4].data();
            v2 = filters2[4].data();
            length = 3;
            break;

        case CASE6:
            // printf("case 6\n");
            start = (i - 2)*SAMPLESPERWINDOW;
            v1 = filters1[5].data();
            v2 = filters2[5].data();
            length = 4;
            break;

        default:
            printf("Unkown case!!\n");
            break;
    }
    return 0; //calculate_energy(start, v1, v2, length*SAMPLESPERWINDOW);
}

int main(int argc, char** argv)
{
    ArrayXXi data(NPIXEL, MAXSAMPLESIZE);
    std::vector<uint8_t> hits[NPIXEL];
    for (int i=0; i<NPIXEL; i++) {
        hits[i].resize(MAXSAMPLESIZE);
    }
    make_events(hits, NUMEVENTS);
    int write_index = make_data(&data);
    int read_index=0;
    bool wraparound=true;

    std::vector<ArrayXf> filters1;
    std::vector<ArrayXf> filters2;
    int lengths[] = {2, 3, 3, 4, 3, 4};
    for (int i=0; i<6; i++) {
        filters1.push_back(ArrayXf::Random(lengths[i]*SAMPLESPERWINDOW));
        filters2.push_back(ArrayXf::Random(lengths[i]*SAMPLESPERWINDOW));;
    }

    int sum=0;

    auto t1 = std::chrono::high_resolution_clock::now();
        read_index = 3*SAMPLESPERWINDOW;
        for (int i = 3; i<NUMEVENTS-1; i++){
            if (wraparound == false && (write_index - read_index <= 2*WINDOWS_IN_BUFFER)){
                write_index = make_data(&data, (WINDOWS_IN_BUFFER/2) * SAMPLESPERWINDOW, write_index);
                cout << "Index updated to : " << write_index << endl;
                if (read_index > write_index) { wraparound = true; }
            }
            if (wraparound == true && (data.cols()-(read_index - write_index) <= 2*WINDOWS_IN_BUFFER)){
                write_index = make_data(&data, (WINDOWS_IN_BUFFER/2) * SAMPLESPERWINDOW, write_index);
                cout << "Index updated to : " << write_index << endl;
            }

//            #pragma omp parallel for shared(data, filters1, filters2, hits) reduction(+:dummy)
//            for (int p=0; p<NPIXEL; p++) {
//                // main loop over time
//                for (int j=2; j<WINDOWS_IN_BUFFER-1; j++) {
//                    if (hits[p][j]) {
//                        float energy = get_energy(hits[p], data.row(p), filters1, filters2, j);
//                        dummy += energy;
//                        // printf("energy %.2f\n", energy);
//                    }
//                }
//            }

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
    cout << "The process took " << fp_ms.count() << "ms" << endl;
    #endif

  //Multiply vectors then sum the coefficients for inner product.

}
