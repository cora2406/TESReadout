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
#define NUMEVENTS 1000
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
    ArrayXf samples = (int_samples.cast<float>() / 1000).head(size);
    float energy = (samples * filters.topLeftCorner(size,1)).sum();

    #if FILTERS > 1
    temp = 1/((samples + 20 * ArrayXf::Constant(size,52))); // linear transformation
    energy = energy + (temp * filters.block(0,1,size,1)).sum();
    #endif

    #if FILTERS > 2
    temp = 1/((samples + 40 * ArrayXf::Constant(size,25))); // linear transformation
    energy = energy + (temp * filters.block(0,2,size,1)).sum();
    #endif

    #if FILTERS > 3
    temp = 1/((samples - 32 * ArrayXf::Constant(size,0.45))); // linear transformation
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

int make_data(ArrayXXi *samples, int new_samples = -1, int start_index=0){
    //fills the array with random data. can specify width
    // returns new write index
        if (new_samples == -1){
        new_samples = (*samples).cols();
        cout << "size : " << new_samples <<endl;
        }

        if (start_index+new_samples >= (*samples).size()){
            int partial_samples = (*samples).size()-start_index;
            (*samples).block(0,start_index,NPIXEL,partial_samples) = ArrayXXi::Random(NPIXEL, partial_samples);
            partial_samples = new_samples-partial_samples;
            (*samples).block(0,0,NPIXEL,partial_samples) = ArrayXXi::Random(NPIXEL, partial_samples);
        }
        else {
            (*samples).block(0,start_index,NPIXEL, new_samples) = ArrayXXi::Random(NPIXEL, new_samples);
        }

        return (start_index+new_samples)%((*samples).size());
}


int main(int argc, char** argv)
{
    ArrayXXi samples(NPIXEL, MAXSAMPLESIZE);
    std::vector<uint8_t> hits[NPIXEL];
    for (int i=0; i<NPIXEL; i++) {
        hits[i].resize(MAXSAMPLESIZE);
    }
    make_events(hits, NUMEVENTS);
    int write_index = make_data(&samples);
    int read_index=0;
    bool wraparound=true;
    ArrayXXf filterGroupA = ArrayXXf::Random(MAXSAMPLESIZE, FILTERS);
    ArrayXXf filterGroupB = ArrayXXf::Random(MAXSAMPLESIZE, FILTERS);
    ArrayXXf selectedFilters(MAXSAMPLESIZE, FILTERS);
    int first_index=0;
    int last_index=100;



    int sum=0;

    auto t1 = std::chrono::high_resolution_clock::now();
        read_index = 3*SAMPLESPERWINDOW;
        for (int i = 3; i<NUMEVENTS-1; i++){
            if (wraparound == false && (write_index - read_index <= 2*WINDOWS_IN_BUFFER)){
                write_index = make_data(&samples, (WINDOWS_IN_BUFFER/2) * SAMPLESPERWINDOW, write_index);
                cout << "Index updated to : " << write_index << endl;
                if (read_index > write_index) { wraparound = true; }
            }
            if (wraparound == true && (samples.size()-(read_index - write_index) <= 2*WINDOWS_IN_BUFFER)){
                write_index = make_data(&samples, (WINDOWS_IN_BUFFER/2) * SAMPLESPERWINDOW, write_index);
                cout << "Index updated to : " << write_index << endl;
            }

            //Increment the data reader
            read_index+= SAMPLESPERWINDOW;
            if (read_index>samples.size()){
                read_index=0;
                wraparound=false;
            }
        }



    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;

    #if VERBOSE > 0
    cout << "The sum is : " << sum << endl;
    cout << "The process took " << fp_ms.count() << "ms" << endl;
    #endif

  //Multiply vectors then sum the coefficients for inner product.

}
