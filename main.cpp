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
#define NPIXEL 25

const uint32_t CASE1 = 16843008;
const uint32_t CASE2 = 65792;
const uint32_t CASE3 = 16842753;
const uint32_t CASE4 = 65537;
const uint32_t CASE5 = 16842752;
const uint32_t CASE6 = 65536;

float calculate_energy(Eigen::ArrayXf dataf, const Eigen::ArrayXf v1, const Eigen::ArrayXf v2)
{
    const float a = 2.0f;
    float energy = 0.0f;
    //cout << dataf.rows() << "  " << dataf.cols() << endl;
    //cout << (v1).rows() << "  " << (v1).cols() << endl;
    energy = (dataf * v1 + (1.0f / (a - dataf)) * v2).sum();
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
                 const ArrayXi& data,
                 std::vector<ArrayXf> filters1,
                 std::vector<ArrayXf> filters2,
                 int i)
{
    // if photon at t-1 we dont care to differentiate what happened at t-2
    if (hits[i-1]) hits[i-2] = 0;
    uint32_t mask = *reinterpret_cast<uint32_t*>(&hits[i - 2]);
    int length=1;
    int start=0;
    ArrayXf v1;
    ArrayXf v2;
    // the 6 different cases
    switch(mask) {
        case CASE1:
            // printf("case 1\n");
            start = (i - 1)*SAMPLESPERWINDOW;
            v1 = filters1[0];
            v2 = filters2[0];
            length = 2;
            break;

        case CASE2:
            // printf("case 2\n");
            start = (i - 1)*SAMPLESPERWINDOW;
            v1 = filters1[1];
            v2 = filters2[1];
            length = 3;
            break;

        case CASE3:
            // printf("case 3\n");
            start = (i - 2)*SAMPLESPERWINDOW;
            v1 = filters1[2];
            v2 = filters2[2];
            length = 3;
            break;

        case CASE4:
            // printf("case 4\n");
            start = (i - 2)*SAMPLESPERWINDOW;
            v1 = filters1[3];
            v2 = filters2[3];
            length = 4;
            break;

        case CASE5:
            // printf("case 5\n");
            start = (i - 2)*SAMPLESPERWINDOW;
            v1 = filters1[4];
            v2 = filters2[4];
            length = 3;
            break;

        case CASE6:
            // printf("case 6\n");
            start = (i - 2)*SAMPLESPERWINDOW;
            v1 = filters1[5];
            v2 = filters2[5];
            length = 4;
            break;

        default:
            printf("Unkown case!!\n");
            break;
    }
    Eigen::ArrayXf dataf = (data.block(start,0,length*SAMPLESPERWINDOW, 1)).cast<float> ();
    return calculate_energy(dataf, v1, v2);
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
        filters2.push_back(ArrayXf::Random(lengths[i]*SAMPLESPERWINDOW));
    }

    //cout << filters1[4].rows() << endl;

    float sum=0;
    std::chrono::duration<double, std::milli> t_acc;

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

            auto t1_algo = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for shared(data, filters1, filters2, hits) reduction(+:sum)

            for (int p=0; p<NPIXEL; p++) {
                // main loop over time
                for (int j=2; j<WINDOWS_IN_BUFFER-1; j++) {
                    if (hits[p][j]) {
                        float energy = get_energy(hits[p], data.row(p), filters1, filters2, j);
                        sum += energy;
                        // printf("energy %.2f\n", energy);
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
    #endif

  //Multiply vectors then sum the coefficients for inner product.

}
